# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

import torch.nn as nn
from time import sleep
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
from copy import deepcopy
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env', default='Breakout-v0', type=str, help='gym environment')
parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
parser.add_argument('--test', default=False, type=bool, help='test mode sets lr=0, chooses most likely actions')
parser.add_argument('--lstm_steps', default=20, type=int, help='steps to train LSTM over')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
parser.add_argument('--gamma', default=0.99, type=float, help='discount for gamma-discounted rewards')
parser.add_argument('--tau', default=1.0, type=float, help='discount for generalized advantage estimation')
parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
args = parser.parse_args()

args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
if args.test:  args.lr = 0 # don't train in render mode
args.num_actions = gym.make(args.env).action_space.n # get the action space of this game
os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()
        
class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 5 * 5, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs)) ; x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x)) ; x = F.elu(self.conv4(x))
        hx, cx = self.lstm(x.view(-1, 32 * 5 * 5), (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # there's a "step += 1" later
            super.step(closure)

torch.manual_seed(args.seed)
shared_model = NNPolicy(channels=1, num_actions=args.num_actions).share_memory()
shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

info = {k : torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
info['frames'] += shared_model.try_load(args.save_dir)*1e6
if int(info['frames'][0]) == 0: printlog(args,'', end='', mode='w') # clear log file

class CustomException(Exception):
    pass

class FullState(object):
    render = False

    def __init__(self, model = None, env = None, state = None, cx = None, hx = None, episode_length = None, lstm_steps = None):
        self.lstm_steps = lstm_steps
        self.model = model
        self.env = env
        self.state = state
        self.cx = cx
        self.hx = hx
        self.episode_length = episode_length
        self.old_state = None
        self.pi_action = None
        self.value = None

    def restore_state(self):
        if self.old_state is None:
            raise CustomException("Tried to restore a full state without making a copy!")

        #self = FullState(self.old_state) #does this work?
        self.lstm_steps = self.old_state.lstm_steps
        self.model = self.old_state.model
        #old_state.env is actuall just an array
        self.env.restore_state(self.old_state.env)

        self.state = self.old_state.state
        self.cx = self.old_state.cx
        self.hx = self.old_state.hx
        self.episode_length = self.old_state.episode_length

        self.old_state = None


    def save_state(self):
        lstm_steps_copy = deepcopy(self.lstm_steps)
        model_copy = deepcopy(self.model)
        env_copy = self.env.clone_state()
        state_copy = self.state.clone()
        cx_copy = self.cx.clone()
        hx_copy = self.hx.clone()
        episode_length_copy = deepcopy(self.episode_length)

        self.old_state = FullState(model_copy, env_copy, state_copy, cx_copy, hx_copy, episode_length_copy,lstm_steps_copy)

    
    def init_game(self, args, initial_frames):
        self.lstm_steps = args.lstm_steps
        self.env = gym.make(args.env) # make a local (unshared) environment
        self.env.seed(args.seed) 
        self.env = self.env.unwrapped
        torch.manual_seed(args.seed ) # seed everything
    
        self.model = NNPolicy(channels=1, num_actions=args.num_actions)
        self.model.load_state_dict(shared_model.state_dict())
        self.state = torch.Tensor(prepro(self.env.reset())) # get first state
        self.cx = Variable(torch.zeros(1, 256)) # lstm memory vector
        self.hx = Variable(torch.zeros(1, 256)) # lstm activation vector
        self.episode_length = 0
        
        for _ in range(initial_frames):
            self.take_single_action(None)

    def take_single_action(self, action):
        self._run_model()

        if action is None: action = self.pi_action

        self.episode_length += 1
        self.state, reward, done, _ = self.env.step(action)
        if FullState.render: self.env.render()

        self.state = torch.Tensor(prepro(self.state)) 
        reward = np.clip(reward, -1, 1) # reward from game
        done = done or self.episode_length >= 1e4 # keep agent from playing one episode too long
        
        if done: # update shared data. maybe print info.
            self.episode_length = 0
            self.state = torch.Tensor(prepro(self.env.reset()))

    #this should only be called from take single action
    def _run_model(self):
        #dont let the memory vectors get out of hand
        if self.episode_length % self.lstm_steps == 0:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)

        value, logit, (hx, cx) = self.model((Variable(self.state.view(1,1,80,80)), (self.hx, self.cx)))
        logp = F.log_softmax(logit)
        action = logp.max(1)[1].data 

        self.pi_action = action.numpy()[0]
        self.value = value.data.cpu().numpy()[0][0]
        #print(self.value)
