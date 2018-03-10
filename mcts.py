#import ptvsd 
#ptvsd.enable_attach('mcts')
#ptvsd.wait_for_attach()

import a3c_mcts as a3c

a3c.train(0,a3c.args,a3c.info)
'''

start_frames = 1001
frames = []
rollouts = 5


a3c.FullState.render = a3c.args.render
complete_state = a3c.FullState()
complete_state.init_game(a3c.args, start_frames)

print("starting rollouts")

for num_frames in frames:
    print "frames: " + str(num_frames)
    complete_state, value_pi = rollout(complete_state, num_frames, rollouts, "pi")
    print("val pi: " + str(value_pi))
    complete_state, value_left = rollout(complete_state, num_frames, rollouts, "left_only")
    print("val left: " + str(value_pi))


if __name__ == "__main__":
	if sys.version_info[0] > 2:
		mp.set_start_method("spawn") #this must not be in global scope
	elif sys.platform == 'linux' or sys.platform == 'linux2': 
		raise "Must be using Python 3 with linux!" #or else you get a deadlock in conv2d
    
    
	processes = []
	for rank in range(args.processes):
		p = mp.Process(target=train, args=(rank, args, info))
		p.start() ; processes.append(p)
	for p in processes:
		p.join()
    '''

#print complete_state
