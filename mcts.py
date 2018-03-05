#import ptvsd 
#ptvsd.enable_attach('mcts')
#ptvsd.wait_for_attach()

import a3c_mcts as a3c


#strategies
def left_only(cur_state):
    return 0

def pi(full_state):
    return None

#end strategies



def get_action(cur_state, strategy_function):
    return  globals()[strategy_function](cur_state)


def rollout(complete_state, num_frames, rollouts, strategy):
    vals = []
    for _ in range(rollouts):
        complete_state.save_state()
        for _ in range(num_frames):
            action = get_action(complete_state, strategy)
            complete_state.take_single_action(action)

        vals.append(complete_state.value)
        complete_state.restore_state()

    return complete_state, sum(vals)/float(rollouts)


start_frames = 100
frames = [100]
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

#print complete_state
