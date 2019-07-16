import tensorflow as tf
import numpy as np
import PVP
import math
from dqn import *
from scipy.spatial.distance import *

env = PVP.game()
winner = ['red  win','blue win','draw']

move_input_size = 42
move_output_size = 9

rotate_input_size = 4
rotate_output_size = 3
 
dis = 0.9
REPLAY_MEMORY = 50000

def ccw(x1, y1, x2, y2, x3, y3):
    temp = x1*y2+x2*y3+x3*y1 - y1*x2-y2*x3-y3*x1
    if temp > 0 :
        return 1
    elif temp < 0:
        return 0
    else:
        return 2

def play_bot(move_mainDQN0, move_mainDQN1, rotate_mainDQN0, rotate_mainDQN1):
    env.pygame_init()
    _, _, sm0, sm1, sr0, sr1 = env.reset()
    done = False

    while not done:
        ar0 = np.argmax(rotate_mainDQN0.predict(sr0))
        ar1 = np.argmax(rotate_mainDQN1.predict(sr1))
        am0 = np.argmax(move_mainDQN0.predict(sm0))
        am1 = np.argmax(move_mainDQN1.predict(sm1))

        done, win, sm0, sm1, sr0, sr1 = env.step(ar0, am0, ar1, am1)
        env.render()
        if done:
            print(winner[win])

def loadDQN(DQN, name, prefix):
    with tf.Session() as sess:
        DQN.load(name+str(prefix))

def training():
    max_episodes = 50000
    
    replay_buffer_m = deque()

    
    replay_buffer_m = deque()

    with tf.Session() as sess:
        move_mainDQN0    = DQN(sess, move_input_size, move_output_size, name="movemain0" )
        move_targetDQN0   = DQN(sess, move_input_size, move_output_size, name="movetarget0" )

        move_mainDQN1     = DQN(sess, move_input_size, move_output_size, name="movemain1" )
        move_targetDQN1   = DQN(sess, move_input_size, move_output_size, name="movetarget1" )

        copy_opsm0 = get_copy_var_ops(dest_scope_name="movetarget0",src_scope_name="movemain0")
        copy_opsm1 = get_copy_var_ops(dest_scope_name="movetarget1",src_scope_name="movemain1")
        

        tf.global_variables_initializer().run()

        for episode in range(max_episodes):

            if episode % 10 == 0:
                env.pygame_init()

            e0 = 1. / ((episode / 10) + 1)
            e1 = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            rm0_sum = 0
            rm1_sum = 0
            _, _, sm0, sm1, sr0, sr1 = env.reset()
            
            while not done:
                if episode % 10 == 0:
                    env.render()
                ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])
                ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])
                am0, am1 = None, None
                # move DQN e-greedy
                if np.random.rand(1) < e0:
                    am0 = np.random.randint(move_output_size, size=1)[0]
                else:
                    am0 = np.argmax(move_mainDQN0.predict(sm0))
            
                if np.random.rand(1) < e1:
                    am1 = np.random.randint(move_output_size, size=1)[0]
                else:
                    am1 = np.argmax(move_mainDQN1.predict(sm1))

                done, win, nsm0, nsm1, nsr0, nsr1 = env.step(ar0, am0, ar1, am1)
                # rm0, rm1 =  cdist([[env.PLAYER0_tmp[0] - env.PLAYER0[0]]],[[env.PLAYER0_tmp[1] - env.PLAYER0[1]]]), cdist([[env.PLAYER1_tmp[0] - env.PLAYER1[0]]],[[env.PLAYER1_tmp[1] - env.PLAYER1[1]]]) * step_count
                
                # rm0_sum += rm0
                # rm1_sum += rm1
                rm0 = -1
                rm1 = -1
                
                        
                replay_buffer_m.append([sm0, am0, rm0, nsm0, done])
                replay_buffer_m.append([sm1, am1, rm1, nsm1, done])
                
                
                if done:
                    if win == 0:
                        for i in range(min(episode,5)):
                            replay_buffer_m[len(replay_buffer_m) - 2*i-1][2] = 10000/step_count
                    else :
                        for i in range(min(episode,5)):
                            replay_buffer_m[len(replay_buffer_m) - 2*i-2][2] = 10000/step_count

                        
                if len(replay_buffer_m) > REPLAY_MEMORY:
                    replay_buffer_m.popleft()

                sm0, sm1, sr0, sr1 = nsm0, nsm1, nsr0, nsr1
                step_count +=1

                if step_count > 10000:
                    win = 2
                    break
                    
            print("episode: {}, winner: {}, {}steps.".format(episode,winner[win],step_count))
            
            if episode % 10 == 4:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer_m, 20)
                    loss, _ = simple_replay_trian(move_mainDQN0, move_targetDQN0, minibatch)

            # if episode % 10 == 4:
            #     for _ in range(50):
            #         minibatch = random.sample(replay_buffer_m_b2, 20)
            #         loss, _ = simple_replay_trian(move_mainDQN0_b2, move_targetDQN0_b2, minibatch)

            # if episode % 10 == 4:
            #     for _ in range(50):
            #         minibatch = random.sample(replay_buffer_m_b3, 20)
            #         loss, _ = simple_replay_trian(move_mainDQN0_b3, move_targetDQN0_b3, minibatch)

            if episode % 10 == 9:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer_m, 20)
                    loss, _ = simple_replay_trian(move_mainDQN1, move_targetDQN1, minibatch)

            # if episode % 10 == 9:
            #     for _ in range(50):
            #         minibatch = random.sample(replay_buffer_m_b2, 20)
            #         loss, _ = simple_replay_trian(move_mainDQN1_b2, move_targetDQN1_b2, minibatch)
                    
            # if episode % 10 == 9:
            #     for _ in range(50):
            #         minibatch = random.sample(replay_buffer_m_b3, 20)
            #         loss, _ = simple_replay_trian(move_mainDQN1_b3, move_targetDQN1_b3, minibatch)

            if episode % 10 == 9:
                #play_bot(move_mainDQN0,move_mainDQN1,rotate_mainDQN0,rotate_mainDQN1)
                # move_mainDQN0_b2.save('./CheckPoints b2/DQN_m0',episode)
                # move_mainDQN1_b2.save('./CheckPoints b2/DQN_m1',episode)
                pass


if __name__ == '__main__':
    training()
    #LoadAndPlay(999)
    #play_game(999)