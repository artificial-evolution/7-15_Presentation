import tensorflow as tf
import numpy as np
import PVP
import math
from dqn import *

env = PVP.game()
winner = ['red win','blue win','draw']

move_input_size = 5
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

def play_game(prefix):
    
    with tf.Session() as sess:
        DQN_m = DQN(sess, move_input_size, move_output_size, name="movemain1")
        loadDQN(DQN_m,'./CheckPoints/DQN_m1-',prefix)

        env.pygame_init()
        _, _, _, sm1, _, sr1 = env.reset()
        done = False

        while not done:
            ud, lr, x, y = env.player_action()
            ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], x, y)
            ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])
            am0 = ud*3 + lr
            am1 = np.argmax(DQN_m.predict(sm1))
            done, win, _, sm1, _, _ = env.step(ar0, am0, ar1, am1)
            env.render()
            if done:
                print(winner[win])

def loadDQN(DQN, name, prefix):
    with tf.Session() as sess:
        DQN.load(name+str(prefix))

def LoadAndPlay(prefix):
    with tf.Session() as sess:
        move_mainDQN0 = DQN(sess, move_input_size, move_output_size, name="movemain0" )
        move_mainDQN1 = DQN(sess, move_input_size, move_output_size, name="movemain1")
        rotate_mainDQN0 = DQN(sess, rotate_input_size, rotate_output_size, name="rotatemain0" )
        rotate_mainDQN1 = DQN(sess, rotate_input_size, rotate_output_size, name="rotatemain1")

        loadDQN(move_mainDQN0,'./CheckPoints/DQN_m0-',prefix)
        loadDQN(move_mainDQN1,'./CheckPoints/DQN_m1-',prefix)
        loadDQN(rotate_mainDQN0,'./CheckPoints/DQN_r0-',prefix)
        loadDQN(rotate_mainDQN1,'./CheckPoints/DQN_r1-',prefix)

        play_bot(move_mainDQN0,move_mainDQN1)

def training(loading, check_epoisode):
    max_episodes = 5000
    
    replay_buffer_m = deque()
    replay_buffer_r = deque()

    with tf.Session() as sess:
        move_mainDQN = DQN(sess, move_input_size, move_output_size, name="movemain" )
        move_targetDQN = DQN(sess, move_input_size, move_output_size, name="movetarget" )
        rotate_mainDQN = DQN(sess, rotate_input_size, rotate_output_size, name="rotatemain" )
        rotate_targetDQN = DQN(sess, rotate_input_size, rotate_output_size, name="rotatetarget" )

        
        if loading:
            loadDQN(move_mainDQN,'./CheckPoints/DQN_m-',check_epoisode)
            loadDQN(move_targetDQN,'./CheckPoints/DQN_m-',check_epoisode)
            loadDQN(rotate_mainDQN,'./CheckPoints/DQN_r-',check_epoisode)
            loadDQN(rotate_targetDQN,'./CheckPoints/DQN_r-',check_epoisode)
            
            check_epoisode +=1
        else:
            check_epoisode = 0

        copy_opsm0 = get_copy_var_ops(dest_scope_name="movetarget0",src_scope_name="movemain0")
        copy_opsm1 = get_copy_var_ops(dest_scope_name="movetarget1",src_scope_name="movemain1")
        
        # copy_opsr0 = get_copy_var_ops(dest_scope_name="rotatetarget0",src_scope_name="rotatemain0")
        # copy_opsr1 = get_copy_var_ops(dest_scope_name="rotatetarget1",src_scope_name="rotatemain1")

        tf.global_variables_initializer().run()

        for episode in range(check_epoisode, check_epoisode+max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            _, _, sm0, sm1, sr0, sr1 = env.reset()
            
            while not done:
                
                # rotate DQN e-greedy
                # if math.fabs(sr0[1]) > math.pi/2:
                #     ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])
                # elif np.random.rand(1) < e:
                #     ar0 = np.random.randint(rotate_output_size, size=1)[0]
                # else:
                #     ar0 = np.argmax(rotate_mainDQN0.predict(sr0))
                
                # if math.fabs(sr1[1]) > math.pi/2:
                #     ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])
                # elif np.random.rand(1) < e:
                #     ar1 = np.random.randint(rotate_output_size, size=1)[0]
                # else:
                #     ar1 = np.argmax(rotate_mainDQN1.predict(sr1))
                
                ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])
                ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])
                
                # move DQN e-greedy
                if np.random.rand(1) < e:
                    am0 = np.random.randint(move_output_size, size=1)[0]
                else:
                    am0 = np.argmax(move_mainDQN0.predict(sm0
                    ))

                if np.random.rand(1) < e:
                    am1 = np.random.randint(move_output_size, size=1)[0]
                else:
                    am1 = np.argmax(move_mainDQN1.predict(sm1))
                done, win, nsm0, nsm1, nsr0, nsr1 = env.step(ar0, am0, ar1, am1)
                rm0, rm1 = -1, -1
                rr0, rr1 = -1, -1
                
                if done:
                    if win == 0:
                        rm1 = -1000
                        rr0 = 1000
                        rm0 = 1000
                        tsr, tar, tnsr = env.fsr0
                        replay_buffer_r.append((tsr, tar, rr0, tnsr, done))
                    else :
                        rm0 = -1000
                        rr1 = 1000
                        rm1 = 1000
                        tsr, tar, tnsr = env.fsr1
                        replay_buffer_r.append((tsr, tar, rr1, tnsr, done))
                
                if sm0[2] != None:
                    replay_buffer_m.append((sm0, am0, rm0, nsm0, done))
                if sm1[2] != None:
                    replay_buffer_m.append((sm1, am1, rm1, nsm1, done))
                
                
                if sr0 != None and not done:
                    replay_buffer_r.append((sr0, ar0, rr0, nsr0, done))
                if sr1 != None and not done:
                    replay_buffer_r.append((sr1, ar1, rr1, nsr1, done))


                if len(replay_buffer_m) > REPLAY_MEMORY:
                    replay_buffer_m.popleft()
                
                if len(replay_buffer_r) > REPLAY_MEMORY:
                    replay_buffer_r.popleft()

                sm0, sm1, sr0, sr1 = nsm0, nsm1, nsr0, nsr1
                step_count +=1

                if step_count > 1000:
                    win = 2
                    break
                    
            print("episode: {}, winner: {}, {}steps.".format(episode,winner[win],step_count))
            
            if episode % 10 == 4:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer_m, 10)
                    loss, _ = simple_replay_trian(move_mainDQN0, move_targetDQN0, minibatch)
                    print("DQN_m0 : {}".format(loss))


            if episode % 10 == 9:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer_m, 10)
                    loss, _ = simple_replay_trian(move_mainDQN1, move_targetDQN1, minibatch)
                    print("DQN_m1 : {}".format(loss))
            
            # if episode % 10 == 4:
            #     for _ in range(50):
            #         minibatch = random.sample(replay_buffer_r, 10)
            #         loss, _ = simple_replay_trian(rotate_mainDQN0, rotate_targetDQN0, minibatch)
            #         print("DQN_r0 : {}".format(loss))

            # if episode % 10 == 8:
            #     for _ in range(50):
            #         minibatch = random.sample(replay_buffer_r, 10)
            #         loss, _ = simple_replay_trian(rotate_mainDQN1, rotate_targetDQN1, minibatch)
            #         print("DQN_r1 : {}".format(loss))
            
            if episode % 100 == 99:
                #play_bot(move_mainDQN0,move_mainDQN1,rotate_mainDQN0,rotate_mainDQN1)
                move_mainDQN0.save('./CheckPoints/DQN_m0',episode)
                move_mainDQN1.save('./CheckPoints/DQN_m1',episode)
                # rotate_mainDQN0.save('./CheckPoints/DQN_r0',episode)
                # rotate_mainDQN1.save('./CheckPoints/DQN_r1',episode)


if __name__ == '__main__':
    training(False, 0)
    #LoadAndPlay(999)