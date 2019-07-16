import tensorflow as tf
import numpy as np
import PVP
from collections import deque
from dqn import *

env = PVP.game()
winner = ['red win','blue win','draw']

move_input_size = 22
move_output_size = 9

rotate_input_size = 22
rotate_output_size = 1

dis = 0.9
REPLAY_MEMORY = 100

def ccw(x1, y1, x2, y2, x3, y3):
    temp = x1*y2+x2*y3+x3*y1 - y1*x2-y2*x3-y3*x1
    if temp > 0 :
        return 1
    elif temp < 0:
        return 0
    else:
        return 2

def play_bot(move_mainDQN0,move_mainDQN1):
    env.pygame_init()
    _, _, sm0, sm1, sr0, sr1 = env.reset()
    done = False

    while not done:
        if np.random.rand(1) < 0.33:
            ar0 = np.random.randint(3, size=1)
        else:
            ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])

        if np.random.rand(1) < 0.33:
            ar1 = np.random.randint(3, size=1)
        else:
            ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])

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

def training():
    max_episodes = 10000
    
    replay_buffer_m = deque()
    
    replay_buffer_r = deque()

    with tf.Session() as sess:
        move_mainDQN0 = DQN(sess, move_input_size, move_output_size, name="movemain0" )
        move_targetDQN0 = DQN(sess, move_input_size, move_output_size, name="movetarget0" )

        move_mainDQN1 = DQN(sess, move_input_size, move_output_size, name="movemain1")
        move_targetDQN1 = DQN(sess, move_input_size, move_output_size, name="movetarget1")
        
        rotate_mainDQN0 = DQN(sess, rotate_input_size, rotate_output_size, name="rotatemain0" )
        rotate_targetDQN0 = DQN(sess, rotate_input_size, rotate_output_size, name="rotatetarget0" )

        rotate_mainDQN1 = DQN(sess, rotate_input_size, rotate_output_size, name="rotatemain1")
        rotate_targetDQN1 = DQN(sess, rotate_input_size, rotate_output_size, name="rotetetarget1")
        

        check_epoisode = 0
        copy_ops0 = get_copy_var_ops(dest_scope_name="movetarget0",src_scope_name="movemain0")
        copy_ops1 = get_copy_var_ops(dest_scope_name="movetarget1",src_scope_name="movemain1")

        rcopy_ops0 = get_copy_var_ops(dest_scope_name="rotatetarget0",src_scope_name="rotatemain0")
        rcopy_ops1 = get_copy_var_ops(dest_scope_name="rotetetarget1",src_scope_name="rotatemain1")

        tf.global_variables_initializer().run()
        

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            final_step = 0
            player0_move = list([])
            player1_move = list([])
            player0_rotate = list([])
            player1_rotate = list([])

            _, _, sm0, sm1, sr0, sr1 = env.reset()
            
            while not done:
                
                ud, lr, diff = env.player_action()
                ar0 = diff
                am0 = ud*3 + lr
                # if np.random.rand(1) < 0.33:
                #     ar0 = np.random.randint(3, size=1)
                # else:
                #     ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])

                if np.random.rand(1) < 0.33:
                    ar1 = np.random.randint(3, size=1)
                else:
                    print(sr1)
                    ar1 = rotate_mainDQN1.predict(sr1)[0]

                # move_red DQN e-greedy
                # if np.random.rand(1) < e:
                #     am0 = np.random.randint(9, size=1)
                # else:
                #     am0 = np.argmax(move_mainDQN0.predict(sm0))

                # move_blue DQN e-greedy
                if np.random.rand(1) < e:
                    am1 = np.random.randint(9, size=1)
                else:
                    am1 = np.argmax(move_mainDQN1.predict(sm1))

                done, win, nsm0, nsm1, nsr0, nsr1 = env.step(ar0, am0, ar1, am1)
                if episode:
                    env.render()

                if done:
                    final_step = step_count

                player0_move.append([sm0, am0, nsm0, done])
                player1_move.append([sm1, am1, nsm1, done])
                

                player0_rotate.append([sm0, am0, nsm0, done])
                player1_rotate.append([sm1, am1, nsm1, done])

                sm0 = nsm0
                sm1 = nsm1
    
                sr0 = nsr0
                sr1 = nsr1

                step_count +=1

                if step_count > max_episodes:
                    win = 2
                    break
                    
            print("episode: {}, winner: {}, {}steps.".format(episode,winner[win],step_count))
    
            if win != 2:
                for i in range(len(player1_move)):
                    if win and i>len(player1_move)-41:
                        player1_move[i].append(-1)
                    else:
                        player1_move[i].append(pow(i/final_step,1/2))
                        
                    if not win and i>len(player1_move)-41:
                        player1_rotate[i].append(1)
                    else:
                        player1_rotate[i].append(pow(i/final_step,1/2))
                    
                for i in range(len(player0_move)):
                    if not win and i>len(player0_move)-41:
                        player0_move[i].append(-1)
                    else:
                        player0_move[i].append(pow(i/final_step,1/2))

                    if win and i>len(player0_move)-41:
                        player0_rotate[i].append(1)
                    else:
                        player0_rotate[i].append(pow(i/final_step,1/2))
            else:
                for i in range(len(player0_move)):
                    player0_move[i].append(1)
                    player1_move[i].append(1)
                    player0_rotate[i].append(1)
                    player1_rotate[i].append(1)

            replay_buffer_m.append(player0_move)
            replay_buffer_m.append(player1_move)
            replay_buffer_r.append(player0_move)
            replay_buffer_r.append(player1_move)

            if len(replay_buffer_m) > REPLAY_MEMORY:
                    replay_buffer_m.popleft()
            if len(replay_buffer_r) > REPLAY_MEMORY:
                    replay_buffer_r.popleft()

            # if episode % 40 == 19:
            #     for _ in range(20):
            #         batch = random.sample(replay_buffer_m, 1)[0]
            #         loss, _ = simple_replay_trian(move_mainDQN0, move_targetDQN0, batch)
            #     print("DQN_m0 : {}".format(loss))

            if episode % 40 == 39:
                for _ in range(20):
                    batch = random.sample(replay_buffer_m, 1 )[0]
                    loss, _ = simple_replay_trian(move_mainDQN1, move_targetDQN1, batch)
                print("DQN_m1 : {}".format(loss))

                for _ in range(20):
                    batch = random.sample(replay_buffer_r, 1 )[0]
                    loss, _ = simple_replay_trian(rotate_mainDQN1, rotate_targetDQN1, batch)
                print("DQN_r1 : {}".format(loss))
        

if __name__ == '__main__':
    training()
    #LoadAndPlay(999)
    #play_game(999)