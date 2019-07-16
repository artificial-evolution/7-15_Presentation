import tensorflow as tf
import numpy as np
import PVP
from collections import deque
from dqn import *

env = PVP.game()
winner = ['red win','blue win','draw']

move_input_size = 5
move_output_size = 9

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

def play_bot(move_mainDQN0,move_mainDQN1):
    env.pygame_init()
    _, _, sm0, sm1, _, _ = env.reset()
    done = False

    while not done:
        ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])
        ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])
        am0 = np.argmax(move_mainDQN0.predict(sm0))
        am1 = np.argmax(move_mainDQN1.predict(sm1))

        done, win, sm0, sm1, _, _ = env.step(ar0, am0, ar1, am1)
        env.render()
        if done:
            print(winner[win])

def play_game(prefix):
    
    with tf.Session() as sess:
        DQN_m = DQN(sess, move_input_size, move_output_size, name="movemain1")
        loadDQN(DQN_m,'./CheckPoints/DQN_m1-',prefix)

        env.pygame_init()
        _, _, _, sm1, _, _ = env.reset()
        done = False

        while not done:
            ud, lr, x, y = env.player_action()
            ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], x, y)
            ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])
            am0 = ud*3 + lr
            am1 = np.argmax(DQN_m.predict(sm1))
            print(am0)
            done, win, _, sm1, _, _ = env.step(ar0, am0, ar1, am1)
            env.render()
            if done:
                print(winner[win])

def training(check_epoisode):
    max_episodes = 5000
    
    replay_buffer_m = deque()

    with tf.Session() as sess:
        move_mainDQN0 = DQN(sess, move_input_size, move_output_size, name="movemain0" )
        move_targetDQN0 = DQN(sess, move_input_size, move_output_size, name="movetarget0" )

        move_mainDQN1 = DQN(sess, move_input_size, move_output_size, name="movemain1")
        move_targetDQN1 = DQN(sess, move_input_size, move_output_size, name="movetarget1")
        
        check_epoisode = 0
        copy_ops0 = get_copy_var_ops(dest_scope_name="movetarget0",src_scope_name="movemain0")
        copy_ops1 = get_copy_var_ops(dest_scope_name="movetarget1",src_scope_name="movemain1")

        tf.global_variables_initializer().run()

        for episode in range(check_epoisode, check_epoisode+max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            _, _, sm0, sm1, _, _ = env.reset()
            
            while not done:
                
                if np.random.rand(1) < 0.33:
                    am0 = np.random.randint(3, size=1)
                else:
                    ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])

                if np.random.rand(1) < 0.33:
                    ar1 = np.random.randint(3, size=1)
                else:
                    ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])

                # move_red DQN e-greedy
                if np.random.rand(1) < e:
                    am0 = np.random.randint(9, size=1)
                else:
                    am0 = np.argmax(move_mainDQN0.predict(sm0))

                # move_blue DQN e-greedy
                if np.random.rand(1) < e:
                    am1 = np.random.randint(9, size=1)
                else:
                    am1 = np.argmax(move_mainDQN1.predict(sm1))

                done, win, nsm0, nsm1, _, _ = env.step(ar0, am0, ar1, am1)

                r0, r1 = -1, -1

                if done:
                    if win == 0:
                        r1 = -1000
                    else :
                        r0 = -1000
                
                if sm0[2] != None:
                    replay_buffer_m.append((sm0, am0, r0, nsm0, done))
                if sm1[2] != None:
                    replay_buffer_m.append((sm1, am1, r1, nsm1, done))

                if len(replay_buffer_m) > REPLAY_MEMORY:
                    replay_buffer_m.popleft()

                sm0 = nsm0
                sm1 = nsm1
                step_count +=1

                if step_count > 1000:
                    win = 2
                    break
                    
            print("episode: {}, winner: {}, {}steps.".format(episode,winner[win],step_count))
            
            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer_m, 10)
                    loss, _ = simple_replay_trian(move_mainDQN0, move_targetDQN0, minibatch)
                print("DQN_m0 : {}".format(loss))

            if episode % 10 == 6:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer_m, 10)
                    loss, _ = simple_replay_trian(move_mainDQN1, move_targetDQN1, minibatch)
                print("DQN_m1 : {}".format(loss))
            
            if episode % 1000 == 999:
                play_bot(move_mainDQN0,move_mainDQN1)
                move_mainDQN0.save('./CheckPoints/DQN_m0',episode)
                move_mainDQN0.save('./CheckPoints/DQN_m1',episode)


if __name__ == '__main__':
    training(0)
    #LoadAndPlay(999)
    #play_game(999)