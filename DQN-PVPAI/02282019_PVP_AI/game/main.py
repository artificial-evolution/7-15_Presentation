import tensorflow as tf
import numpy as np
import PVP
import DQN

env = PVP.game()

winner = ['red','blue']

def ccw(x1, y1, x2, y2, x3, y3):
    temp = x1*y2+x2*y3+x3*y1 - y1*x2-y2*x3-y3*x1
    if temp > 0 :
        return 1
    elif temp < 0:
        return 0
    else:
        return 2

if __name__ == '__main__':
    max_episodes = 5000
    
    with tf.Session() as sess:
        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()
            
            while not done:
                ar0 = ccw(env.PLAYER0[1], env.PLAYER0[0], env.GUN0[1], env.GUN0[0], env.PLAYER1[1], env.PLAYER1[0])
                ar1 = ccw(env.PLAYER1[1], env.PLAYER1[0], env.GUN1[1], env.GUN1[0], env.PLAYER0[1], env.PLAYER0[0])

                am0 = int(np.random.rand(1) * 9)
                am1 = int(np.random.rand(1) * 9)

                # print(ar0, am0, ar1, am1)
                done, win, sm0, sm1, sr0, sr1 = env.step(ar0, am0, ar1, am1)
                
            print("episode: {}, winner: {}".format(episode,winner[win]))
            
