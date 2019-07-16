import pvpgame
import contextlib
import tensorflow as tf
import numpy as np
with contextlib.redirect_stdout(None):
    import pygame

learning_rate = 1e-1
input_size_m  = 3
output_size_m = 4

env = pvpgame.game()
episodes = 10
dis = 0.9
rList = []

X_m     = tf.placeholder(tf.float32, [None, input_size_m], name="input_x")
W1_m    = tf.get_variable("W1_m", shape=[input_size_m, output_size_m], initializer=tf.contrib.layers.xavier_initializer())
Qpred_m = tf.matmul(X_m,W1_m)
Y_m     = tf.placeholder(shape=[None, output_size_m], dtype=tf.float32)
loss    = tf.reduce_sum(tf.square(Y_m-Qpred_m))
train   = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

win = ['Red','Blue']
for i in range(episodes):
    e = 1. /((i / 10) + 1)
    rAll = 0
    step_count = 0
    _, _, _, _, s_m = env.init()
    done = False
    sess = tf.Session()

    while not done:
        step_count += 1
        x_m = np.reshape(s_m, [1, input_size_m])
        Qs_m = sess.run(Qpred_m, feed_dict={X_m: x_m})
        
        poab = False
        
        while not poab:
            a_r = np.random.rand(2)
            if np.random.rand(1) < e:
                a_m = np.random.rand(4)
            else:
                a_m = np.argmax(Qs_m)
                
            done, win, s1_r, poab, s1_m  = env.step(a_r, a_m)
            print(done)
        if done:
            if win == 0:
                Qs_m[0, a_m] = -100
        else:
            x1_m          = np.reshape(s_m, [1, input_size_m])
            Qs1_m         = sess.run(Qpred_m, feed_dict={X_m: x1_m})
            Qs1_m[0, a_m] = 1 + dis * np.max(Qs1_m)

        sess.run(train, feed_dict={X_m: x_m, Y_m:Qs1_m})
        s_m = s1_m
    print(win[win],'win')