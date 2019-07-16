import tensorflow as tf
import numpy as np
import random
from collections import deque

dis = 0.9
REPLAY_MEMORY = 50000

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=15, l_rate=1e-1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32,  [None, self.input_size], name="input_x")
            

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            Z1 = tf.nn.relu(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            Z2 = tf.nn.relu(tf.matmul(Z1, W2))

            W3 = tf.get_variable("W3", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.matmul(Z2, W3)

            self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
            self._loss = tf.reduce_sum(tf.square(self._Y- self._Qpred))
            self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
    
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

    def save(self, name, step):
        saver = tf.train.Saver()
        saver.save(self.session, name, global_step=step, write_meta_graph=False)

    def load(self, name):  
        saver = tf.train.Saver()
        saver.restore(self.session, name)


def simple_replay_trian(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack,y_stack)

def get_copy_var_ops(*, dest_scope_name="target",src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder
