from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)
 
df = pd.read_csv('dino.csv', header=None)
 
dataset = df.values
X = dataset[:,0:5]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
Y = dataset[:,5]
Y = to_categorical(Y, num_classes=None)

model = Sequential()
#model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(LSTM(64, input_shape=(1,5),activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(2, activation='sigmoid'))

#loss='categorical_crossentropy'
model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])
 
model.fit(X, Y, epochs=100, batch_size=50)
 
model.save('dino.h5')
 
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))