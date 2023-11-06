import numpy as np
from numpy import pi,sin,sqrt
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras import backend as K
from keras import layers, models
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
import matplotlib.pyplot as plt

N=5000
x=np.linspace(0, 1, N)
ground_state=np.sqrt(2)*np.sin(np.pi*x)
first_excited=(1)*np.sqrt(2)*np.sin(np.pi*2*x)
first_excited_minus=(-1)*np.sqrt(2)*np.sin(np.pi*2*x)



#境界で0出ない場合ペナルてxい跳ね上げる
#調和振動子
#Adam変えてみる


def psi(y):
    #indices = tf.constant([[0], [-1]]) 
    #updates = tf.constant([[0.0], [0.0]], dtype=tf.float32) 
    #y_symmetrized = tf.tensor_scatter_nd_update(y, indices, updates)
    y_symmetrized=y
    return y_symmetrized



#def psi(y):
    y_rev = K.reverse(y,0)
    y_symmetrized = y + y_rev -y[0] -y[-1]
    return y_symmetrized

#奇関数かつ？橋が0になる条件作る
#値を直接定義するのもあり

#def psi(y):
    first_half = y[:y.shape[0]//2]
    second_half = -K.reverse(first_half, axes=0)
    return K.concatenate([second_half, first_half], axis=0)

def dpsi(y):
    y_shifted_f = tf.roll(y, shift=-1, axis=0)
    y_shifted_b = tf.roll(y, shift=+1, axis=0)
    dy = (y_shifted_f - y_shifted_b) / 2
    return dy

def variationalE(y_true, y_pred):
    wave = psi(y_pred)
    wave_nom = K.l2_normalize(wave, 0)
    dwave = dpsi(wave_nom)
    kinetic_energy = N**2 * K.sum(K.square(dwave)) / pi**2
    
    orthogonality = K.sum(ground_state * wave_nom)
    orthogonality_penalty =  orthogonality**2

    edge_penalty = K.square(y_pred[0]) + K.square(y_pred[-1])

    #positive_penalty = -1e10 * K.minimum(y_pred[1000], 0)
    
    return kinetic_energy + orthogonality_penalty * 2e-8 + edge_penalty * 1e5

model = Sequential([
    Dense(256, input_dim=1),
    LeakyReLU(alpha=0.3),
    #BatchNormalization(),

    Dense(128),
    LeakyReLU(alpha=0.3),
    #BatchNormalization(),

    Dense(128),
    LeakyReLU(alpha=0.3),
    #BatchNormalization(),

    Dense(128),
    LeakyReLU(alpha=0.3),
    #BatchNormalization(),

    Dense(64),
    LeakyReLU(alpha=0.3),
    #BatchNormalization(),

    Dense(64),
    LeakyReLU(alpha=0.3),
    #BatchNormalization(),

    Dense(1, activation="linear")
])

def adjusted_variationalE(y_true, y_pred):
    original_loss = variationalE(y_true, y_pred)
    loss_adjustment = 1e-2 * (original_loss - 4)**2
    return original_loss + loss_adjustment -4

model.compile(loss=adjusted_variationalE, optimizer="Adam")
model.summary()

from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=1e-5, verbose=1)

results = model.fit(x, first_excited, epochs=50000, steps_per_epoch=1, verbose=1, shuffle=False, callbacks=[reduce_lr])

pred = model.predict(x)
func = psi(pred)
func = func / np.sqrt(np.sum(func**2) / N)
plt.xlim(0, 1)
plt.plot(x, func, label="fitted")
plt.plot(x, first_excited, label="answer", linestyle='dashed')
plt.plot(x, first_excited_minus, label="answer", linestyle='dashed')
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")
plt.show()
