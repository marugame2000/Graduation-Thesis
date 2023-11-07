import numpy as np
from numpy import pi, sin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras import backend as K
from keras import layers, optimizers
from keras.layers import Dense, Dropout, Activation, LeakyReLU
import matplotlib.pyplot as plt
from keras.optimizers import Adam

N = 3000
x = np.linspace(0, 1, N)
ground_state = np.sqrt(2) * np.sin(pi * x)
first_excited = (1)*np.sqrt(2) * np.sin(2 * pi * x)
first_excited_minus = (-1)*np.sqrt(2) * np.sin(2 * pi * x)
second_excited_answer = np.sqrt(2) * np.sin(3 * pi * x)
second_excited_answer_minus = (-1)*np.sqrt(2) * np.sin(3 * pi * x)
third_excited_answer = np.sqrt(2) * np.sin(4 * pi * x)
fourth_excited_answer = np.sqrt(2) * np.sin(5 * pi * x)

#中間層を減らしてニューロンの数を増やす
#ソースコード、バージョンの管理
#値の設定とグラフの対応の保存
#値ごとの精度をパワポにまとめる

def psi(y):
    y_rev = K.reverse(y, axes=0) 
    y_symmetrized = 0.5 * (y + y_rev)
    #y_symmetrized=y
    return y_symmetrized

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
    
    orthogonality_ground = (K.sum(ground_state * wave_nom))**2
    orthogonality_first_excited = (K.sum(first_excited * wave_nom))**2
    orthogonality_first_minus_excited = (K.sum(first_excited_minus * wave_nom))**2
    #orthogonality_third_excited = (K.sum(third_excited_answer * wave_nom))**2
    #orthogonality_fourth_excited = (K.sum(fourth_excited_answer * wave_nom))**2
    orthogonality_penalty = orthogonality_ground + orthogonality_first_excited + orthogonality_first_minus_excited

    edge_penalty = K.square(y_pred[0]) + K.square(y_pred[-1])

    #normalization_penalty = K.square(1.0 - K.sum(wave_nom**2)/N)
    
    return kinetic_energy + orthogonality_penalty * 5e-8 + edge_penalty * 1e9


def compute_kinetic_energy_and_orthogonality(y_pred):
    wave = psi(y_pred)
    wave_nom = wave / K.sqrt(K.sum(K.square(wave)) / N)
    dwave = dpsi(wave_nom)
    kinetic_energy = N**2 * K.sum(K.square(dwave)) / pi**2

    orthogonality_ground = (K.sum(ground_state * wave_nom))**2
    orthogonality_first_excited = (K.sum(first_excited * wave_nom))**2
    orthogonality_first_excited_minus = (K.sum(first_excited_minus * wave_nom))**2
    orthogonality_penalty = orthogonality_ground + orthogonality_first_excited+orthogonality_first_excited_minus

    return kinetic_energy.numpy(), orthogonality_penalty.numpy()

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate,epsilon=1e-9)

model = Sequential([
model = Sequential([
    Dense(256, input_dim=1),
    LeakyReLU(alpha=0.3),

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
    Dense(1, activation="linear")
])

model.compile(loss=variationalE, optimizer=optimizer)
model.summary()

from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=1e-5, verbose=1)

results = model.fit(x, second_excited_answer, epochs=30000, steps_per_epoch=1, verbose=1, shuffle=False, callbacks=[reduce_lr])

pred = model.predict(x)
func = psi(pred)
func = func / np.sqrt(np.sum(func**2) / N)
#kinetic_energy_val, orthogonality_penalty_val = compute_kinetic_energy_and_orthogonality(pred)
#print(f"Kinetic Energy: {kinetic_energy_val}")
#print(f"Orthogonality Penalty: {orthogonality_penalty_val}")
plt.xlim(0, 1)
plt.plot(x, func, label="fitted")
plt.plot(x, second_excited_answer, label="answer", linestyle='dashed')
plt.plot(x, second_excited_answer_minus, label="answer", linestyle='dashed')
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")
plt.show()