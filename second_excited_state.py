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
first_excited = np.sqrt(2) * np.sin(2 * pi * x)
second_excited_answer = np.sqrt(2) * np.sin(3 * pi * x)
second_excited_answer_minus = (-1)*np.sqrt(2) * np.sin(3 * pi * x)
third_excited_answer = np.sqrt(2) * np.sin(4 * pi * x)

def psi(y):
    y_rev = K.reverse(y,0)
    y_symmetrized = y + y_rev -y[0] -y[-1]
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
    
    orthogonality_ground = 1* (K.sum(ground_state * wave_nom))**2
    orthogonality_first_excited = 1*(K.sum(first_excited * wave_nom))**2
    #orthogonality_third_excited = 1*(K.sum(third_excited_answer * wave_nom))**2
    
    orthogonality_penalty = (orthogonality_ground + orthogonality_first_excited)

    index_at_0_1 = int(0.1 * N)
    positive_penalty = -1e5 * K.minimum(y_pred[index_at_0_1], 0)

    #positive_penalty = -1e12 * K.minimum(y_pred[500], 0)
    
    return kinetic_energy * 1 + orthogonality_penalty * 5e-9

def compute_kinetic_energy_and_orthogonality(y_pred):
    wave = psi(y_pred)
    wave_nom = wave / K.sqrt(K.sum(K.square(wave)) / N)
    dwave = dpsi(wave_nom)
    kinetic_energy = N**2 * K.sum(K.square(dwave)) / pi**2
    
    orthogonality_ground = (K.sum(ground_state * wave_nom))**2
    orthogonality_first_excited = (K.sum(first_excited * wave_nom))**2
    orthogonality_penalty = orthogonality_ground + orthogonality_first_excited

    return kinetic_energy.numpy(), orthogonality_penalty.numpy()


learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate,epsilon=1e-9)

model = Sequential()

model.add(Dense(2, input_dim=1, activation="relu"))
model.add(Dense(1, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(loss=variationalE, optimizer=optimizer)
model.summary()

results = model.fit(x, second_excited_answer, epochs=2000, steps_per_epoch=1, verbose=1, shuffle=False)


pred = model.predict(x)
func = psi(pred)
func = func / np.sqrt(np.sum(func**2) / N)
#kinetic_energy_val, orthogonality_penalty_val = compute_kinetic_energy_and_orthogonality(pred)
#print(f"Kinetic Energy: {kinetic_energy_val}")
#print(f"Orthogonality Penalty: {orthogonality_penalty_val}")
plt.xlim(0, 1)
plt.plot(x, func, label="fitted")
plt.plot(x, second_excited_answer, label="answer")
plt.plot(x, second_excited_answer_minus, label="answer")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")
plt.show()

