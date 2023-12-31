import numpy as np
from numpy import pi, sin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, LeakyReLU
import matplotlib.pyplot as plt

N = 5000  
epochs = 30000  

orthogonality_penalty_weight = 2e-8  
edge_penalty_weight = 1e5  
loss_adjustment_weight = 1e-2  


x = np.linspace(0, 1, N)
ground_state = np.sqrt(2) * sin(pi * x)
first_excited = np.sqrt(2) * sin(pi * 2 * x)
first_excited_minus = -first_excited


def psi(y):
    return y


def dpsi(y):
    y_shifted_f = tf.roll(y, shift=-1, axis=0)
    y_shifted_b = tf.roll(y, shift=1, axis=0)
    dy = (y_shifted_f - y_shifted_b) / 2
    return dy

def variationalE(y_true, y_pred):
    wave = psi(y_pred)
    wave_nom = K.l2_normalize(wave, axis=0)
    dwave = dpsi(wave_nom)
    kinetic_energy = N ** 2 * K.sum(K.square(dwave)) / pi ** 2

    orthogonality = K.sum(ground_state * wave_nom)
    orthogonality_penalty = orthogonality ** 2 * orthogonality_penalty_weight

    edge_penalty = (K.square(y_pred[0]) + K.square(y_pred[-1])) * edge_penalty_weight

    return kinetic_energy + orthogonality_penalty + edge_penalty

model = Sequential([
    Dense(256, input_dim=1, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(1, activation="linear")
])

def adjusted_variationalE(y_true, y_pred):
    original_loss = variationalE(y_true, y_pred)
    loss_adjustment = loss_adjustment_weight * (original_loss - 4) ** 2
    return original_loss + loss_adjustment - 4

model.compile(loss=adjusted_variationalE, optimizer="Adam")
model.summary()

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=1e-5, verbose=1)

results = model.fit(x, first_excited, epochs=epochs, steps_per_epoch=1, verbose=1, shuffle=False, callbacks=[reduce_lr])

pred = model.predict(x)
func = psi(pred)
func = func / np.sqrt(np.sum(func ** 2) / N)


plt.figure(figsize=(10, 5))  
plt.subplot(1, 2, 1)
plt.xlim(0, 1)
plt.plot(x, func, label="Fitted")
plt.plot(x, first_excited, "--", label="Answer")
plt.plot(x, first_excited_minus, "--", label="Answer Minus")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")

plt.subplot(1,2,2)
plt.ylim(0, 10)
plt.plot(results.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss (Log Scale)')

last_100_epochs_loss = results.history['loss'][-100:]

average_loss_last_100 = np.mean(last_100_epochs_loss)

plt.text(0.5, 0.95, f'Avg. Loss (Last 100 Epochs): {average_loss_last_100:.4e}', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)

info_text = (f'N: {N}\n'
             f'Epochs: {epochs}\n'
             f'Orthogonality Penalty Weight: {orthogonality_penalty_weight:e}\n'
             f'Edge Penalty Weight: {edge_penalty_weight:e}\n'
             f'Avg. Loss (Last 100 Epochs): {average_loss_last_100:.4e}')
plt.figtext(0.5, 0.05, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})


plt.tight_layout(rect=[0, 0.03, 1, 0.95])  

plt.show()

