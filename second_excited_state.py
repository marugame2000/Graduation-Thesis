import numpy as np
from numpy import pi, sin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, LeakyReLU
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

N = 5000 
epochs = 100000  

orthogonality_penalty_weight = 5e-3
edge_penalty_weight = 1e9 
normalization_penalty_weight=0

ground_state_weight = 1
first_state_weight = 1

x = np.linspace(0, 1, N)

ground_state = np.sqrt(2) * sin(pi * x)
first_excited =  (-1)*np.sqrt(2) * sin(2 * pi * x)
first_excited_minus = -first_excited
second_excited_answer = np.sqrt(2) * sin(3 * pi * x)
second_excited_answer_minus = -second_excited_answer
third_excited_answer = np.sqrt(2) * sin(4 * pi * x)
fourth_excited_answer = np.sqrt(2) * sin(5 * pi * x)


def psi(y):
    y_rev = K.reverse(y, axes=0)
    y_symmetrized = 0.5 * (y + y_rev)
    return y_symmetrized

def dpsi(y):
    y_shifted_f = tf.roll(y, shift=-1, axis=0)
    y_shifted_b = tf.roll(y, shift=+1, axis=0)
    dy = (y_shifted_f - y_shifted_b) / 2
    return dy

def variationalE(y_true, y_pred):
    wave = psi(y_pred)
    wave_nom = K.l2_normalize(wave, axis=0)
    dwave = dpsi(wave_nom)
    kinetic_energy = N ** 2 * K.sum(K.square(dwave)) / pi ** 2

    orthogonality_ground = K.sum(ground_state * wave_nom) * ground_state_weight
    orthogonality_first = K.sum(first_excited * wave_nom) * first_state_weight
    orthogonality_penalty = (orthogonality_ground ** 2 + orthogonality_first ** 2) * orthogonality_penalty_weight

    edge_penalty = (K.square(y_pred[0]) + K.square(y_pred[-1])) * edge_penalty_weight

    node1 = N // 3
    node2 = 2 * N // 3
    node_penalty = (K.square(wave_nom[node1]) + K.square(wave_nom[node2])) * 0

    normalization_penalty = K.square(K.sum(K.square(wave_nom)) - 1) * normalization_penalty_weight

    return (kinetic_energy-9)**2 + orthogonality_penalty + edge_penalty + node_penalty + normalization_penalty

model = Sequential([
    Dense(256, input_dim=1, activation=LeakyReLU(alpha=0.3)),
    Dense(256, activation=LeakyReLU(alpha=0.3)),
    Dense(256, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(1)
])

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate, epsilon=1e-9)
model.compile(loss=variationalE, optimizer=optimizer)

model.summary()


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=5e-6, verbose=1)

results = model.fit(
    x, 
    second_excited_answer, 
    epochs=epochs, 
    steps_per_epoch=1, 
    verbose=1, 
    shuffle=False, 
    callbacks=[reduce_lr]
)

pred = model.predict(x)
func = psi(pred)
#func = func*1800
func = func / np.sqrt(np.sum(func ** 2) / N) 

plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1)  
plt.xlim(0, 1)
plt.plot(x, func, label="Fitted")
plt.plot(x, second_excited_answer, "--", label="Answer")
plt.plot(x, second_excited_answer_minus, "--", label="Answer Minus")
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
            f'ground_state_weight: {ground_state_weight:e}\n'
            f'first_state_weight: {first_state_weight:e}\n'
            f'Avg. Loss (Last 100 Epochs): {average_loss_last_100:.4e}')
plt.figtext(0.5, 0.05, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rectパラメータで図の余白を調整

plt.show()

