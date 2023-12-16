import numpy as np
from numpy import pi, sin, sqrt,cos
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, LeakyReLU
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import sys
from scipy.integrate import quad

#基底状態の波動関数と直交させる

N = 3000 
epochs = 30000

orthogonality_penalty_weight = 5e-5
edge_penalty_weight = 1e9 
normalization_penalty_weight=0

ground_state_weight = 1
first_state_weight = 1

x = np.linspace(0, 1, N)

ground_state =lambda x:  np.sqrt(2) * sin(pi * x)
first_excited = lambda x:  (-1)*np.sqrt(2) * sin(2 * pi * x)
first_excited_minus = lambda x: -first_excited
second_excited_answer = lambda x: np.sqrt(2) * sin(3 * pi * x)
second_excited_answer_minus =lambda x:  -second_excited_answer
third_excited_answer = lambda x:(-1) * np.sqrt(2) * sin(4 * pi * x)
fourth_excited_answer = np.sqrt(2) * sin(5 * pi * x)

tf.random.set_seed(1)

#tf.random_seed(1)
#psi関数が機能していない可能性

def inner_product_tf(f, g, a, b, num_points=N):
    x_values = tf.linspace(a, b, num_points)
    f_values = f(x_values)
    g_values = g(x_values)

    f_values = tf.cast(f_values, tf.float32)
    g_values = tf.cast(g_values, tf.float32)

    integrand_values = f_values * g_values
    inner_product = tf.reduce_sum(integrand_values) * (b - a) / num_points
    return inner_product

def psi1(y_pred):
    a = 0.0
    b = 1.0
    pi_tf = tf.constant(np.pi, dtype=tf.float32)

    def ground_state_tf(x):
        return tf.sqrt(2.0) * tf.sin(pi_tf * x)

    inner_fg = inner_product_tf(ground_state_tf, lambda x: y_pred, a, b)
    norm_f_squared = inner_product_tf(ground_state_tf, ground_state_tf, a, b)
    y_orthogonalized = y_pred - (inner_fg / norm_f_squared) * ground_state_tf(tf.linspace(a, b, len(y_pred)))
    
    return y_orthogonalized

def psi2(y_pred):
    a = 0.0
    b = 1.0
    pi_tf = tf.constant(np.pi, dtype=tf.float32)

    def ground_state_tf(x):
        return tf.sqrt(2.0) * tf.sin(2 *pi_tf * x)

    inner_fg = inner_product_tf(ground_state_tf, lambda x: y_pred, a, b)
    norm_f_squared = inner_product_tf(ground_state_tf, ground_state_tf, a, b)
    y_orthogonalized = y_pred - (inner_fg / norm_f_squared) * ground_state_tf(tf.linspace(a, b, len(y_pred)))
    
    return y_orthogonalized

def psi3(y_pred):
    a = 0.0
    b = 1.0
    pi_tf = tf.constant(np.pi, dtype=tf.float32)

    def ground_state_tf(x):
        return tf.sqrt(2.0) * tf.sin(3 *pi_tf * x)

    inner_fg = inner_product_tf(ground_state_tf, lambda x: y_pred, a, b)
    norm_f_squared = inner_product_tf(ground_state_tf, ground_state_tf, a, b)
    y_orthogonalized = y_pred - (inner_fg / norm_f_squared) * ground_state_tf(tf.linspace(a, b, len(y_pred)))
    
    return y_orthogonalized


def dpsi(y):
    y_shifted_f1 = tf.roll(y, shift=-1, axis=0)
    y_shifted_f2 = tf.roll(y, shift=-2, axis=0)
    y_shifted_f3 = tf.roll(y, shift=-3, axis=0)
    y_shifted_f4 = tf.roll(y, shift=-4, axis=0)

    y_shifted_b1 = tf.roll(y, shift=+1, axis=0)
    y_shifted_b2 = tf.roll(y, shift=+2, axis=0)
    y_shifted_b3 = tf.roll(y, shift=+3, axis=0)
    y_shifted_b4 = tf.roll(y, shift=+4, axis=0)
    
    dy = (y_shifted_f1 - y_shifted_b1) * 4/5 + (y_shifted_f2 - y_shifted_b2) * (-1/5) + (y_shifted_f3 - y_shifted_b3) * 4/105 + (y_shifted_f4 - y_shifted_b4) * (-1/280)
    return dy

def variationalE(y_true, y_pred):

    # y_pred = tf.reshape(y_pred,[-1]
    #wave = K.l2_normalize(y_pred, axis=0)
    #wave = y_pred / K.sqrt(K.sum(K.square(y_pred)))
    wave = K.l2_normalize(y_pred, axis=0)
    wave = tf.squeeze(wave)
    print(wave.shape)
    wave = psi1(wave)
    wave = psi2(wave)
    wave = psi3(wave)
    print(wave.shape)
    
    #wave = wave / K.sqrt(K.sum(K.square(wave)) / N)
    wave = tf.cast(wave, tf.float32)
    #wave_nom=wave
    wave_nom = K.l2_normalize(wave, axis=0)
    
    dwave = dpsi(wave_nom)
    kinetic_energy = N ** 2 * K.sum(K.square(dwave)) / pi ** 2 

    edge_penalty = (K.square(wave_nom[0]) + K.square(wave_nom[-1])) * edge_penalty_weight
    
    return kinetic_energy + edge_penalty*0.5


model = Sequential([
    Dense(512, input_dim=1, activation=LeakyReLU(alpha=0.3)),
    Dense(256, activation=LeakyReLU(alpha=0.3)),
    Dense(256, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(1, activation="linear")
])

def adjusted_variationalE(y_true, y_pred):
    original_loss = variationalE(y_true, y_pred)
    loss_adjustment = 1e-2 * (original_loss - 4) ** 2
    return original_loss + loss_adjustment - 4

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss=variationalE, optimizer=optimizer)

model.summary()

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=3e-5, verbose=1)
y_target = second_excited_answer(x)
results = model.fit(
    x, 
    y_target, 
    epochs=epochs, 
    steps_per_epoch=1, 
    verbose=1, 
    shuffle=False, 
    callbacks=[reduce_lr, ]
)

pred = model.predict(x)
print(pred.shape)
pred = tf.squeeze(pred)
pred = psi1(pred)
func = psi2(pred)
func = psi3(pred)
print(func.shape)
func = func / np.sqrt(np.sum(func ** 2)/N) 

inner_product = np.trapz(ground_state(x) * func, x)
print(inner_product)
inner_product = np.trapz(first_excited(x) * func, x)
print(inner_product)

plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1)  
plt.xlim(0, 1)
plt.plot(x, func, label="Fitted")
plt.plot(x, third_excited_answer(x), "--", label="Answer")
#plt.plot(x, second_excited_answer_minus, "--", label="Answer Minus")
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

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  

plt.show()