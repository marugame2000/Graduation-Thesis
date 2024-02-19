import numpy as np
from numpy import pi, sin,cos , sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import Dense, LeakyReLU,BatchNormalization,Input
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sympy import symbols, pi, cos, integrate
from scipy.integrate import quad
import time
from libs.analytical_solution_dimensionless import solution
import json
from datetime import datetime  # for generating timestamp
import os
from tqdm import tqdm
import random

# 定数の設定
X_CENTER_MIN = -0.5
X_CENTER_MAX = 0.5
X_MIN = -5
X_MAX = 5
N_SAMPLES = 3000
epochs = 30000
N_BASIS = 11
x = np.linspace(X_MIN, X_MAX, N_SAMPLES)

def psi(n, l, x):
    l = l + 1
    mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
    l = tf.cast(l / 20, tf.float32)
    x = tf.cast(x, tf.float32) 
    coefficient = 1 / (l * tf.sqrt(2 * np.pi))
    exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
    return coefficient * tf.exp(exponent)


def deriv_phi(n, l, x):
    l = l + 1
    mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
    l = tf.cast(l / 20, tf.float32)
    x = tf.cast(x, tf.float32)  
    coefficient = 1 / (l * tf.sqrt(2 * np.pi))
    exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
    normal_dist = coefficient * tf.exp(exponent)
    derivative = -(x - mu_n) / (l ** 2)
    return derivative * normal_dist


def trapezoidal_rule(f, a, b, n=200):
    h = (b - a) / n
    x = tf.linspace(a, b, n)
    y = f(x)
    return h * (0.5 * (y[0] + y[-1]) + tf.reduce_sum(y[1:-1]))


cached_matrices = {}  

def calculate_matrices(h_tensor):
    h_value = h_tensor.numpy()[0]  
    if h_value in cached_matrices:
    
        return cached_matrices[h_value]

    def V(x,h):
        return np.where((x < X_CENTER_MAX) & (x > X_CENTER_MIN), 0, h)
    
    def calc_h(n, l, m, k):
        integrand = lambda x: (deriv_phi(n, l, x) * deriv_phi(m, k, x) / 2 + psi(n, l, x) * V(x,h_value) * psi(m, k, x))
        return trapezoidal_rule(integrand, X_MIN, X_MAX)

    def calc_s(n, l, m, k):
        integrand = lambda x: psi(n, l, x) * psi(m, k, x)
        return trapezoidal_rule(integrand, X_MIN, X_MAX)

    S = np.zeros((N_BASIS, N_BASIS, N_BASIS, N_BASIS), dtype=np.float32)
    H = np.zeros((N_BASIS, N_BASIS, N_BASIS, N_BASIS), dtype=np.float32)

    for i in tqdm(range(N_BASIS), desc="Progress for i"):
        for j in tqdm(range(N_BASIS), desc="Progress for j", leave=False):
            for k in tqdm(range(N_BASIS), desc="Progress for k", leave=False):
                for l in tqdm(range(N_BASIS), desc="Progress for l", leave=False):
                    S[i, j, k, l] = calc_s(i, j, k, l)
                    H[i, j, k, l] = calc_h(i, j, k, l)

    cached_matrices[h_value] = (S, H)
    return S, H

solution_cache = {}

def get_cached_solution(h, index):
    key = (h, index)
    if key not in solution_cache:
        solution_cache[key] = solution(h, index)
        print(solution_cache[key])
    return solution_cache[key]

def integrate_psi_squared(psi, x):
    return np.trapz(np.abs(psi) ** 2, x)


def model_loss(c, h,epoch):
    S, H = calculate_matrices(h)
    c = tf.reshape(c, [N_BASIS, N_BASIS])
    h = h.numpy()[0] 

    def calc_energy(c):
        
        Hc = tf.tensordot(H, c, axes=([2, 3], [0, 1]))
        Sc = tf.tensordot(S, c, axes=([2, 3], [0, 1]))
        cHc = tf.tensordot(c, Hc, axes=([0, 1], [0, 1]))
        cSc = tf.tensordot(c, Sc, axes=([0, 1], [0, 1]))
        return cHc / cSc
    
    if epoch%50==0:
        predicted_psi = tf.zeros_like(x, dtype=tf.float32) 
        for i in range(N_BASIS):
            for j in range(N_BASIS):
                predicted_psi += c[i, j] * psi(i, j, x)
        predicted_psi = K.l2_normalize(predicted_psi, axis=0)
        norm_factor = np.sqrt(integrate_psi_squared(predicted_psi, x))
        predicted_psi /= norm_factor
        


        psi_solution = get_cached_solution(h, 1)
        psi_solution_minus = -psi_solution
        #predicted_psi_ground = K.l2_normalize(predicted_psi_ground, axis=0)
        predicted_psi = tf.cast(predicted_psi, tf.float32)  # tf.float32にキャスト


        
        print(f"h {h}")
        #print(c)
        #print(Lagrange_multiplier)
        plt.plot(x, predicted_psi, label="Predicted")
        #plt.plot(x, predicted_psi_ground ,"--", label="Answer")
        plt.plot(x, psi_solution ,"--", label="Answer")
        plt.plot(x, psi_solution_minus ,"--", label="Answer")
        plt.xlim(-2, 2)
        plt.xlabel("Coordinate $x$ [Bohr]")
        plt.ylabel("Wave amplitude")
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)  # 10秒表示
        plt.close()  # 図を閉じる

    
    energy_loss = calc_energy(c)
    return energy_loss

from keras import regularizers

class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=LeakyReLU(alpha=0.3))
        #self.dropout1 = tf.keras.layers.Dropout(0.01)
        self.dense2 = tf.keras.layers.Dense(256, activation=LeakyReLU(alpha=0.3))
        #self.dropout2 = tf.keras.layers.Dropout(0.01)
        # 以下、同様に他の層にも正則化とドロップアウトを追加
        #self.dense3 = tf.keras.layers.Dense(20, activation='tanh')
        #self.dropout3 = tf.keras.layers.Dropout(0.01)
        #self.dense4 = tf.keras.layers.Dense(20, activation='tanh')
        #self.dropout4 = tf.keras.layers.Dropout(0.01)
        #self.dense5 = tf.keras.layers.Dense(20, activation='tanh')
        #self.dropout5 = tf.keras.layers.Dropout(0.01)
        #self.dense6 = tf.keras.layers.Dense(20, activation='tanh')
        #self.dense7 = tf.keras.layers.Dense(20, activation='tanh')
        #self.dropout6 = tf.keras.layers.Dropout(0.01)
        self.out = tf.keras.layers.Dense(N_BASIS**2)

    def call(self, inputs):
        h = tf.reshape(inputs, [-1, 1])  
        x = self.dense1(h)
        x = self.dense2(x)
        #x = self.dense3(x)
        #x = self.dense4(x)
        #x = self.dense5(x)
        #x = self.dense6(x)
        #x = self.dense7(x)
        return self.out(x)

    
def calculate_norm(c, S):
    c_2d = tf.reshape(c, [N_BASIS, N_BASIS])
    norm = np.tensordot(c_2d, np.tensordot(S, c_2d, axes=([2, 3], [0, 1])), axes=([0, 1], [0, 1]))
    return norm

def train_step(h_values, epochs_per_h=2000, training=True):
    if training:
        for epoch in range(epochs_per_h):

            random.shuffle(h_values)

            for h in h_values:
                if training:
                    with tf.GradientTape() as tape:
                        h_tensor = tf.constant([h], dtype=tf.float32)
                        c = model(h_tensor)
                        loss = model_loss(c, h_tensor,epoch)

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                c_2d = tf.reshape(model(tf.constant([h], dtype=tf.float32)), (N_BASIS, N_BASIS))
                norm_value = calculate_norm(c_2d, cached_matrices[h][0])
                trained_cs.append((h, c_2d))
                norm_values.append(norm_value)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, h={h}, Loss: {loss.numpy()}")

    else:
        
        if isinstance(h_values, list):
            h = h_values[0] 
        else:
            h = h_values

        h_tensor = tf.constant([h], dtype=tf.float32)
        c_2d = tf.reshape(model(h_tensor), (N_BASIS, N_BASIS))
        #norm_value = calculate_norm(c_2d, S)
        return c_2d


model = PINN()
optimizer = Adam(learning_rate=0.001)


random_h_values_low = np.random.randint(3, 10, size=0)
random_h_values_middle = np.random.randint(10, 100, size=10)
random_h_values_high = np.random.randint(100, 1000, size=0)
random_h_values_very_high = np.random.randint(3000, 100000, size=0)
h_values = np.concatenate((random_h_values_low, random_h_values_middle,random_h_values_high,random_h_values_very_high))
h_values=[5,7,10,15,20,30,50,70,100,200,300,1000,10000,100000]

trained_cs = []
norm_values = []
train_step(h_values)


model_save_path = "model\ground_state"

tf.keras.models.save_model(model, model_save_path)



h_test = 600
print(f"Testing for h = {h_test}")
c_test = train_step(h_test, training=False)

predicted_psi = np.zeros_like(x)
for i in range(N_BASIS):
    for j in range(N_BASIS):
        predicted_psi += c_test[i, j] * psi(i, j, x)
#predicted_psi = predicted_psi * (1.0 / np.sqrt(norm_test))
        
def integrate_psi_squared(psi, x):
    return np.trapz(np.abs(psi) ** 2, x)


norm_factor = np.sqrt(integrate_psi_squared(predicted_psi, x))
predicted_psi = predicted_psi / norm_factor

psi_solution = solution(h_test,1)
psi_solution_minus = -psi_solution

plt.plot(x, predicted_psi, label="Predicted")
plt.plot(x, psi_solution ,"--", label="Answer")
plt.plot(x, psi_solution_minus ,"--", label="Answer")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("Coordinate $x$ [Bohr]")
plt.ylabel("Wave amplitude")
plt.legend()
plt.show(block=False)
plt.pause(0.1)  # 10秒表示
plt.close()  # 図を閉じる

