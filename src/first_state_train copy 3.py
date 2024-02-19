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
N_BASIS = 9
x = np.linspace(X_MIN, X_MAX, N_SAMPLES)

def integrate_psi_squared(psi, x):
    return np.trapz(np.abs(psi) ** 2, x)

def psi(n, l, x):
    l = l + 1
    mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
    l = tf.cast(l / 20, tf.float32)
    x = tf.cast(x, tf.float32)  # xをfloat32にキャスト
    coefficient = 1 / (l * tf.sqrt(2 * np.pi))
    exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
    return coefficient * tf.exp(exponent)


def deriv_phi(n, l, x):
    l = l + 1
    mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
    l = tf.cast(l / 20, tf.float32)
    x = tf.cast(x, tf.float32)  # xをfloat32にキャスト
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
    h_value = h_tensor.numpy()[0]  # テンソルから数値を取得
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

def model_loss(c, h,epoch):
    S, H = calculate_matrices(h)  
    c = tf.reshape(c, [N_BASIS, N_BASIS])

    def calc_energy(c):
        
        Hc = tf.tensordot(H, c, axes=([2, 3], [0, 1]))
        Sc = tf.tensordot(S, c, axes=([2, 3], [0, 1]))
        cHc = tf.tensordot(c, Hc, axes=([0, 1], [0, 1]))
        cSc = tf.tensordot(c, Sc, axes=([0, 1], [0, 1]))
        return cHc / cSc
    

    base_multiplier = 1e-7
    multiplier_factor = 10 ** (epoch // 1000)  # 200エポックごとに10倍になる
    Lagrange_multiplier = base_multiplier * multiplier_factor

    def orthogonality_penalty(c,h):

        h_value = h.numpy()[0]  

            # Lagrange乗数を計算


        predicted_psi = tf.zeros_like(x, dtype=tf.float32) 
        for i in range(N_BASIS):
            for j in range(N_BASIS):
                predicted_psi += c[i, j] * psi(i, j, x)
        predicted_psi = K.l2_normalize(predicted_psi, axis=0)
        norm_factor = np.sqrt(integrate_psi_squared(predicted_psi, x))
        predicted_psi /= norm_factor
        

        predicted_psi_ground = get_cached_solution(h_value, 1)
        psi_solution = get_cached_solution(h_value, 2)
        psi_solution_minus = -psi_solution
        #predicted_psi_ground = K.l2_normalize(predicted_psi_ground, axis=0)
        predicted_psi = tf.cast(predicted_psi, tf.float32)  # tf.float32にキャスト
        predicted_psi_ground = tf.cast(predicted_psi_ground, tf.float32)  # tf.float32にキャスト
        overlap = tf.tensordot(predicted_psi, predicted_psi_ground, axes=1)
        if epoch%50==0:
            print(h_value)
            print(Lagrange_multiplier)
            plt.plot(x, predicted_psi, label="Predicted")
            #plt.plot(x, predicted_psi_ground ,"--", label="Answer")
            plt.plot(x, psi_solution ,"--", label="Answer")
            plt.plot(x, psi_solution_minus ,"--", label="Answer")
            plt.xlim(-2, 2)
            plt.xlabel("Coordinate $x$ [Bohr]")
            plt.ylabel("Wave amplitude")
            plt.legend()
            plt.show()
        print(overlap**2)
        return overlap**2



    energy_loss = calc_energy(c) + orthogonality_penalty(c,h) * Lagrange_multiplier
    return energy_loss

from keras import regularizers

class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=LeakyReLU(alpha=0.3),
                                            kernel_regularizer=regularizers.l2(0.001))
        self.dropout1 = tf.keras.layers.Dropout(0.01)
        self.dense2 = tf.keras.layers.Dense(256, activation=LeakyReLU(alpha=0.3),
                                            kernel_regularizer=regularizers.l2(0.001))
        self.dropout2 = tf.keras.layers.Dropout(0.01)
        # 以下、同様に他の層にも正則化とドロップアウトを追加
        self.dense3 = tf.keras.layers.Dense(256, activation=LeakyReLU(alpha=0.3),
                                            kernel_regularizer=regularizers.l2(0.001))
        self.dropout3 = tf.keras.layers.Dropout(0.01)
        self.dense4 = tf.keras.layers.Dense(128, activation=LeakyReLU(alpha=0.3),
                                            kernel_regularizer=regularizers.l2(0.001))
        self.dropout4 = tf.keras.layers.Dropout(0.01)
        self.dense5 = tf.keras.layers.Dense(128, activation=LeakyReLU(alpha=0.3),
                                            kernel_regularizer=regularizers.l2(0.001))
        self.dropout5 = tf.keras.layers.Dropout(0.01)
        self.dense6 = tf.keras.layers.Dense(64, activation=LeakyReLU(alpha=0.3),
                                            kernel_regularizer=regularizers.l2(0.001))
        self.dropout6 = tf.keras.layers.Dropout(0.01)
        self.out = tf.keras.layers.Dense(N_BASIS**2)

    def call(self, inputs):
        h = tf.reshape(inputs, [-1, 1])  
        x = self.dropout1(self.dense1(h))
        x = self.dropout2(self.dense2(x))
        x = self.dropout3(self.dense3(x))
        x = self.dropout4(self.dense4(x))
        x = self.dropout5(self.dense5(x))
        x = self.dropout6(self.dense6(x))
        return self.out(x)


def calculate_norm(c, S):
    c_2d = tf.reshape(c, [N_BASIS, N_BASIS])
    norm = np.tensordot(c_2d, np.tensordot(S, c_2d, axes=([2, 3], [0, 1])), axes=([0, 1], [0, 1]))
    return norm

# 各エポックの損失の合計を記録するためのリスト
total_loss_per_epoch = []

def train_step(h_values_batch, epochs_per_h=4000, training=True):
    global total_loss_per_epoch
    if training:
        for epoch in range(epochs_per_h):
            total_loss = 0

            for h_batch in h_values_batch:
                with tf.GradientTape() as tape:
                    losses = []
                    for h in h_batch:
                        h_tensor = tf.constant([h], dtype=tf.float32)
                        c = model(h_tensor)
                        loss = model_loss(c, h_tensor, epoch)
                        losses.append(loss)

                    # バッチ内の損失の平均を計算
                    batch_loss = tf.reduce_mean(tf.stack(losses))

                grads = tape.gradient(batch_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                total_loss += batch_loss.numpy()

            total_loss_per_epoch.append(total_loss)
            print(f"Epoch {epoch}, Total Loss: {total_loss}")

    else:
        # テストの場合の処理
        if isinstance(h_values, list):
            h = h_values[0]  # リストの最初の要素を使用
        else:
            h = h_values  # h_values が単一の数値の場合

        h_tensor = tf.constant([h], dtype=tf.float32)
        c_2d = tf.reshape(model(h_tensor), (N_BASIS, N_BASIS))
        return c_2d

# モデルとオプティマイザーのインスタンス化
model = PINN()
optimizer = Adam(learning_rate=0.001)

# 10〜100と100〜1000の範囲からランダムなh値を生成
random_h_values_low = np.random.randint(3, 10, size=0)
random_h_values_middle = np.random.randint(10, 100, size=1)
random_h_values_high = np.random.randint(100, 1000, size=2)
random_h_values_very_high = np.random.randint(3000, 100000, size=1)
h_values = np.concatenate((random_h_values_low, random_h_values_middle,random_h_values_high,random_h_values_very_high))

# バッチを作成
batch_size = 2  # バッチサイズを設定
h_values_batches = [h_values[i:i + batch_size] for i in range(0, len(h_values), batch_size)]

# トレーニングステップを実行
train_step(h_values_batches)

plt.figure(figsize=(10, 6))
plt.plot(total_loss_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss over Epochs')
plt.grid(True)
plt.show()

# モデルの保存先ディレクトリを指定
model_save_path = "model\sfirst_state"
# モデルを保存
tf.keras.models.save_model(model, model_save_path)


# テスト用の h 値
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

# 波動関数の正規化
norm_factor = np.sqrt(integrate_psi_squared(predicted_psi, x))
predicted_psi = predicted_psi / norm_factor

psi_solution = solution(h_test,2)
psi_solution_minus = -psi_solution

# 結果のプロット
plt.plot(x, predicted_psi, label="Predicted")
plt.plot(x, psi_solution ,"--", label="Answer")
plt.plot(x, psi_solution_minus ,"--", label="Answer")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("Coordinate $x$ [Bohr]")
plt.ylabel("Wave amplitude")
plt.legend()
plt.show()

