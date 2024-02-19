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

# 既存の定数と関数の設定
X_CENTER_MIN = -0.5
X_CENTER_MAX = 0.5
X_MIN = -5
X_MAX = 5
N_SAMPLES = 3000
epochs = 30000
N_BASIS = 9
x = np.linspace(X_MIN, X_MAX, N_SAMPLES)

def psi(n, l, x):
    l = l + 1
    mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
    l = tf.cast(l / 20, tf.float32)
    x = tf.cast(x, tf.float32)
    coefficient = 1 / (l * tf.sqrt(2 * np.pi))
    exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
    return coefficient * tf.exp(exponent)

model_save_path = "model"
loaded_model = tf.keras.models.load_model(model_save_path)

def integrate_psi_squared(psi, x):
    return np.trapz(np.abs(psi) ** 2, x)

# 仮定する関数「solution」の存在を前提として
# 実際のコードでは、solution関数がどのように定義されているかによって変更が必要

# 10〜100と100〜1000の範囲からランダムなh値を生成
random_h_values_low = np.random.randint(10, 100, size=10)
random_h_values_high = np.random.randint(100, 1000, size=10)
random_h_values = np.concatenate((random_h_values_low, random_h_values_high))

# 各h値に対する差のリストを初期化
differences = []
prediction_times = []

# 各h値に対して処理を実行
for h_test in random_h_values:
    print(f"Predicting wave function for h = {h_test}...")  # 現在のh値を表示

    h_test_tensor = tf.constant([h_test], dtype=tf.float32)

    start_time = time.time()  # 予測開始時間

    # モデルを使用して波動関数を予測
    c_test = tf.reshape(loaded_model(h_test_tensor), (N_BASIS, N_BASIS))

    end_time = time.time()  # 予測終了時間
    prediction_time = end_time - start_time
    prediction_times.append(prediction_time)

    # 予測された波動関数の計算
    predicted_psi = np.zeros_like(x)
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            predicted_psi += c_test[i, j] * psi(i, j, x).numpy()

    # 波動関数の正規化
    norm_factor = np.sqrt(integrate_psi_squared(predicted_psi, x))
    predicted_psi /= norm_factor

    # 解析解の取得
    psi_solution = solution(h_test, 1)  # 'solution'関数の詳細に応じて調整

    # 差の計算
    difference = np.mean(np.abs(predicted_psi - psi_solution))
    # h値と差をタプルとしてリストに追加
    differences.append((h_test, difference.item()))

# 平均差の計算
average_difference = np.mean(differences)
average_prediction_times = np.mean(prediction_times)


for h_value, diff in differences:
    print(f"Difference for h = {h_value}: {diff}")
print(f"Mean Absolute Difference: {prediction_times}")
print(f"Mean Absolute average Difference: {average_prediction_times}")

