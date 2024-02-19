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

X_CENTER_MIN = -0.5
X_CENTER_MAX = 0.5
X_MIN = -5
X_MAX = 5
N_SAMPLES = 3000
epochs = 30000
N_BASIS = 11
x = np.linspace(X_MIN, X_MAX, N_SAMPLES)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_folder = "logs"
session_folder = os.path.join(log_folder, timestamp)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

def psi(n, l, x):
    l = l + 1
    mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
    l = tf.cast(l / 20, tf.float32)
    x = tf.cast(x, tf.float32)
    coefficient = 1 / (l * tf.sqrt(2 * np.pi))
    exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
    return coefficient * tf.exp(exponent)

model_save_path = "model\ground_state2"
loaded_model = tf.keras.models.load_model(model_save_path)

def integrate_psi_squared(psi, x):
    return np.trapz(np.abs(psi) ** 2, x)


random_h_values_low = np.random.randint(3, 10, size=3)
random_h_values_middle = np.random.randint(10, 100, size=4)
random_h_values_high = np.random.randint(100, 1000, size=3)
random_h_values = np.concatenate((random_h_values_low, random_h_values_middle,random_h_values_high))
random_h_values=[4,5,6,7,8,9,10,15,20,30,50,70,100,200,300,1000,10000,100000,500000]


differences = []
prediction_times = []


for h_test in random_h_values:
    print(f"Predicting wave function for h = {h_test}...")  

    h_test_tensor = tf.constant([h_test], dtype=tf.float32)

    start_time = time.time() 


    c_test = tf.reshape(loaded_model(h_test_tensor), (N_BASIS, N_BASIS))

    end_time = time.time()  # 予測終了時間
    prediction_time = end_time - start_time
    prediction_times.append(prediction_time)


    predicted_psi = np.zeros_like(x)
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            predicted_psi += c_test[i, j] * psi(i, j, x).numpy()


    norm_factor = np.sqrt(integrate_psi_squared(predicted_psi, x))
    predicted_psi /= norm_factor

    predicted_psi=-predicted_psi
    psi_solution =solution(h_test, 1)  # 'solution'関数の詳細に応じて調整


    plt.plot(x, predicted_psi, label="Predicted", linewidth=2.5)  # 線の太さを2に設定
    plt.plot(x, psi_solution, "--", label="Answer", linewidth=2.5)  # 線の太さを2に設定
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("Wave amplitude", fontsize=18)
    plt.legend(fontsize=12)

    # 座標軸の目盛りの文字サイズを大きくする
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.show()


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


log_data = {
    'differences': [(int(h), float(diff)) for h, diff in differences],
    'prediction_times': [float(time) for time in prediction_times]
}
# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Log data with a unique filename
log_filename = os.path.join(session_folder, "log.json")

with open(log_filename, 'w') as log_file:
    json.dump(log_data, log_file, indent=4)

