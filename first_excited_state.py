import numpy as np
from numpy import pi, sin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, LeakyReLU
import matplotlib.pyplot as plt

# 定数の定義
N = 5000  # 格子点の数
epochs = 3000  # 学習のエポック数

# 損失関数の重みの定義
orthogonality_penalty_weight = 2e-8  # 直交性ペナルティの重み
edge_penalty_weight = 1e5  # 端のペナルティの重み
loss_adjustment_weight = 1e-2  # 損失調整の重み

# x軸の値と目標の波動関数
x = np.linspace(0, 1, N)
ground_state = np.sqrt(2) * sin(pi * x)
first_excited = np.sqrt(2) * sin(pi * 2 * x)
first_excited_minus = -first_excited

# 波動関数の対称化（ここでは対称化は行っていない）
def psi(y):
    return y

# 波動関数の微分
def dpsi(y):
    y_shifted_f = tf.roll(y, shift=-1, axis=0)
    y_shifted_b = tf.roll(y, shift=1, axis=0)
    dy = (y_shifted_f - y_shifted_b) / 2
    return dy

# バリエーショナルエネルギー計算
def variationalE(y_true, y_pred):
    wave = psi(y_pred)
    wave_nom = K.l2_normalize(wave, axis=0)
    dwave = dpsi(wave_nom)
    kinetic_energy = N ** 2 * K.sum(K.square(dwave)) / pi ** 2

    orthogonality = K.sum(ground_state * wave_nom)
    orthogonality_penalty = orthogonality ** 2 * orthogonality_penalty_weight

    edge_penalty = (K.square(y_pred[0]) + K.square(y_pred[-1])) * edge_penalty_weight

    return kinetic_energy + orthogonality_penalty + edge_penalty

# ニューラルネットワークモデルの構築
model = Sequential([
    Dense(1024, input_dim=1, activation=LeakyReLU(alpha=0.3)),
    Dense(512, activation=LeakyReLU(alpha=0.3)),
    Dense(1, activation="linear")
])

# ロス関数を調整
def adjusted_variationalE(y_true, y_pred):
    original_loss = variationalE(y_true, y_pred)
    loss_adjustment = loss_adjustment_weight * (original_loss - 4) ** 2
    return original_loss + loss_adjustment - 4

model.compile(loss=adjusted_variationalE, optimizer="Adam")
model.summary()

# 学習率の減衰
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=1e-5, verbose=1)

# 学習の実行
results = model.fit(x, first_excited, epochs=epochs, steps_per_epoch=1, verbose=1, shuffle=False, callbacks=[reduce_lr])

# 予測と描画
pred = model.predict(x)
func = psi(pred)
func = func / np.sqrt(np.sum(func ** 2) / N)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlim(0, 1)
plt.plot(x, func, label="fitted")
plt.plot(x, first_excited, "--", label="answer")
plt.plot(x, first_excited_minus,  "--",label="answer_minus")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\Psi(x)$")

# Plotting loss
plt.subplot(1, 2, 2)
plt.ylim(0, 10)
plt.plot(results.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')

#plt.tight_layout()
plt.show()
