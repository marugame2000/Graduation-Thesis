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

# 定数定義
N = 3000
x = np.linspace(0, 1, N)
# 目標とする波動関数の定義
ground_state = np.sqrt(2) * sin(pi * x)
first_excited = np.sqrt(2) * sin(2 * pi * x)
first_excited_minus = -first_excited
second_excited_answer = np.sqrt(2) * sin(3 * pi * x)
second_excited_answer_minus = -second_excited_answer
third_excited_answer = np.sqrt(2) * sin(4 * pi * x)
fourth_excited_answer = np.sqrt(2) * sin(5 * pi * x)

# 波動関数の対称化
def psi(y):
    y_rev = K.reverse(y, axes=0)
    y_symmetrized = 0.5 * (y + y_rev)
    return y_symmetrized

# 波動関数の導関数
def dpsi(y):
    y_shifted_f = tf.roll(y, shift=-1, axis=0)
    y_shifted_b = tf.roll(y, shift=+1, axis=0)
    dy = (y_shifted_f - y_shifted_b) / 2
    return dy

# バリエーショナルエネルギーの計算
def variationalE(y_true, y_pred):
    wave = psi(y_pred)
    wave_nom = K.l2_normalize(wave, axis=0)
    dwave = dpsi(wave_nom)
    kinetic_energy = N**2 * K.sum(K.square(dwave)) / pi**2

    # 直交性ペナルティの計算
    orthogonality_penalty = sum([
        (K.sum(state * wave_nom))**2
        for state in [ground_state, first_excited, first_excited_minus]
    ])

    # 端のペナルティの計算
    edge_penalty = K.square(y_pred[0]) + K.square(y_pred[-1])

    # 総エネルギーを返す
    return kinetic_energy + orthogonality_penalty * 5e-8 + edge_penalty * 1e9

# ニューラルネットワークモデルの構築
model = Sequential([
    Dense(256, input_dim=1),
    LeakyReLU(alpha=0.3),
    Dense(128),
    LeakyReLU(alpha=0.3),
    Dense(128),
    LeakyReLU(alpha=0.3),
    Dense(128),
    LeakyReLU(alpha=0.3),
    Dense(64),
    LeakyReLU(alpha=0.3),
    Dense(64),
    LeakyReLU(alpha=0.3),
    Dense(1, activation="linear")
])

# モデルのコンパイル
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate, epsilon=1e-9)
model.compile(loss=variationalE, optimizer=optimizer)

# モデルサマリーの表示
model.summary()

# 学習率低減のコールバック設定
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-5, verbose=1)

# モデルの訓練
results = model.fit(
    x, 
    second_excited_answer, 
    epochs=5000, 
    steps_per_epoch=1, 
    verbose=1, 
    shuffle=False, 
    callbacks=[reduce_lr]
)

# 予測結果の取得
pred = model.predict(x)
func = psi(pred)
func /= np.sqrt(np.sum(func**2) / N)

# 結果のプロット
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlim(0, 1)
plt.plot(x, func, label="Fitted")
plt.plot(x, second_excited_answer, label="Answer")
plt.plot(x, second_excited_answer_minus, label="Answer Minus")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")

# 損失のプロット
plt.subplot(1, 2, 2)
plt.ylim(0, 10)
plt.plot(results.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
