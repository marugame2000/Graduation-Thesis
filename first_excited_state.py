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
epochs = 50000  # 学習のエポック数

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
    Dense(2048, input_dim=1, activation=LeakyReLU(alpha=0.3)),
    Dense(1024, activation=LeakyReLU(alpha=0.3)),
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
# 結果のプロット
plt.figure(figsize=(10, 5))  # サイズ調整をすることで下部にスペースを作る
plt.subplot(1, 2, 1)  # 2行1列の1番目のプロットとして設定
plt.xlim(0, 1)
plt.plot(x, func, label="Fitted")
plt.plot(x, first_excited, "--", label="Answer")
plt.plot(x, first_excited_minus, "--", label="Answer Minus")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")

# 損失のプロット（片対数グラフ）
plt.subplot(1,2,2)  # 2行1列の2番目のプロットとして設定
plt.yscale('log')  # 縦軸を対数スケールに設定
plt.ylim(0, 50)
plt.plot(results.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss (Log Scale)')

# 最後の100エポックの損失値を取得
last_100_epochs_loss = results.history['loss'][-100:]

# 最後の100エポックの平均損失値を計算
average_loss_last_100 = np.mean(last_100_epochs_loss)

# 最後の100エポックの平均損失値をグラフに追加
plt.text(0.5, 0.95, f'Avg. Loss (Last 100 Epochs): {average_loss_last_100:.4e}', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)

# グラフ全体にテキスト情報を追加
info_text = (f'N: {N}\n'
             f'Epochs: {epochs}\n'
             f'Orthogonality Penalty Weight: {orthogonality_penalty_weight:e}\n'
             f'Edge Penalty Weight: {edge_penalty_weight:e}\n'
             f'Avg. Loss (Last 100 Epochs): {average_loss_last_100:.4e}')
plt.figtext(0.5, 0.05, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

# 表示前にレイアウトを調整
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rectパラメータで図の余白を調整

# 表示
plt.show()

