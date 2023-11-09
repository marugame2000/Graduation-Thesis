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


N = 5000  # 格子点の数
epochs = 3000  # 学習のエポック数

orthogonality_penalty_weight = 5e-8  # 直交性ペナルティの重み
edge_penalty_weight = 1e9 # 端のペナルティの重み
normalization_penalty_weight=1e4

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
    kinetic_energy = N ** 2 * K.sum(K.square(dwave)) / pi ** 2

    orthogonality_ground = K.sum(ground_state * wave_nom)
    orthogonality_first = K.sum(first_excited * wave_nom)
    orthogonality_penalty = (orthogonality_ground ** 2 + orthogonality_first ** 2) * orthogonality_penalty_weight

    edge_penalty = (K.square(y_pred[0]) + K.square(y_pred[-1])) * edge_penalty_weight

    # 正規化条件ペナルティ
    normalization_penalty = K.square(K.sum(K.square(wave_nom)) - 1) * normalization_penalty_weight

    return kinetic_energy  + orthogonality_penalty + edge_penalty + normalization_penalty

# ニューラルネットワークモデルの構築
model = Sequential([
    Dense(256, input_dim=1, activation=LeakyReLU(alpha=0.3)),
    Dense(256, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(128, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(64, activation=LeakyReLU(alpha=0.3)),
    Dense(1, activation="linear")
])

# モデルのコンパイル
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate, epsilon=1e-9)
model.compile(loss=variationalE, optimizer=optimizer)

# モデルサマリーの表示
model.summary()

# 学習率低減のコールバック設定
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=5e-6, verbose=1)

# モデルの訓練
results = model.fit(
    x, 
    second_excited_answer, 
    epochs=epochs, 
    steps_per_epoch=1, 
    verbose=1, 
    shuffle=False, 
    callbacks=[reduce_lr]
)

# 予測と描画
pred = model.predict(x)
func = psi(pred)
func = func*1500
#func = func / np.sqrt(np.sum(func ** 2) / N) 

# 結果のプロット
plt.figure(figsize=(10, 5))  # サイズ調整をすることで下部にスペースを作る
plt.subplot(1, 2, 1)  # 2行1列の1番目のプロットとして設定
plt.xlim(0, 1)
plt.plot(x, func, label="Fitted")
plt.plot(x, second_excited_answer, "--", label="Answer")
plt.plot(x, second_excited_answer_minus, "--", label="Answer Minus")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")

# 損失のプロット（片対数グラフ）
plt.subplot(1,2,2)  # 2行1列の2番目のプロットとして設定
plt.ylim(0, 10)
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

