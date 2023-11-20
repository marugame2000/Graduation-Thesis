import numpy as np
from numpy import pi, sin, sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, LeakyReLU,BatchNormalization
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

N =6000 
epochs = 10000

kinetic_energy_weight=1
orthogonality_penalty_weight = 5e-7
edge_penalty_weight = 1e8
symmetry_penalty_weight = 2e5

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
    #y_symmetrized = y
    return y_symmetrized

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

class TrainingHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.orthogonality_penalties = []
        self.orthogonality_penalties2 = []
        self.kinetic_energies = []
        self.wavefunction_difference_penalty = []

    def on_epoch_end(self, epoch, logs={}):
        self.orthogonality_penalties.append(K.get_value(orthogonality_penalty_global))
        self.orthogonality_penalties2.append(K.get_value(orthogonality_penalty2_global))
        self.kinetic_energies.append(K.get_value(kinetic_energy_global))
        self.wavefunction_difference_penalty.append(K.get_value(wavefunction_difference_penalty_global))

training_history = TrainingHistory()


# グローバル変数を定義
orthogonality_penalty_global = K.variable(0.0)
orthogonality_penalty2_global = K.variable(0.0)
kinetic_energy_global = K.variable(0.0)
wavefunction_difference_penalty_global= K.variable(0.0)

def variationalE(y_true, y_pred):

    global orthogonality_penalty_global
    global wavefunction_difference_penalty_global
    wave = psi(y_pred)
    #wave=y_pred
    #wave_nom=wave
    #wave_nom = wave / K.sqrt(K.sum(K.square(wave)) / N)
    wave_nom = K.l2_normalize(wave, axis=0)
    dwave = dpsi(wave_nom)
    kinetic_energy = N ** 2 * K.sum(K.square(dwave)) / pi ** 2  * kinetic_energy_weight
    kinetic_energy_global.assign(kinetic_energy)

        # y_predの左半分と右半分を取得
    left_half = y_pred[:N // 2]
    right_half = y_pred[N // 2:]

    # 右半分を逆順にして左半分と比較
    right_half_reversed = K.reverse(right_half, axes=0)
    
    # 対称性の違反を計算（左右の差の2乗の和）
    #symmetry_penalty = K.sum(K.square(left_half - right_half_reversed)) * symmetry_penalty_weight

    max_idx = K.max(wave_nom)
    min_idx = K.min(wave_nom)
    symmetry_penalty = K.square(K.abs(max_idx) - K.abs(min_idx) )* symmetry_penalty_weight


    orthogonality_ground = K.sum(ground_state * wave_nom) * ground_state_weight
    orthogonality_first = K.sum(first_excited * wave_nom) * first_state_weight
    orthogonality_first_minus = K.sum(first_excited * wave_nom) * first_state_weight
    orthogonality_second = K.sum(second_excited_answer * wave_nom) * first_state_weight
    orthogonality_second_minus = K.sum(second_excited_answer_minus * wave_nom) * first_state_weight

    orthogonality_penalty = (orthogonality_ground ** 2 + orthogonality_first ** 2 + orthogonality_first_minus** 2 ) * orthogonality_penalty_weight
    orthogonality_penalty_global.assign(orthogonality_penalty)

    edge_penalty = (K.square(y_pred[0]) + K.square(y_pred[-1])) * edge_penalty_weight
    
    orthogonality_penalty2 = K.square(K.abs(max_idx) - K.abs(min_idx) )* symmetry_penalty_weight
    orthogonality_penalty2_global.assign(orthogonality_penalty2)

    node1 = N // 3
    node2 = 2 * N // 3
    node_penalty = (K.square(wave_nom[node1]) + K.square(wave_nom[node2])) * 1e7
    
    #wave_nom = K.l2_normalize(wave, axis=0)
    wavefunction_difference_penalty = K.sum(K.square(wave_nom - first_excited))*1e-2
    wavefunction_difference_penalty_global.assign(wavefunction_difference_penalty)
    
    #normalization_penalty = K.square(K.sum(K.square(wave_nom)) - 1) * normalization_penalty_weight
    
    #そもそも正規化できているという可能性

    normalization_penalty = K.square(K.sum(K.square(wave_nom))/N - 1) * normalization_penalty_weight

    return kinetic_energy + orthogonality_penalty + edge_penalty + symmetry_penalty + node_penalty


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

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=3e-6, verbose=1)

results = model.fit(
    x, 
    second_excited_answer, 
    epochs=epochs, 
    steps_per_epoch=1, 
    verbose=1, 
    shuffle=False, 
    callbacks=[reduce_lr,  training_history]
)

pred = model.predict(x)
func = psi(pred)
#func=pred
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
plt.ylim(8, 12)
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

plt.plot(training_history.orthogonality_penalties)
plt.xlabel('Epochs')
plt.ylabel('Orthogonality Penalty')
plt.ylim(0, 30)
plt.title('Orthogonality Penalty During Training')
plt.show()

plt.plot(training_history.orthogonality_penalties2)
plt.xlabel('Epochs')
plt.ylabel('symmetry_penalty')
plt.ylim(0, 10)
plt.title('Orthogonality Penalty During Training')
plt.show()

plt.plot(training_history.kinetic_energies)
plt.xlabel('Epochs')
plt.ylabel('Kinetic Energy')
plt.ylim(0, 10)
plt.title('Kinetic Energy During Training')
plt.show()

#plt.plot(training_history.wavefunction_difference_penalty)
#plt.xlabel('Epochs')
#plt.ylabel('Kinetic Energy')
#plt.ylim(0, 100)
#plt.title('Kinetic Energy During Training')
#plt.show()
