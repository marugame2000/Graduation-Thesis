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
from ground_state import ground_state_psi

N_BASIS = 3
X_CENTER_MIN = -0.5
X_CENTER_MAX = 0.5
X_MIN = -3.5
X_MAX = 3.5
N_SAMPLES = 3000
epochs=3000

#積分範囲を広げる

def  first_state_psi(h):

    c_ground = ground_state_psi(h,N_BASIS)

    t0=time.time()

    def psi(n, l, x):
        l = l + 1
        mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
        l = tf.cast(l / 20, tf.float32)
        coefficient = 1 / (l * tf.sqrt(2 * np.pi))
        exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
        return coefficient * tf.exp(exponent)


        #return tf.exp(-l * (x - mu_n)**2)
        #return tf.cast(tf.math.sin((n + 1) * np.pi * (x - X_MIN) / (X_MAX - X_MIN)) + tf.math.sin((l + 1) * np.pi * (x - X_MIN) / (X_MAX - X_MIN)), tf.float32)
    def deriv_phi(n, l, x):
        l = l + 1
        mu_n = tf.cast(X_CENTER_MIN + (X_CENTER_MAX - X_CENTER_MIN) * n / (N_BASIS - 1), tf.float32)
        l = tf.cast(l / 20, tf.float32)
        coefficient = 1 / (l * tf.sqrt(2 * np.pi))
        exponent = -(x - mu_n) ** 2 / (2 * l ** 2)
        normal_dist = coefficient * tf.exp(exponent)
        derivative = -(x - mu_n) / (l ** 2)
        return derivative * normal_dist

        
        return -2 * l * (x - mu_n) * psi(n, l, x)
        #dn = (n + 1) * np.pi * tf.math.cos((n + 1) * np.pi * (x - X_MIN) / (X_MAX - X_MIN)) / (X_MAX - X_MIN)
        #dl = -(l + 1) * np.pi * tf.math.sin((l + 1) * np.pi * (x - X_MIN) / (X_MAX - X_MIN)) / (X_MAX - X_MIN)
        #return dn + dl

    def V(x):
        return np.where((x < X_CENTER_MAX) & (x > X_CENTER_MIN), 0, h)
        #return (1/2) * x**2


    #ガウスの解析解を試す(np.erf)

    def trapezoidal_rule(f, a, b, n=200):
        h = (b - a) / n
        x = np.linspace(a, b, n)
        y = f(x)
        return h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))


    def calc_h(n, l, m, k):
        integrand = lambda x: (deriv_phi(n, l, x) * deriv_phi(m, k, x) / 2 + psi(n, l, x) * V(x) * psi(m, k, x))
        return trapezoidal_rule(integrand, X_MIN, X_MAX)

    def calc_s(n, l, m, k):
        integrand = lambda x: psi(n, l, x) * psi(m, k, x)
        return trapezoidal_rule(integrand, X_MIN, X_MAX)



    S = np.zeros((N_BASIS, N_BASIS, N_BASIS, N_BASIS), dtype=np.float32)
    H = np.zeros((N_BASIS, N_BASIS, N_BASIS, N_BASIS), dtype=np.float32)


    from tqdm import tqdm

    for i in tqdm(range(N_BASIS), desc="Progress for i"):
        #print(i)
        for j in tqdm(range(N_BASIS), desc="Progress for j", leave=False):
            #print(j)
            for k in tqdm(range(N_BASIS), desc="Progress for k", leave=False):
                for l in tqdm(range(N_BASIS), desc="Progress for l", leave=False):
                    S[i, j, k, l] = calc_s(i, j, k, l)
                    H[i, j, k, l] = calc_h(i, j, k, l)


    t1=time.time()

    def calc_energy(c):
        Hc = tf.tensordot(H, c, axes=([2, 3], [0, 1]))
        Sc = tf.tensordot(S, c, axes=([2, 3], [0, 1]))
        cHc = tf.tensordot(c, Hc, axes=([0, 1], [0, 1]))
        cSc = tf.tensordot(c, Sc, axes=([0, 1], [0, 1]))
        return cHc / cSc

    def orthogonality_penalty(c):
        c = tf.reshape(c, [N_BASIS, N_BASIS])

        predicted_psi = tf.zeros_like(x, dtype=tf.float32) 
        for i in range(N_BASIS):
            for j in range(N_BASIS):
                predicted_psi += c[i, j] * psi(i, j, x)

        predicted_psi_ground = tf.zeros_like(x, dtype=tf.float32) 
        for i in range(N_BASIS):
            for j in range(N_BASIS):
                predicted_psi_ground += c_ground[i, j] * psi(i, j, x)
        overlap = tf.tensordot(predicted_psi, predicted_psi_ground, axes=1)
        return overlap**2

    def model_loss(_, c):
        energy_loss = calc_energy(c)

        return energy_loss + orthogonality_penalty(c) * 3e-2

    class CustomNormalizationLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(CustomNormalizationLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return inputs / K.sum(inputs)

    model = Sequential([
        Dense(512, input_dim=2, activation=LeakyReLU(alpha=0.3)),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dense(256, activation=LeakyReLU(alpha=0.3)),
        Dense(128, activation=LeakyReLU(alpha=0.3)),
        Dense(128, activation=LeakyReLU(alpha=0.3)),
        Dense(128, activation=LeakyReLU(alpha=0.3)),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(64, activation=LeakyReLU(alpha=0.3)),
        Dense(1, activation="linear"),
    ])


    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=model_loss, optimizer=optimizer)

    t2=time.time()

    x = np.linspace(X_MIN, X_MAX, N_SAMPLES)
    index_n = np.arange(N_BASIS)
    index_l = np.arange(N_BASIS)
    index = np.array(np.meshgrid(index_n, index_l)).T.reshape(-1, 2)
    dummy = np.zeros((len(index), 1))

    #model.fit(index, dummy, epochs=50000, steps_per_epoch=1, verbose=1, shuffle=False)

    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=3e-6, verbose=1)

    results = model.fit(
        index, 
        dummy, 
        epochs=epochs, 
        steps_per_epoch=1, 
        verbose=1, 
        shuffle=False, 
        callbacks=[reduce_lr]
    )

    c = model.predict(index).reshape(N_BASIS, N_BASIS)
    norm = np.tensordot(c, np.tensordot(S, c, axes=([2, 3], [0, 1])), axes=([0, 1], [0, 1]))
    print(c)
    print(K.sum(c))
    energy = calc_energy(c)

    x_np = x
    predicted_psi = np.zeros_like(x_np)
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            predicted_psi += c[i, j] * psi(i, j, x_np)
    predicted_psi = predicted_psi * (1.0 / np.sqrt(norm))
    #predicted_psi = predicted_psi / np.sqrt(np.sum(predicted_psi ** 2)) 

    print(np.sqrt(np.sum(predicted_psi ** 2)))

    second_excited_answer = np.sqrt(2) * np.sin(np.pi * x_np)


    import matplotlib.pyplot as plt
    plt.plot(x_np, predicted_psi)
    plt.plot(x_np, second_excited_answer, "--", label="Answer")
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("Coordinate $x$ [Bohr]")
    plt.ylabel("Wave amplitude")
    plt.show()

    t3=time.time()

    print("t1-t0 = %e"%(t1-t0))
    print("t2-t1 = %e"%(t2-t1))
    print("t3-t2 = %e"%(t3-t2))

    return c

first_state_psi(1e2)