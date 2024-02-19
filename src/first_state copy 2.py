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
from scipy.optimize import minimize, Bounds
from libs.ground_state import ground_state_psi

#N_BASIS = 3
X_CENTER_MIN = -0.5
X_CENTER_MAX = 0.5
X_MIN = -5
X_MAX = 5
N_SAMPLES = 3000
epochs=10000

#積分範囲を広げる

def  first_state_psi(h,N_BASIS):
    
    start_time = time.time()

    # 日付と時間に基づいてサブフォルダを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = "logs"
    session_folder = os.path.join(log_folder, timestamp)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)

    #理論解を求める
    psi_solution = solution(h,2)
    psi_solution_minus = -psi_solution

    #基底状態の波動関数を求める
    c_ground = ground_state_psi(h,N_BASIS)
    print(c_ground.dtype)

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

    def calc_energy(c_flat):
        c = c_flat.reshape(N_BASIS, N_BASIS)
        c = tf.cast(c, dtype=tf.float32)
        #print(c.dtype)
        Hc = np.tensordot(H, c, axes=([2, 3], [0, 1]))
        Sc = np.tensordot(S, c, axes=([2, 3], [0, 1]))
        cHc = np.tensordot(c, Hc, axes=([0, 1], [0, 1]))
        cSc = np.tensordot(c, Sc, axes=([0, 1], [0, 1]))
        return cHc / cSc

    def orthogonality_penalty(c):
        c = tf.reshape(c, [N_BASIS, N_BASIS])
        c = tf.cast(c, dtype=tf.float32)

        predicted_psi = tf.zeros_like(x, dtype=tf.float32) 
        for i in range(N_BASIS):
            for j in range(N_BASIS):
                predicted_psi += c[i, j] * psi(i, j, x)
        predicted_psi = K.l2_normalize(predicted_psi, axis=0)

        predicted_psi_ground = tf.zeros_like(x, dtype=tf.float32) 
        for i in range(N_BASIS):
            for j in range(N_BASIS):
                predicted_psi_ground += c_ground[i, j] * psi(i, j, x)
        predicted_psi_ground = K.l2_normalize(predicted_psi_ground, axis=0)
        overlap = tf.tensordot(predicted_psi, predicted_psi_ground, axes=1)
        return overlap**2

    def model_loss(c):
        energy_loss = calc_energy(c)
        penalty_loss=orthogonality_penalty(c)
        #print(energy_loss.dtype)
        #print(penalty_loss.dtype)
        return energy_loss + penalty_loss * 1e7

    t2=time.time()

    x = np.linspace(X_MIN, X_MAX, N_SAMPLES)
    index_n = np.arange(N_BASIS)
    index_l = np.arange(N_BASIS)
    index = np.array(np.meshgrid(index_n, index_l)).T.reshape(-1, 2)
    dummy = np.zeros((len(index), 1))


    def norm_constraint(c_flat):
        c = c_flat.reshape(N_BASIS, N_BASIS)
        norm = np.tensordot(c, np.tensordot(S, c, axes=([2, 3], [0, 1])), axes=([0, 1], [0, 1]))
        return np.sqrt(norm) - 1

    cons = {'type': 'eq', 'fun': norm_constraint}

    lower_bounds = -np.inf * np.ones(N_BASIS * N_BASIS) 
    upper_bounds = np.inf * np.ones(N_BASIS * N_BASIS) 
    bounds = Bounds(lower_bounds, upper_bounds)
    
    c_initial = np.random.rand(N_BASIS * N_BASIS)
    from scipy.optimize import minimize

    result = minimize(model_loss, c_initial,bounds=bounds, constraints=cons, options={'maxiter': 200}, method='trust-constr')

    c = result.x.reshape(N_BASIS, N_BASIS)
    norm = np.tensordot(c, np.tensordot(S, c, axes=([2, 3], [0, 1])), axes=([0, 1], [0, 1]))
    energy = result.fun

    print(c)
    print(orthogonality_penalty(c))

    x_np = x
    predicted_psi = np.zeros_like(x_np)
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            predicted_psi += c[i, j] * psi(i, j, x_np)
    predicted_psi = predicted_psi * (1.0 / np.sqrt(norm))
    #predicted_psi = predicted_psi / np.sqrt(np.sum(predicted_psi ** 2)) 

    print(np.sqrt(np.sum(predicted_psi ** 2)))

    second_excited_answer = np.sqrt(2) * np.sin(np.pi * x_np)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_filename = os.path.join(session_folder, "graph.png")

    import matplotlib.pyplot as plt
    plt.plot(x_np, predicted_psi)
    #plt.plot(x_np, second_excited_answer, "--", label="Answer")
    plt.plot(x_np, psi_solution ,"--", label="Answer")
    plt.plot(x_np, psi_solution_minus ,"--", label="Answer")
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("Coordinate $x$ [Bohr]")
    plt.ylabel("Wave amplitude")


    plt.savefig(graph_filename)
    plt.show()
    plt.close()


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total Execution Time: {execution_time} seconds")

    difference = np.mean(np.abs(predicted_psi - psi_solution))
    difference_minus = np.mean(np.abs(predicted_psi - psi_solution_minus))
    energy = calc_energy(c)
    # 結果の出力とログ
    print(f"Mean Absolute Difference: {difference}")
    print(f"Energy: {energy}")
    # Ensure all values are serializable
    log_data = {
        'method=trust-constr'
        'N_BASIS': N_BASIS,
        'h': h,
        'epochs' : epochs,
        'difference': float(difference),  # Convert to float
        'difference_minus': float(difference_minus),  # Convert to float
        'energy': float(energy.numpy()) if hasattr(energy, 'numpy') else float(energy),  # Convert TensorFlow tensor to float
        'execution_time': float(execution_time)  # Convert to float
    }

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log data with a unique filename
    log_filename = os.path.join(session_folder, "log.json")

    with open(log_filename, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

    return c

first_state_psi(50,9)

