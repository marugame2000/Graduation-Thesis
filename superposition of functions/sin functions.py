import numpy as np
from numpy import pi, sin,cos , sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU,BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sympy import symbols, pi, cos, integrate
from scipy.integrate import quad

N_BASIS = 12
X_MIN = 0.0
X_MAX = 1.0

def psi(n, x):
    return tf.math.sin((n + 1) * np.pi * (x - X_MIN) / (X_MAX - X_MIN))

def deriv_phi(n, x):
    return (n + 1) * pi * cos((n + 1) * pi * (x - X_MIN) / (X_MAX - X_MIN)) / (X_MAX - X_MIN)

def calc_h(n, m):
    integrand = lambda x: ((deriv_phi(n, x) * deriv_phi(m, x))/2)
    integral, _ = quad(integrand, X_MIN, X_MAX)
    return integral

def calc_s(n, m):
    integrand = lambda x: psi(n, x) * psi(m, x)
    integral, _ = quad(integrand, X_MIN, X_MAX)
    return integral

S = np.zeros([N_BASIS, N_BASIS], dtype = np.float32 )
H = np.zeros([N_BASIS, N_BASIS], dtype = np.float32 )
for i in range(N_BASIS):
    for j in range(N_BASIS):
        S[i,j] = calc_s(i,j)
        H[i,j] = calc_h(i,j)

def calc_energy(c):
    Hc = tf.tensordot(H,c,axes=(1,0))
    Sc = tf.tensordot(S,c,axes=(1,0))
    cHc = tf.tensordot(c,Hc,axes=(0,0))
    cSc = tf.tensordot(c,Sc,axes=(0,0))
    return cHc /cSc

def orthogonality_penalty(c, ground_state_psi):
    
    ground_state_psi = tf.cast(ground_state_psi, tf.float32)

    predicted_psi = tf.zeros_like(ground_state_psi)
    
    for i in range(N_BASIS):
        predicted_psi += c[i] * psi(i, x)

    overlap = tf.tensordot(predicted_psi, ground_state_psi, axes=1)
    
    return overlap**2

x = tf.linspace(X_MIN, X_MAX, 1000) 


import numpy as np  

def model_loss(_, c):
    energy_loss = calc_energy(c)
    ground_state = tf.math.sin(np.pi * x) 
    first_excited = (-1) * tf.math.sin(2 * np.pi * x)
    penalty = orthogonality_penalty(c, ground_state)
    penalty2 = orthogonality_penalty(c, first_excited)
    return energy_loss + 10 * (penalty + penalty2)


model = Sequential([
    Dense(512, input_dim=1, activation=LeakyReLU(alpha=0.3)),
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

model.compile(loss = model_loss , optimizer = 'adam')

index = np.arange(N_BASIS)
dummy = np.zeros_like(index)
model.fit(
    index, 
    dummy, 
    epochs=3000, 
    steps_per_epoch=1, 
    verbose=1, 
    shuffle=False, 
)

index = np.arange(N_BASIS)
c = model.predict(index)[:,0]
print(c)

norm = np.dot(c,np.dot(S,c))
energy = np.dot(c, np.dot(H,c))/norm

print("Calculated total energy: %f" % energy)
pi_tf = tf.constant(np.pi, dtype=tf.float32)
ground_state_tf = tf.sqrt(2.0) * tf.math.sin(3 * pi_tf * x)
first_excited_tf = (-1) * tf.sqrt(2.0) * tf.math.sin(3 * pi_tf * x)


x_np = x.numpy()
ground_state_np = ground_state_tf.numpy()
first_excited_np = first_excited_tf.numpy()


y_tf = tf.zeros_like(x)
for i in range(N_BASIS):
    y_tf += c[i] * psi(i, x)
y_tf = y_tf * (1.0 / tf.sqrt(norm))


y_np = y_tf.numpy()


plt.plot(x_np, y_np,label="fitted")
plt.plot(x_np, ground_state_np, "--", label="Second Excited State_minus")
plt.plot(x_np, first_excited_np, "--", label="Second Excited State")
plt.legend()
plt.xlim([X_MIN, X_MAX])
plt.xlabel("Coordinate $x$ [Bohr]")
#plt.ylim([-1.8, +1.8])
plt.ylabel("Wave amplitude")
plt.show()
