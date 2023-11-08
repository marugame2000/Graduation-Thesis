import numpy as np
from numpy import pi,sin,sqrt
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras import backend as K
from keras import layers, models
from keras.layers import Dense,Activation,LeakyReLU
import matplotlib.pyplot as plt

N=10000
x=np.linspace(0,1,N)
dummy=np.sqrt(2)*np.sin(np.pi*x)

def psi(y):
    y_rev = K.reverse(y,0)
    y_symmetrized = y + y_rev -y[0] -y[-1]
    return y_symmetrized

def dpsi(y):
    y_shifted_f = tf.roll(y,shift=-1, axis=0)
    y_shifted_b = tf.roll(y,shift=+1,axis=0)
    dy=(y_shifted_f-y_shifted_b)/2
    return dy

def variationalE(y_true,y_pred):
    wave=psi(y_pred)
    wave_nom=K.l2_normalize(wave,0)
    dwave = dpsi(wave_nom)
    return N**2 * K.sum(K.square(dwave))/pi**2

model=Sequential()
model.add(Dense(2,input_dim=1,activation="sigmoid"))
model.add(Dense(1,activation="linear"))
model.compile(loss=variationalE,optimizer="Adam")
model.summary()

results = model.fit(x,dummy,epochs=300,steps_per_epoch=1,verbose=1,shuffle=False)

pred=model.predict(x)
func=psi(pred)
func=np.abs(func)/np.sqrt(np.sum(func**2)/N)
plt.xlim(0,1)
plt.plot(x,func,label="fitted")
plt.plot(x,dummy, "--",label="answer")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")
plt.show()