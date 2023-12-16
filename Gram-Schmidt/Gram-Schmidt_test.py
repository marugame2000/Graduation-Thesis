
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.integrate import quad
from numpy import pi, sin, sqrt,cos

def inner_product(f, g, a, b):

    integrand = lambda x: f(x) * g(x)
    return quad(integrand, a, b)[0]


def gram_schmidt_orthogonalize(f, g, a, b):

    inner_fg = inner_product(f, g, a, b)

    norm_f_squared = inner_product(f, f, a, b)

    orthogonal_g = lambda x: g(x) - (inner_fg / norm_f_squared) * f(x)

    return orthogonal_g


f = lambda x: np.sqrt(2) * sin(2 * pi * x) 
g = lambda x: np.sqrt(2) * sin(2 *pi * x)  

a = 0 
b = 1  

orthogonal_g = gram_schmidt_orthogonalize(f, g, a, b)

x_values = np.linspace(a, b, 100)  
plt.xlim(0,1)
plt.plot(x_values,orthogonal_g(x_values),label="fitted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel(r"$\psi(x)$")
plt.show()
