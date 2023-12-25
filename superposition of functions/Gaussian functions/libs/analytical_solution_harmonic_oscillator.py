import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import factorial

def solution(k,n):
    # Constants
    m = 1  # mass
    hbar = 1  # Reduced Planck constant
    #omega = 1  # Angular frequency for a simple harmonic oscillator
    omega = k**(1/2)

    # Defining the wave functions for the harmonic oscillator
    def psi_n(x, n):
        # Hermite polynomial
        Hn = hermite(n)
        # Normalization factor
        normalization = (2**n * factorial(n) * np.sqrt(np.pi))**(-0.5) * (m * omega / hbar)**0.25
        # The wave function
        return normalization * np.exp(-m * omega * x**2 / (2 * hbar)) * Hn(np.sqrt(m * omega / hbar) * x)

    # Range of x for plotting
    x = np.linspace(-5, 5, 5000)

    # Wave functions for the ground state (n=0), first excited state (n=1), and second excited state (n=2)
    psi_0 = psi_n(x, 0)
    psi_1 = psi_n(x, 1)
    psi_2 = psi_n(x, 2)

    return(psi_n(x,n))

    # Plotting the wave functions
    #plt.figure(figsize=(10, 6))
    #plt.plot(x, psi_0, label='Ground State (n=0)')
    #plt.plot(x, psi_1, label='First Excited State (n=1)')
    #plt.plot(x, psi_2, label='Second Excited State (n=2)')
    #plt.xlabel('Position (x)')
    #plt.ylabel('Wave Function Ψ')
    #plt.title('Wave Functions of a Quantum Harmonic Oscillator')
    #plt.legend()
    #plt.grid()
    #plt.show()
