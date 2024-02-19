import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import factorial

def solution(k,n):
    
    m = 1  
    hbar = 1 
    omega = k**(1/2)

    
    def psi_n(x, n):
      
        Hn = hermite(n)
        
        normalization = (2**n * factorial(n) * np.sqrt(np.pi))**(-0.5) * (m * omega / hbar)**0.25
      
        return normalization * np.exp(-m * omega * x**2 / (2 * hbar)) * Hn(np.sqrt(m * omega / hbar) * x)

    x = np.linspace(-5, 5, 3000)

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
