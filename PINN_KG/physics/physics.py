import numpy as np
import torch

m = 5.1e-5

d, w0 = 2, 20
mu, k = 2*d, w0**2

def oscillator(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y


def potential(x):
    """
    Defines the quadratic potential
    """
    return 0.5 * m**2 * x**2;

def hubble(x, dx):
    """
    Defines the Hubble parameter in terms of e-folds number
    """
    v = potential(x)
    h2 = (v / 3) / (1 - ( dx**2 / 6 ) )
    h = torch.sqrt(h2)
    return h

def klein_gordon_equation(x, dx, dx2):
    return dx2 + mu*dx + k*x
