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

# 自動微分で計算したい
def d_potential(x):
    """
    Defines the quadratic potential
    """
    return m**2 * x;

def hubble(x, dx):
    """
    Defines the Hubble parameter in terms of e-folds number
    """
    v = potential(x)
    h2 = (v / 3) / (1 - ( dx**2 / 6 ) )
    h = torch.sqrt(h2)
    return h

def srp_epsilon(x):
    """
    Defines the first-order slow-roll parameter for the potential.
    The prefix "srp" represents "slow-roll parameter".

    x: field value
    """
    v = potential(x)
    dv = d_potential(x)
    eps = 0.5 * (dv / v)**2
    return eps


def klein_gordon_equation(x, dx, dx2):
    dv = d_potential(x)
    h = hubble(x, dx)
    e = srp_epsilon(x)
    eq = dx2 + (3 - e)*dx + dv/h**2
    return eq


def harmonic_oscillator_equation(x, dx, dx2):
    return dx2 + mu*dx + k*x
    
