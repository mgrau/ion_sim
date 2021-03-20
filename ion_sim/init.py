import numpy as np


def thermal(ions, T):
    '''
    Creates an initial state vector of n ions described by a temperature T
    and a gaussian spread in position sigma=[sigma_x, sigma_y, sigma_z]

    Parameters
    ----------
    T : float
        the initial temperature of the ions
    sigma : tuple
        sigma=[sigma_x, sigma_y, sigma_z] is a tuple of 3 std used to describe
        the gaussian probaiblity distribution of initial position
    '''
    N = len(ions.m)
    d = ions.d
    σv = ((ions.u.k * T / ions.m)**(1/2)).to_base_units()
    v0 = σv * np.random.normal(0, 1, (d, N))
    return v0


def gaussian(ions, sigma=(1e-6, 1e-6, 1e-6)):
    N = len(ions.m)
    d = ions.d
    σx = np.broadcast_to(sigma, (N, d)).T
    x0 = σx * np.random.normal(0, 1, (d, N))
    return x0


def string(ions, dx, dim=0, center=True):
    x0 = dx * np.zeros((ions.d, ions.N))
    if center:
        x0[dim, :] = dx * np.linspace(-(ions.N-1)/2, (ions.N-1)/2, ions.N)
    else:
        x0[dim, :] = dx * np.arange(ions.N)
    return x0
