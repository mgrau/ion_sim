import numpy as np
import matplotlib.pyplot as plt
from ion_sim import IonSim, plot, init
from pint import UnitRegistry
from math import pi as π

u = UnitRegistry()

m = u('40 amu')
ωx = 2*π * u('1 MHz')
ωy = 2*π * u('10 MHz')
ωz = 2*π * u('10 MHz')
ξ = ((u.e**2 / (4*π*u.ε_0 * m * ωx**2))**(1/3)).to_base_units()


class harmonic(IonSim):
    def U(self, x, y, z, t):
        m = self.m
        Ux = (1/2) * m * ωx**2 * x**2
        Uy = (1/2) * m * ωy**2 * y**2
        Uz = (1/2) * m * ωz**2 * z**2
        return (Ux + Uy + Uz).to_base_units().m


def test_normal_modes_2():
    ions = harmonic(u)
    print(f'{"":<6}{"Eigenvalue":6}  {"Eigenvector"}')

    ions.m = m * np.ones(2)
    ions.x0 = init.string(ions, dx=ξ)
    f, b = ions.normal_modes()
    μ = np.around(((2*π*f/ωx)**2).to('dimensionless').m, 4)

    assert μ[0] == 1
    assert μ[1] == 3


def test_normal_modes_3():
    ions = harmonic(u)
    print(f'{"":<6}{"Eigenvalue":6}  {"Eigenvector"}')

    ions.m = m * np.ones(3)
    ions.x0 = init.string(ions, dx=ξ)
    f, b = ions.normal_modes()
    μ = np.around(((2*π*f/ωx)**2).to('dimensionless').m, 4)

    assert μ[0] == 1
    assert μ[1] == 3
    assert μ[2] == 5.8


def test_normal_modes_4():
    ions = harmonic(u)
    print(f'{"":<6}{"Eigenvalue":6}  {"Eigenvector"}')

    ions.m = m * np.ones(4)
    ions.x0 = init.string(ions, dx=ξ)
    f, b = ions.normal_modes()
    μ = np.around(((2*π*f/ωx)**2).to('dimensionless').m, 4)

    assert μ[0] == 1
    assert μ[1] == 3
    assert np.round(μ[2], 2) == 5.81
    assert np.round(μ[3], 3) == 9.308
