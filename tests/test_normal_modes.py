import pytest
import numpy as np
from ion_sim import IonSim, init
from pint import UnitRegistry
from math import pi as π

u = UnitRegistry()

m = u('40 amu')
q = u('e')
ωx = 2*π * u('1 MHz')
ωy = 2*π * u('10 MHz')
ωz = 2*π * u('10 MHz')
l = ((q**2/(4*π*u.ε_0 * m * ωx**2))**(1/3)).to_base_units()


class Example(IonSim):
    def U(self, x, y, z, t):
        m = self.m
        Ux = (1/2) * m * ωx**2 * x**2
        Uy = (1/2) * m * ωy**2 * y**2
        Uz = (1/2) * m * ωz**2 * z**2
        return (Ux + Uy + Uz).to_base_units().m


def test_normal_modes_2():
    ions = Example(u)
    print(f'{"":<6}{"Eigenvalue":6}  {"Eigenvector"}')

    ions.m = m * np.ones(2)
    ions.x0 = init.string(ions, dx=l)
    f, b = ions.normal_modes()
    μ = np.around(((2*π*f/ωx)**2).to('dimensionless').m, 4)

    assert μ[0] == 1
    assert μ[1] == 3

    assert np.abs(np.inner(b[0, :, 0], np.array(
        [0.7071, 0.7071]))) == pytest.approx(1, 1e-4)
    assert np.abs(np.inner(b[0, :, 1], np.array(
        [-0.7071, 0.7071]))) == pytest.approx(1, 1e-4)


def test_normal_modes_3():
    ions = Example(u)
    print(f'{"":<6}{"Eigenvalue":6}  {"Eigenvector"}')

    ions.m = m * np.ones(3)
    ions.x0 = init.string(ions, dx=l)
    f, b = ions.normal_modes()
    μ = np.around(((2*π*f/ωx)**2).to('dimensionless').m, 4)

    assert μ[0] == 1
    assert μ[1] == 3
    assert μ[2] == 5.8

    assert np.abs(np.inner(b[0, :, 0], np.array(
        [0.5774, 0.5774, 0.5774]))) == pytest.approx(1, 1e-4)
    assert np.abs(np.inner(b[0, :, 1], np.array(
        [-0.7071, 0, 0.7071]))) == pytest.approx(1, 1e-4)
    assert np.abs(np.inner(b[0, :, 2], np.array(
        [0.4082, -0.8165, 0.4082]))) == pytest.approx(1, 1e-4)


def test_normal_modes_4():
    ions = Example(u)
    print(f'{"":<6}{"Eigenvalue":6}  {"Eigenvector"}')

    ions.m = m * np.ones(4)
    ions.x0 = init.string(ions, dx=l)
    f, b = ions.normal_modes()
    μ = np.around(((2*π*f/ωx)**2).to('dimensionless').m, 4)

    assert μ[0] == 1
    assert μ[1] == 3
    assert μ[2] == pytest.approx(5.81, 0.01)
    assert μ[3] == pytest.approx(9.308, 0.001)

    assert np.abs(np.inner(b[0, :, 0], np.array(
        [0.5, 0.5, 0.5, 0.5]))) == pytest.approx(1, 1e-4)
    assert np.abs(np.inner(b[0, :, 1], np.array(
        [-0.6742, -0.2132, 0.2132, 0.6742]))) == pytest.approx(1, 1e-4)
    assert np.abs(np.inner(b[0, :, 2], np.array(
        [0.5, -0.5, -0.5, 0.5]))) == pytest.approx(1, 1e-4)
    assert np.abs(np.inner(b[0, :, 3], np.array(
        [-0.2132, 0.6742, -0.6742, 0.2132]))) == pytest.approx(1, 1e-4)
