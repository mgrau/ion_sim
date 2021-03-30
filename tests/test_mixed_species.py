import pytest
import numpy as np
from ion_sim import IonSim, init
from pint import UnitRegistry
from math import pi as π

u = UnitRegistry()

ωx = np.array([12.26, 4.82]) * 2*π * u('MHz')
ωy = np.array([11.19, 3.72]) * 2*π * u('MHz')
ωz = np.array([2.69, 1.65]) * 2*π * u('MHz')


class MixedExample(IonSim):
    def U(self, x, y, z, t):
        m = self.m
        Ux = (1/2) * m * (ωx)**2 * x**2
        Uy = (1/2) * m * (ωy)**2 * y**2
        Uz = (1/2) * m * (ωz)**2 * z**2
        return (Ux + Uy + Uz).to_base_units().m


class MixedExample_Field(IonSim):
    E = u('200 V/m')

    def U(self, x, y, z, t):
        m = self.m
        Ux = (1/2) * m * (ωx)**2 * x**2
        Uy = (1/2) * m * (ωy)**2 * y**2 + u('e/m')*self.E * y
        Uz = (1/2) * m * (ωz)**2 * z**2
        return (Ux + Uy + Uz).to_base_units().m


def test_mixed_species():
    ions = MixedExample(u)

    ions.m = np.array([9.012, 23.985]) * u('1 amu')
    ions.x0 = init.string(ions, dim=2, dx=u('10 um'))
    ν, e = ions.normal_modes()

    freq = ν.to('MHz').m
    assert freq[0] == pytest.approx(1.90, 0.01)
    assert freq[1] == pytest.approx(3.53, 0.01)
    assert freq[2] == pytest.approx(4.04, 0.01)
    assert freq[3] == pytest.approx(4.68, 0.01)
    assert freq[4] == pytest.approx(11.03, 0.01)
    assert freq[5] == pytest.approx(12.11, 0.01)

    def norm(vec):
        return vec / np.linalg.norm(vec)
    μ = np.sqrt(ions.m.m)

    assert np.abs(np.sum(norm(
        e[:, :, 0] * μ) * np.array([[0, 0], [0, 0], [0.379, 0.926]]))) == pytest.approx(1, 1e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 1] * μ) * np.array([[0, 0], [0.020, -1],  [0, 0]]))) == pytest.approx(1, 1e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 2] * μ) * np.array([[0.0, 0], [0, 0], [-0.926, 0.378]]))) == pytest.approx(1, 1e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 3] * μ) * np.array([[0.18, -1], [0.0, 0], [0, 0]]))) == pytest.approx(1, 5e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 4] * μ) * np.array([[0.0, 0], [1, 0.020], [0, 0]]))) == pytest.approx(1, 5e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 5] * μ) * np.array([[1, 0.18], [0, 0], [0, 0]]))) == pytest.approx(1, 5e-3)


def test_mixed_species_field():
    ions = MixedExample_Field(u)

    ions.m = np.array([9.012, 23.985]) * u('1 amu')
    ions.x0 = init.string(ions, dim=2, dx=u('10 um'))
    ν, e = ions.normal_modes()

    freq = ν.to('MHz').m
    assert freq[0] == pytest.approx(1.89, 0.01)
    assert freq[1] == pytest.approx(3.42, 0.01)
    assert freq[2] == pytest.approx(4.04, 0.01)
    assert freq[3] == pytest.approx(4.67, 0.01)
    assert freq[4] == pytest.approx(11.06, 0.01)
    assert freq[5] == pytest.approx(12.11, 0.01)

    def norm(vec):
        return vec / np.linalg.norm(vec)
    μ = np.sqrt(ions.m.m)

    assert np.abs(np.sum(norm(
        e[:, :, 0] * μ) * np.array([[0, 0], [-0.005, 0.038], [0.36, 0.932]]))) == pytest.approx(1, 1e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 1] * μ) * np.array([[0, 0], [-0.027, 0.882],  [-0.450, 0.137]]))) == pytest.approx(1, 1e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 2] * μ) * np.array([[0.0, 0], [-0.017, -0.470], [-0.817, 0.334]]))) == pytest.approx(1, 1e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 3] * μ) * np.array([[0.18, -1], [0.0, 0], [0, 0]]))) == pytest.approx(1, 5e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 4] * μ) * np.array([[0.0, 0], [-0.999, -0.016], [0.024, -0.014]]))) == pytest.approx(1, 5e-3)
    assert np.abs(np.sum(norm(
        e[:, :, 5] * μ) * np.array([[1, 0.18], [0, 0], [0, 0]]))) == pytest.approx(1, 5e-3)
