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


def test_equilibrium_position():
    ions = Example(u)

    ions.m = m * np.ones(2)
    ions.x0 = init.string(ions, dx=l)
    positions = (ions.equilibrium_position() / l).to('dimensionless').m[0, :]
    np.testing.assert_allclose(
        positions, np.array([-0.62996, 0.62996]), atol=1e-5)

    ions.m = m * np.ones(3)
    ions.x0 = init.string(ions, dx=l)
    positions = (ions.equilibrium_position() / l).to('dimensionless').m[0, :]
    np.testing.assert_allclose(
        positions, np.array([-1.0772, 0, 1.0772]), atol=1e-4)

    ions.m = m * np.ones(4)
    ions.x0 = init.string(ions, dx=l)
    positions = (ions.equilibrium_position() / l).to('dimensionless').m[0, :]
    np.testing.assert_allclose(
        positions, np.array([-1.4368, -0.45438, 0.45438, 1.4368]), atol=1e-4)

    ions.m = m * np.ones(5)
    ions.x0 = init.string(ions, dx=l)
    positions = (ions.equilibrium_position() / l).to('dimensionless').m[0, :]
    np.testing.assert_allclose(
        positions, np.array([-1.7429, -0.8221, 0, 0.8221, 1.7429]), atol=1e-4)
