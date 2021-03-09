import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
import autograd
import pint

from math import pi as π


class IonSim:
    __u: pint.UnitRegistry = None
    __d: int
    __m: float = None
    __m_unit: pint.Unit = None
    __Γ: float = None
    __Γ_unit: pint.Unit = None

    __x0: np.array = None
    __x_unit: pint.Unit = None
    __v0: np.array = None
    __v_unit: pint.Unit = None
    __t: np.array = None
    __t_unit: pint.Unit = None
    __x: np.array = None
    __v: np.array = None

    def __init__(self, unit_registry: pint.UnitRegistry = None, d=3):
        self.__d = d
        self.u = unit_registry
        self.m = self.u('1 amu')

        # self.gamma = np.array([0, 0, 0]) * 1e-6

    def __repr__(self):
        s = '''
        Ion Simulation
        Number of Ions: %d
        ''' % (self.x0.shape[1])
        return s

    @property
    def d(self):
        return self.__d

    @property
    def u(self):
        return self.__u

    @u.setter
    def u(self, unit_registry: pint.UnitRegistry):
        self.__u = unit_registry

        # recalculate constants in base units whenever unit registry is set
        self.__kq2 = (self.u.e**2/(4*π*self.u.ε_0)).to_base_units().m

    @property
    def m(self):
        return (self.__m * self.u('kg').to_base_units().units).to(self.__m_unit)

    @m.setter
    def m(self, m: pint.Quantity):
        self.__m = m.to_base_units().m
        self.__m_unit = m.units
        self.Γ = self.u('0 Hz')
        self.x0 = np.zeros((self.d, self.N)) * self.u('m')
        self.v0 = np.zeros((self.d, self.N)) * self.u('m/s')

    @property
    def N(self):
        if self.m is None:
            return 0
        try:
            return len(self.m)
        except TypeError:
            return 1

    @property
    def gamma(self):
        return (self.__Γ * self.u('s^-1').to_base_units()).to(self.__Γ_unit)

    @gamma.setter
    def gamma(self, gamma):
        if hasattr(gamma, 'shape') and (gamma.shape == (self.d, self.N)):
            Γ = gamma
        else:
            try:
                if len(gamma) == self.N:
                    Γ = np.broadcast_to(gamma, (self.d, self.N))
                elif len(gamma) == self.d:
                    Γ = np.broadcast_to(gamma, (self.N, self.d)).T
            except TypeError:
                Γ = gamma * np.ones((self.d, self.N))
        self.__Γ = Γ.to_base_units().m
        self.__Γ_unit = Γ.units

    @property
    def Γ(self):
        return self.gamma

    @Γ.setter
    def Γ(self, gamma):
        self.gamma = gamma

    @property
    def x0(self):
        return (self.__x0 * self.u('m').to_base_units().units).to(self.__x_unit)

    @x0.setter
    def x0(self, x0: pint.Quantity):
        self.__x0 = x0.to_base_units().m
        self.__x_unit = x0.units

    @property
    def v0(self):
        return (self.__v0 * self.u('m/s').to_base_units().units).to(self.__v_unit)

    @v0.setter
    def v0(self, v0: pint.Quantity):
        self.__v0 = v0.to_base_units().m
        self.__v_unit = v0.units

    @property
    def t(self):
        return (self.__t * self.u('s').to_base_units().units).to(self.__t_unit)

    @t.setter
    def t(self, t: pint.Quantity):
        self.__t = t.to_base_units().m
        self.__t_unit = t.units

    @property
    def x(self):
        return (self.__x * self.u('m').to_base_units().units).to(self.__x_unit)

    @property
    def v(self):
        return (self.__v * self.u('m/s').to_base_units().units).to(self.__v_unit)

    def U(self, x, y, z, t=0):
        '''
        By default the potential is 0. This function can be overloaded with a custom
        potential.
        '''
        return 0

    def U_Coulomb(self, x, y, z):
        r2 = (x[:, None]-x)**2 + (y[:, None]-y)**2 + (z[:, None]-z)**2
        # pylint: disable=no-member
        return autograd.numpy.sum(self.__kq2 / r2[autograd.numpy.triu_indices(x.size, 1)]**(1/2))

    def U_total(self, x, y, z, t=0):
        # pylint: disable=no-member
        return autograd.numpy.sum(self.U(x, y, z, t)) + self.U_Coulomb(x, y, z)

    def F_U(self, x, y, z, t=0):
        '''
        Calculates the force at point x, y, z using the technique of automatic
        differentiation of potential U(t, x, y, z).
        '''
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return -1 * np.stack(autograd.elementwise_grad(self.U, argnum=(0, 1, 2))(x, y, z, t))

    def F_Coulomb_autograd(self, x, y, z):
        '''
        calculates repulsive Coulomb force using automatic differentiation
        assumes all particles have unit charge with the same sign.
        This is about 2x slower than F_Coulomb, where the force has been explicitly
        vectorized with numpy
        '''
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return -1 * np.stack(autograd.elementwise_grad(self.U_Coulomb, argnum=(0, 1, 2))(x, y, z))

    def F_damp(self, v):
        '''
        A damping force that is -Gamma*v
        '''
        return -self.__m * self.__Γ * v

    def F_Coulomb(self, x, y, z):
        '''
        repulsive Coulomb force, assumes all particles have unit charge
        with the same sign
        '''
        n = x.size
        r = np.zeros((self.d, n, n))
        for i, w in enumerate([x, y, z]):
            r[i, :, :] = w[:, None] - w
        r2 = np.sum(r**2, axis=0)
        np.fill_diagonal(r2, np.inf)
        r3 = r2**(-3 / 2)
        F = np.zeros((self.d, n))

        for i in range(self.d):
            F[i, :] = np.sum(r[i, :, :] * r3, axis=1)
        return self.__kq2*F

    def F_total(self, x, y, z, v=0, t=0):
        '''
        Total force
        '''
        F = self.F_U(x, y, z, t)
        F += self.F_damp(v)
        F += self.F_Coulomb(x, y, z)
        return F

    def equilibrium_position(self, x0=None, y0=None, z0=None, t=0, tol=1):
        '''
        Finds the equilibrium positions by calculating the local potential minimum
        '''
        if (x0 is None) or (y0 is None) or (z0 is None):
            x0 = self.x0[0, :]
            y0 = self.x0[1, :]
            z0 = self.x0[2, :]
        guess = pint.Quantity(np.concatenate([x0, y0, z0]).ravel())
        # using the negative of the force as the gradient makes the
        # minmization much more efficient

        # we need to use the electric constant as the tolerance to make sure the
        # minimization algorithm doesn't confuse a small numerical potential as
        # expressed in SI units as adequate and simply exit immediately
        result = scipy.optimize.minimize(
            fun=lambda x: self.U_total(*x.reshape(self.d, -1), t=t),
            jac=lambda x: -self.F_total(*x.reshape(self.d, -1)).ravel(),
            x0=guess.to_base_units().m,
            tol=tol*self.__kq2)
        return (result.x.reshape((self.d, -1)) * self.u('m').to_base_units().units).to(guess.units)

    def Hessian(self, x, y, z, t=0):
        '''
        Calculate the Hessian using automatic differentiation of the potential
        '''

        def U(x): return self.U_total(*x.reshape(self.d, -1), t=t)
        # pylint: disable=no-value-for-parameter
        def F(x): return (autograd.jacobian(U)(
            x).reshape(self.d, -1) / self.__m).ravel()
        # pylint: disable=no-value-for-parameter
        return autograd.jacobian(F)(np.concatenate([x, y, z]).ravel())

    def normal_modes(self, x=None, y=None, z=None):
        '''
        calculates normal mode frequencies (f) and eigenvectors (v) about some position
        given by (x0, y0, z0), nominally the equilibrium position
        '''
        if (x is None) or (y is None) or (z is None):
            x0 = self.equilibrium_position()
        else:
            x0 = np.vstack([x, y, z])
        H = self.Hessian(*x0.to_base_units().m)
        λ, v = np.linalg.eig(H)

        sort = np.abs(λ).argsort()
        v = np.reshape(v[:, sort], self.x0.shape + (-1,))
        ω2 = λ[sort]
        f = ((np.sign(ω2)*abs(ω2)**(1/2)/(2*π)) *
             self.u('Hz').to_base_units().units).to('Hz')
        return f, v

    def f(self, t, x):
        '''
        calculates the derivative of the position and velocity
        '''
        x = np.reshape(x, (2*self.d, -1))
        xp = np.zeros(x.shape)
        xp[:self.d, :] = x[self.d:, :]
        xp[self.d:, :] = self.F_total(
            *x[:self.d, :], x[self.d:, :], t) / self.__m
        return xp.ravel()

    def run(self, t, method='RK45'):
        y0 = np.vstack((self.__x0, self.__v0)).ravel()
        self.t = t
        sol = scipy.integrate.solve_ivp(self.f, self.__t[[0, -1]],
                                        y0, t_eval=self.__t, method=method)
        y = np.reshape(sol.y, (2*self.d, -1, len(sol.t)))
        self.__t = sol.t
        self.__x = y[:3, :, :]
        self.__v = y[3:, :, :]
        return sol.y
