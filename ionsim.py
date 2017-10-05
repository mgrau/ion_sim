import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.constants as _c
import numpy.linalg as LA

import autograd

from scipy import integrate, optimize
from matplotlib import animation


class IonSim:
    def __init__(self):
        self.m = 40 * _c.atomic_mass
        self.kq2 = _c.elementary_charge**2 / (4 * _c.pi * _c.epsilon_0)
        self.gamma = np.array([0, 0, 0]) * 1e-6
        self.x0 = []

    def __repr__(self):
        s = '''
        Ion Simulation
        Number of Ions: %d
        ''' % (self.x0.shape[1])
        return s

    def init_thermal(self, n, T, sigma=(1e-6, 1e-6, 1e-6)):
        '''
        Creates an initial state vector of n ions described by a temperature T
        and a gaussian spread in position sigma=[sigma_x, sigma_y, sigma_z]
        
        Parameters
        ----------
        n : int
            The number of ions to initialized.
        T : float
            the initial temperature of the ions
        sigma : tuple
            sigma=[sigma_x, sigma_y, sigma_z] is a tuple of 3 std used to describe
            the gaussian probaiblity distribution of initial position
        '''
        self.x0 = np.zeros((6, n))
        self.x0[0, :] = np.random.normal(0, sigma[0], (1, n))
        self.x0[1, :] = np.random.normal(0, sigma[1], (1, n))
        self.x0[2, :] = np.random.normal(0, sigma[2], (1, n))
        sigma_v = np.sqrt(_c.k * T / self.m)
        self.x0[3:6, :] = np.random.normal(0, sigma_v, (3, n))

    def U(self, t, x, y, z):
        '''
        By default the potential is 0. This function can be overloaded with a custom
        potential.
        '''
        return 0

    def U_Coulomb(self, x, y, z):
        r2 = (x[:,None]-x)**2 + (y[:,None]-y)**2 + (z[:,None]-z)**2
        r = np.sqrt(r2[np.triu_indices(np.size(x),1)])
        return np.sum(self.kq2/r)

    def U_total(self, t, x, y, z):
        return np.sum(self.U(t, x, y, z)) + self.U_Coulomb(x, y, z)

    def F_U(self, t, x, y, z):
        '''
        Calculates the force at point x, y, z using the technique of automatic
        differentiation of potential U(t, x, y, z).
        '''
        return -np.stack(autograd.multigrad(self.U, [1,2,3])(0, x, y, z))
        
    def F_Coulomb_autograd(self, x, y, z):
        '''
        calculates repulsive Coulomb force using automatic differentiation
        assumes all particles have unit charge with the same sign
        '''
        return -np.stack(autograd.multigrad(self.U_Coulomb, [0,1,2])(x, y, z))
        
    def F_damp(self, v):
        '''
        A damping force that is -Gamma*|v|^2
        '''
        return -np.expand_dims(self.gamma, axis=1).repeat(self.x0.shape[1], axis=1) * v

    def F_Coulomb(self, x, y, z):
        '''
        repulsive Coulomb force, assumes all particles have unit charge
        with the same sign
        '''
        n = x.size
        r = np.zeros((3, n, n))
        for i, w in enumerate([x, y, z]):
            r[i, :, :] = w[:, None] - w
        r2 = np.sum(r**2, axis=0)
        np.fill_diagonal(r2, np.inf)
        r3 = r2**(-3 / 2)
        F = np.zeros((3, n))
        for i in range(3):
            F[i, :] = self.kq2 * np.sum(r[i, :, :] * r3, axis=1)
        return F

    def F_total(self, t, x, y, z, v=0):
        '''
        Total force
        '''
        F = self.F_U(t, x, y, z)
        F += self.F_damp(v)
        F += self.F_Coulomb(x, y, z)
        return F

    def Hessian(self, t, x, y, z):
        '''
        Calculate the Hessian using automatic differentiation of the potential
        '''
        U = lambda x: self.U_total(t, *np.reshape(x, (3, -1)))
        return autograd.hessian(U)(np.concatenate([x, y, z]).ravel())

    def equilibrium_position(self, x0, y0, z0):
        '''
        Finds the equilibrium positions by calculating the local potential minimum
        '''
        U = lambda x: self.U_total(0, *np.reshape(x, (3, -1)))
        # using the negative of the force as the gradient makes the 
        # minmization much more efficient
        jac = lambda x: -self.F_total(0, *np.reshape(x, (3, -1))).ravel()
        guess = np.concatenate([x0, y0, z0]).ravel()
        # we need to use the electric constant as the tolerance to make sure the 
        # minimization algorithm doesn't confuse a small numerical potential as 
        # expressed in SI units as adequate and simply exit immediately
        result = optimize.minimize(U, guess, jac=jac, tol=self.kq2)
        return np.reshape(result.x, (3, -1))

    def normal_modes(self, x0, y0, z0):
        H = self.Hessian(0, x0, y0, z0)
        w2, v = np.linalg.eig(H)
        
        v = v[:,np.abs(w2).argsort()]
        f = w2[np.abs(w2).argsort()]/self.m
        f = np.sign(f)*np.sqrt(abs(f))/(2*_c.pi)
        return f, v

    def penning_modes(self, x0, y0, z0, B):
        omega_c = _c.elementary_charge*B/self.m
        n = x0.size
        I = np.eye(3*n)
        T = np.kron(np.cross(np.eye(3), omega_c), np.eye(n))
        K = self.Hessian(0, x0, y0, z0)/self.m
        A = np.bmat([[1j*T, I], [I, 0*T]])
        B = np.bmat([[K, 0*K] ,[0*K, I]])
        w, v = LA.eig(np.dot(LA.inv(A), B))
        v = v[0:(3*n), np.real(w).argsort()]
        v = v/LA.norm(v, axis=0)
        f = np.real(w[np.real(w).argsort()])/(2*_c.pi)
        return f, v.A

    def f(self, t, x):
        '''
        calculates the derivative of the position and velocity
        '''
        x = np.reshape(x, (6, -1))
        xp = np.zeros(x.shape)
        xp[0:3, :] = x[3:6, :]
        xp[3:6, :] = self.F_total(t, *x[0:3, :], x[3:6, :]) / self.m
        return xp.ravel()

    def run(self, t):
        '''
        Performs Dormand & Prince adaptive 4th order explicit numerical integration
        '''
        r = integrate.ode(self.f)
        r.set_integrator('dopri5')

        try:
            self.x0 = self.x[-1, :, :]
            print("rerunning")
        except:
            pass
        
        x = np.zeros((t.size,) + self.x0.shape)
        r.t = t[0]
        r.set_initial_value(self.x0.ravel(), 0)
        x[0, :, :] = self.x0
        for i, t_i in enumerate(t[1:]):
            x[i + 1:, :] = np.reshape(r.integrate(t_i), (6, -1))
        
        try:
            self.t = np.concatenate((self.t, self.t[-1] + t))
            self.x = np.concatenate((self.x, x))
        except:
            self.t = t
            self.x = x


def plot(sim, i=[], dim=0):
    '''
    Plot ion position or velocity as a function of time.
    Makes a matplotlib call and creates a figure.
    
    Parameters
    ----------
    i : array, optional
        A list of indices of ions to plot. An empty list may be used to plot
        all ions.
    dim : int, optional
        The dimension of the ion state vector to plot. The default is 0,
        corresponding to the 'x' position.
    '''
    if not i:
        plt.plot(sim.t, sim.x[:, dim, :]);
    else:
        plt.plot(sim.t, sim.x[:, dim, i]);
        
    plt.xlabel('Time')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

def animate(sim):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    rmax = np.max(sim.x[:, 0:2, :]) * 1.1
    ax = plt.axes(xlim=(-rmax, rmax), ylim=(-rmax, rmax))
    points, = ax.plot([], [], 'b.', markersize=20)
    lines = [ax.plot([], [], 'k-')[0] for _ in range(sim.x.shape[-1])]
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    # initialization function: plot the background of each frame
    def init():
        points.set_data([], [])
        for line in lines:
            line.set_data([], [])
        return lines

    # animation function.  This is called sequentially
    def draw_frame(frame_i):
        points.set_data(sim.x[frame_i, 0, :], sim.x[frame_i, 1, :])
        for i, line in enumerate(lines):
            line.set_data(sim.x[(frame_i - 5):frame_i, 0, i],
                          sim.x[(frame_i - 5):frame_i, 1, i])
        return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, draw_frame, init_func=init,
                                   frames=sim.t.size,
                                   interval=(1000 / 60),
                                   blit=True,
                                   repeat=False)
    plt.close(fig)
    return anim

def plot_normal_modes(sim, x, i=0, scale=3, surface=False):
    freq, v = sim.normal_modes(*x)

    freq_i = freq[i]
    v_i = np.reshape(np.real(v[:,i]), (3, -1))
    
    drum = np.sum(np.abs(v_i[2,:])) > 0.1
    if drum:
        rmax = np.max(np.abs(x)) * 1.2
    else:
        rmax = np.max(np.abs(x)) * 3
    if not surface:
        drum = False
        

    plt.figure(figsize=(10,8))

    n=5
    plt.subplot2grid((n, 1), (0, 0), rowspan=n-1)
    if drum:
        plt.tripcolor(x[0,:], x[1,:], v_i[2,:], cmap='coolwarm', shading='gouraud')
    else:
        plt.plot(x[0,:], x[1,:], 'o', markersize=20)
        plt.quiver(x[0,:], x[1,:], (v_i+x)[0,:], (v_i+x)[1,:], scale=scale)
    plt.xlim(-rmax,rmax)
    plt.ylim(-rmax,rmax)
    plt.gca().set_aspect('equal', 'datalim')

    plt.subplot2grid((n, 1), (n-1, 0), rowspan=1)
    plt.stem(freq, np.ones(freq.shape), linefmt='0.5', markerfmt='0.75', basefmt='0.75')
    _, stemlines, _ = plt.stem([freq[i]], [1], linewidth=5)
    plt.setp(stemlines, 'linewidth', 3)
    plt.ylim((0.25,0.75))
    plt.yticks([])
    plt.xlabel('Frequency')

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.show()
