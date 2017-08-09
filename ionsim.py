import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as _c

from scipy import integrate
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
        return np.zeros(x.shape)

    def F_U(self, t, x, y, z):
        '''
        Calculates a force at point x, y, z by taking a numerical
        gradient of U(t,x,y,z). The gradient is calculated by taking
        the second order difference with a step size that is 1000
        times the float precision.
        '''
        _d = 1e3 * np.finfo(float).eps
        Fx = -(self.U(t, x + _d, y, z) - self.U(t, x - _d, y, z)) / (2 * _d)
        Fy = -(self.U(t, x, y + _d, z) - self.U(t, x, y - _d, z)) / (2 * _d)
        Fz = -(self.U(t, x, y, z + _d) - self.U(t, x, y, z - _d)) / (2 * _d)
        return np.array([Fx, Fy, Fz])

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
            r[i, :, :] = np.outer(w, np.ones((1, n))) - \
                np.outer(np.ones((1, n)), w)
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
    
    def min_d(self):
        '''
        calculated the minimum distance between ions
        '''
        x = self.x[-1, 0, :]
        y = self.x[-1, 1, :]
        z = self.x[-1, 2, :]
        n = x.size
        r = np.zeros((3, n, n))
        for i, w in enumerate([x, y, z]):
            r[i, :, :] = np.outer(w, np.ones((1, n))) - \
                np.outer(np.ones((1, n)), w)
        r = np.sqrt(np.sum(r**2, axis=0))
        np.fill_diagonal(r, np.inf)
        return np.min(r)

    def Hessian(self, t, x, y, z):
        '''
        Calculate the Hessian numerically
        '''
        n = len(x)
        H = np.zeros((3, 3 * n, n))
        F = lambda x, y, z: (self.F_U(t, x, y, z) + self.F_Coulomb(x, y, z)).ravel()

        _d = 1e3 * np.finfo(float).eps
        for i in range(n):
            _nd = np.zeros(n)
            _nd[i] = _d
            H[0, :, i] = -(F(x + _nd, y, z) - F(x - _nd, y, z)) / (2 * _d)
            H[1, :, i] = -(F(x, y + _nd, z) - F(x, y - _nd, z)) / (2 * _d)
            H[2, :, i] = -(F(x, y, z + _nd) - F(x, y, z - _nd)) / (2 * _d)

        H /= self.m
        return H.transpose([0, 2, 1]).reshape((3 * n, 3 * n))

    def normal_modes(self):
        t = 0
        x = self.x[-1, 0, :]
        y = self.x[-1, 1, :]
        z = self.x[-1, 2, :]
        H = self.Hessian(t, x, y, z)
        w, v = np.linalg.eig(H)
        
        v = v[:,np.abs(w).argsort()]
        w = w[np.abs(w).argsort()]
        return w, v

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

def plot_normal_modes(sim, i=0, scale=3, surface=False):
    w, v = sim.normal_modes()
    freq = np.sqrt(np.abs(w)) / (2 * _c.pi)

    freq_i = freq[i]
    v_i = np.reshape(np.real(v[:,i]), (3, -1))
    x = sim.x[-1,0:3,:]
    
    drum = np.sum(np.abs(v_i[2,:])) > 0.1
    if drum:
        rmax = np.max(np.abs(sim.x[-1, 0:3, :])) * 1.2
    else:
        rmax = np.max(np.abs(sim.x[-1, 0:3, :])) * 2
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
