import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


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
        plt.plot(sim.t, sim.x[dim, :, :].T)
    else:
        plt.plot(sim.t, sim.x[dim, i, :].T)

    plt.xlabel('Time')
    plt.ylabel(['x Position', 'y Position', 'z Position',
                'vx Velocity', 'vy Velocity', 'vz Velocity'][dim])
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))


def animate(ions):
    # First set up the figure, the axis, and the plot element we want to animate
    h_axis = 0
    v_axis = 1
    rmax = np.max(ions.x[:, 0:2, :]) * 1.1

    fig = plt.figure()
    ax = plt.axes(xlim=(-rmax, rmax), ylim=(-rmax, rmax))

    points, = ax.plot([], [], 'b.', markersize=20)
    lines = [ax.plot([], [], 'k-')[0] for _ in range(ions.N)]
#     plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    def init():
        points.set_data([], [])
        for line in lines:
            line.set_data([], [])
        return lines

    # animation function.  This is called sequentially
    def draw_frame(frame_i):
        points.set_data(ions.x[h_axis, :, frame_i], ions.x[v_axis, :, frame_i])
        for i, line in enumerate(lines):
            line.set_data(ions.x[h_axis, i, (frame_i - 5):frame_i],
                          ions.x[v_axis, i, (frame_i - 5):frame_i])
        return lines

    anim = animation.FuncAnimation(fig, draw_frame, init_func=init,
                                   frames=ions.t.size,
                                   interval=(1000 / 60),
                                   blit=True,
                                   repeat=False)

    plt.close(fig)
    return anim
