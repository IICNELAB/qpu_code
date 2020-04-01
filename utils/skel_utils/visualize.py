import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation 
import numpy as np
from .skeleton import Skeleton


def plot_skeleton(frame, ax, xyz, parents, points=None, lines=None, colors=None):
    """
    Visualize a human skeleton.
    Args:
        frame: frame to be visulized
        ax: Axes3D
        xyz: np.array(3, T, J) 3D coordinates of skeleton
        parents: [parent[i], ...] list of parents
        points: [[ax.plot()], ...]
        lines: [[ax.plot()], ...]
    """
    # Init point and line plots
    if points is None:
        points = [ax.plot([], [], [], marker='o') for _ in range(xyz.shape[2])]
    if lines is None:
        lines = [ax.plot([], [], []) for _ in range(xyz.shape[2] - 1)]
    if colors is not None:
        for i in range(len(lines)):
            lines[i][0].set_color(colors[i])

    skeleton = xyz[:, frame, :]

    # Plot joints
    for i in range(len(parents)):
        points[i][0].set_xdata(skeleton[0][i])
        points[i][0].set_ydata(skeleton[1][i])
        points[i][0].set_3d_properties(skeleton[2][i])
    # Plot bones
    i = 0
    for i, parent in enumerate(parents):
        if parent is not -1:
            lines[i][0].set_xdata([skeleton[0, i], skeleton[0, parent]])
            lines[i][0].set_ydata([skeleton[1, i], skeleton[1, parent]])
            lines[i][0].set_3d_properties([skeleton[2, i], skeleton[2, parent]])
    # Set plot range
    x_max, x_min = np.max(xyz[0, :, :]), np.min(xyz[0, :, :])
    y_max, y_min = np.max(xyz[1, :, :]), np.min(xyz[1, :, :])
    z_max, z_min = np.max(xyz[2, :, :]), np.min(xyz[2, :, :])
    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def skeleton_anim(fig, ax, xyz, parents, fps=20, colors=None):
    """Create an animation of skeletons in xyz at given fps.
    Args:
        fig: plt.figure
        ax: Axes3D
        xyz: np.array(3, T, J)
        parents: [parent[i], ...]
        fps: number
    """
    
    # Init point and line plots
    points = [ax.plot([], [], [], marker='o') for _ in range(xyz.shape[2])]
    lines = [ax.plot([], [], []) for _ in range(xyz.shape[2])]
    anim = FuncAnimation(fig, plot_skeleton, frames=np.arange(0, xyz.shape[1]), interval=1000./fps, fargs=[ax, xyz, parents, points, lines, colors], repeat_delay=2000)
    return anim

