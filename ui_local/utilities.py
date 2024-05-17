import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def ellipse_point(theta, center, axes, angle):
    """Calculate the coordinates of a point on an ellipse given an angle."""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    x = center[0] + axes[0] * np.cos(theta) * c - axes[1] * np.sin(theta) * s
    y = center[1] + axes[0] * np.cos(theta) * s + axes[1] * np.sin(theta) * c
    return x, y

# def plot_cov_ellipse(cov, center, data_points, nstd=2, ax=None, **kwargs):
#     """Plot an ellipse representing the covariance matrix."""
#     if ax is None:
#         fig, ax = plt.subplots()

#     data_x, data_y = zip(*data_points) 
#     ax.scatter(data_x, data_y, c='blue', label='Data Points')

#     vals, vecs = np.linalg.eigh(cov)
#     order = vals.argsort()[::-1]
#     vals = vals[order]
#     vecs = vecs[:,order]

#     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
#     width, height = 2 * nstd * np.sqrt(vals)
#     ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)
#     ax.add_patch(ellipse)
#     ax.set_aspect('equal')
#     return ax

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_cov_ellipse(cov, center, data_points, nstd=2, ax=None, **kwargs):
    """Plot an ellipse representing the covariance matrix."""
    if ax is None:
        fig, ax = plt.subplots()


    # Ensure there are data points to process
    if data_points:
        data_x, data_y = zip(*data_points)  # Unzip data points
        ax.scatter(data_x, data_y, c='orange', marker='x', label='PAQ queries')
    else:
        print("Warning: No data points provided.")

    ax.scatter(center[0], center[1], c='blue', marker='o', s=20, label='Reference points')
    # Calculate eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Eigenvalue check
    if np.all(vals < 1e-10):
        print("Warning: Very small eigenvalues; ellipse might be too small to see.")
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    print("Width:", width, "Height:", height)  # Debugging output

    # # Check and adjust width and height if too small
    # if width < 1e-2 or height < 1e-2:
    #     print("Adjusting width/height for visibility.")
    #     width = height = 0.1  # Default small value to make ellipse visible

    ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, edgecolor='r', facecolor='none')

    ax.add_patch(ellipse)
    ax.set_aspect('equal')

    # Create a proxy artist for the legend (use a red line for simplicity)
    legend_proxy = mpatches.Patch(color='red', label='My Ellipse')

    # Add legend using the proxy artist
    ax.legend(handles=[legend_proxy])   


    # Calculate endpoints of the major axis
    elongation_factor = 3.8
    angle_rad = np.radians(theta)
    dx = width / 2 * np.cos(angle_rad) * elongation_factor  # Multiply dx by elongation factor
    dy = width / 2 * np.sin(angle_rad) * elongation_factor  # Multiply dy by elongation factor
    
    # Define points for the major axis line
    x1, y1 = center[0] - dx, center[1] - dy
    x2, y2 = center[0] + dx, center[1] + dy

    # Add line for the major axis
    ax.plot([x1, x2], [y1, y2], 'red', linestyle='--', label="Major Axis")

    return ax
