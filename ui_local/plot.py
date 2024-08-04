import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Ellipse
from matplotlib import colors as mcolors

def mahalanobis_distance(x, mean, inv_cov):
    """Calculate the Mahalanobis distance."""
    delta = x - mean
    return delta.T @ inv_cov @ delta

def objective_function(cov_params, data_points, fixed_point):
    """Objective function to minimize the sum of squared Mahalanobis distances."""
    cov_matrix = np.array([[cov_params[0], cov_params[1]], [cov_params[1], cov_params[2]]])
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    total_cost = 0
    for point in data_points:
        dist = mahalanobis_distance(np.array(point), np.array(fixed_point), inv_cov_matrix)
        total_cost += dist**2
    return total_cost

#fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

# Load the JSON data from the file
people = ['ashwin', 'austin', 'jingyan', 'lorraine'][:2]
plot_colors = [mcolors.CSS4_COLORS['crimson'], mcolors.CSS4_COLORS['palegreen'], mcolors.CSS4_COLORS['forestgreen'], mcolors.CSS4_COLORS['crimson']][:2]
for person, plot_color in zip(people, plot_colors): 
    with open(f'./data/prev/color_data_{person}.json', 'r') as file:
        data = json.load(file)

    fixed_points_data = {}

    # Collect data points for each fixed point
    for page, details in data.items():
        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        query_x = details['query_vec']['x']
        query_y = details['query_vec']['y']
        gamma = details['gamma']
        data_x = fixed[0] + gamma * query_x
        data_y = fixed[1] + gamma * query_y

        if fixed not in fixed_points_data:
            fixed_points_data[fixed] = []
        fixed_points_data[fixed].append((data_x, data_y))

    for fixed, data_points in fixed_points_data.items():
        data_points = np.array(data_points)
        initial_cov_params = [0.1, 0, 0.1]  # Initial guesses for variances and covariance

        # Optimize the covariance parameters
        result = minimize(objective_function, initial_cov_params, args=(data_points, fixed), bounds=[(1e-6, None), (None, None), (1e-6, None)])
        optimal_cov_params = result.x
        cov_matrix = np.array([[optimal_cov_params[0], optimal_cov_params[1]], [optimal_cov_params[1], optimal_cov_params[2]]])

        # Plot the ellipse
        vals, vecs = np.linalg.eigh(cov_matrix)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:,order]
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 0.3 * np.sqrt(vals)  # 2 std deviations
        ellipse_patch = Ellipse(fixed, width, height, angle=theta, edgecolor=plot_color, facecolor='none', lw=2)
        ax.add_patch(ellipse_patch)
        
        #ax.scatter(*zip(*data_points), color=plot_color, s=30)
        ax.scatter(*fixed, color='blue', s=100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Optimized Mahalanobis Distance Ellipses')
plt.xlim([0, 0.9])
plt.ylim([0, 0.9])
plt.grid(True)
plt.show()
