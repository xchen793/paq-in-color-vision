import json
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import colour
from matplotlib import colors as mcolors


solvers = [cp.SCS, cp.CVXOPT]
name = "L"

center_points = [(0.25, 0.34), (0.29, 0.40), (0.37, 0.40), (0.35, 0.35)]

def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def divide_data_by_flag(data):
    fast_data = {}
    medium_data = {}
    slow_data = {}

    for page_id, details in data.items():
        flag = details['endColor']['flag']
        
        # Constructing a unique key from both fixedColor and query_vec
        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
        unique_key = (fixed, query)
        
        if flag == 'fast':
            fast_data[unique_key] = details
        elif flag == 'medium':
            medium_data[unique_key] = details
        elif flag == 'slow':
            slow_data[unique_key] = details
        else:
            print(f"Unexpected flag value: {flag} for page_id {page_id}")

    return fast_data, medium_data, slow_data

# def check_key_completeness(fast_data, medium_data, slow_data):
#     fast_keys = set(fast_data.keys())
#     medium_keys = set(medium_data.keys())
#     slow_keys = set(slow_data.keys())

#     all_keys = fast_keys | medium_keys | slow_keys

#     if not (fast_keys == medium_keys == slow_keys):
#         print("Mismatch in keys across datasets")
#         if fast_keys != medium_keys:
#             print("Differences between fast and medium:", fast_keys.symmetric_difference(medium_keys))
#         if fast_keys != slow_keys:
#             print("Differences between fast and slow:", fast_keys.symmetric_difference(slow_keys))
#         if medium_keys != slow_keys:
#             print("Differences between medium and slow:", medium_keys.symmetric_difference(slow_keys))
#     else:
#         print("All datasets have the same keys.")

def get_person_label(person: str, idx: int) -> str:
    if person == 'ashwin':
        return 'Color blind user'
    else:
        return f'Color normal, user {idx}'

def plot_cov_ellipse(cov, center, data_points, plot_color, person, idx, nstd=2, ax=None, **kwargs):
    """Plot an ellipse representing the covariance matrix."""
    if ax is None:
        fig, ax = plt.subplots()


    # Ensure there are data points to process
    if data_points:
        data_x, data_y = zip(*data_points)  # Unzip data points
        #ax.scatter(data_x, data_y, c='orange', marker='x', label='PAQ queries')
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

    ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, edgecolor=plot_color, facecolor='none', label=get_person_label(person, idx))

    ax.add_patch(ellipse)
    ax.set_aspect('equal')

    # Create a proxy artist for the legend (use a red line for simplicity)
    legend_proxy = mpatches.Patch(color=plot_color, label='My Ellipse')

    # Add legend using the proxy artist
    ax.legend(handles=[legend_proxy])   


    # Calculate endpoints of the major axis
    elongation_factor = 14
    angle_rad = np.radians(theta)
    dx = width / 2 * np.cos(angle_rad) * elongation_factor  # Multiply dx by elongation factor
    dy = width / 2 * np.sin(angle_rad) * elongation_factor  # Multiply dy by elongation factor
    
    # Define points for the major axis line
    x1, y1 = center[0] - dx, center[1] - dy
    x2, y2 = center[0] + dx, center[1] + dy

    # Add line for the major axis
    #ax.plot([x1, x2], [y1, y2], plot_color, linestyle='--')

    return ax

<<<<<<< HEAD
def metric_est_and_ellipse_plot(data, thresh: float = 1e-3, num_meas: int = 20, max_iter: int = 10000, flag: str = ""):
    
    est_out = []
    data_points = {}
    gamma_squared = {}

    fig, ax = plt.subplots(figsize=(6, 5)) 
    Sig_hat = cp.Variable((2,2), PSD=True)
    losses = [0,0,0,0]
    i = 0
    for _, details in data.items():

        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        gamma = details['gamma']
        query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
        # flag = details['endColor']['flag']
        key = (fixed, query)
        if key not in gamma_squared:
            gamma_squared[key] = None

        
        a_x = details['query_vec']['x']
        a_y = details['query_vec']['y']
        a = np.array([a_x, a_y])
        feature_matrix = gamma * np.outer(a, a)

        data_x = fixed[0] + gamma * a_x
        data_y = fixed[1] + gamma * a_y

        if fixed not in data_points: 
            data_points[fixed] = []
        data_points[fixed].append((data_x, data_y))


        if i < 20:
            losses[0] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
        elif i < 40:
            losses[1] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
        elif i < 60:
            losses[2] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
        else:
            losses[3] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
        
        i += 1


    for j in range(4):
        obj = cp.Minimize(losses[j])
        prob = cp.Problem(obj)
        prob.solve(solver=cp.SCS, max_iters = max_iter)

        ct = 1
        while prob.status != cp.OPTIMAL and ct < len(solvers):
            prob.solve(solver=solvers[ct], max_iters = max_iter)
            ct += 1

        est_out.append(Sig_hat.value)

    for i in range(len(center_points)):
        plot_cov_ellipse(est_out[i], center_points[i], data_points[tuple(center_points[i])], nstd=0.3, ax=ax, edgecolor='red', facecolor='none')

    # Set plot limits based on the range of data points and centers
    all_points = [point for sublist in data_points.values() for point in sublist] + center_points
    all_x = [point[0] for point in all_points]
    all_y = [point[1] for point in all_points]
    
    ax.set_xlim(0.12, 0.50)
    ax.set_ylim(0.23, 0.53)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='x-large')
    ax.grid()
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title(f"Color blind D ellipse plot", fontsize=20)
    plt.savefig(f'D_elliipse.png', format='png', dpi=300)
=======
def metric_est_and_ellipse_plot(thresh: float = 5 * 1e-4, num_meas: int = 10, max_iter: int = 10000, flag: str = ""):
    fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

    # Load the JSON data from the file
    people = ['ashwin', 'austin', 'jingyan', 'lorraine']
    plot_colors = [mcolors.CSS4_COLORS['crimson'], mcolors.CSS4_COLORS['palegreen'], mcolors.CSS4_COLORS['forestgreen'], mcolors.CSS4_COLORS['darkgreen']]
    for idx, (person, plot_color) in enumerate(zip(people, plot_colors)): 
        data = load_data(f'./data/prev/color_data_{person}.json')

        est_out = []
        data_points = {}
        gamma_squared = {}

        #fig, ax = plt.subplots() 
        Sig_hat = cp.Variable((2,2), PSD=True)
        losses = [0,0,0,0]
        i = 0
        for _, details in data.items():

            fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
            gamma = details['gamma']
            query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
            # flag = details['endColor']['flag']
            key = (fixed, query)
            if key not in gamma_squared:
                gamma_squared[key] = None

            
            a_x = details['query_vec']['x']
            a_y = details['query_vec']['y']
            a = np.array([a_x, a_y])
            feature_matrix = gamma * np.outer(a, a)

            data_x = fixed[0] + gamma * a_x
            data_y = fixed[1] + gamma * a_y

            if fixed not in data_points: 
                data_points[fixed] = []
            data_points[fixed].append((data_x, data_y))


            if i < 10:
                losses[0] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
            elif i < 20:
                losses[1] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
            elif i < 30:
                losses[2] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
            else:
                losses[3] += (thresh - cp.trace(feature_matrix @ Sig_hat))**2 / num_meas
            
            i += 1


        for j in range(4):
            obj = cp.Minimize(losses[j])
            prob = cp.Problem(obj)
            prob.solve(solver=cp.SCS, max_iters = max_iter)

            ct = 1
            while prob.status != cp.OPTIMAL and ct < len(solvers):
                prob.solve(solver=solvers[ct], max_iters = max_iter)
                ct += 1

            est_out.append(Sig_hat.value)

        for i in range(len(center_points)):
            plot_cov_ellipse(est_out[i], center_points[i], data_points[tuple(center_points[i])], plot_color, person, idx, nstd=0.5, ax=ax, edgecolor=plot_color, facecolor='none')

        # Set plot limits based on the range of data points and centers
        all_points = [point for sublist in data_points.values() for point in sublist] + center_points
        all_x = [point[0] for point in all_points]
        all_y = [point[1] for point in all_points]
        
        #ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
        #ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid()
        #ax.set_title(f"Normal trichromat {name} ellipse plot")
    #plt.show()
    plt.savefig(f'all_elliipse.png', format='png', dpi=300)
>>>>>>> a7ae9fd7f04d7c9b6dea1aa82a304ec2baba7fcd

    return gamma_squared


if __name__ == "__main__":
<<<<<<< HEAD
    data = load_data('ui_local/data/prev/color_data_ashwin.json')
=======
    
>>>>>>> a7ae9fd7f04d7c9b6dea1aa82a304ec2baba7fcd
    # fast_data, medium_data, slow_data = divide_data_by_flag(data)
    # check_key_completeness(fast_data, medium_data, slow_data)
    metric_est_and_ellipse_plot()