import json
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import statistics
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from utilities import plot_cov_ellipse, euclidean_distance, ellipse_point

solvers = [cp.SCS, cp.CVXOPT]


center_points = [(0.25, 0.34), (0.29, 0.40), (0.37, 0.40), (0.35, 0.35)]

def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def divide_data_by_flag(data):
    """Divide data into three sets based on the 'flag' value."""
    fast_data = {}
    medium_data = {}
    slow_data = {}

    for page_id, values in data.items():

        flag = values['endColor']['flag']
        if flag == 'fast':
            fast_data[page_id] = values
        elif flag == 'medium':
            medium_data[page_id] = values
        else:
            slow_data[page_id] = values

    return fast_data, medium_data, slow_data

def subtract_lists(l1, l2):
    l3 = l1.copy()
    for item in l2:
        if item in l3:
            l3.remove(item)
    return l3

def metric_est_and_ellipse_plot(data, thresh: float = 1e-3, num_meas: int = 10, max_iter: int = 10000, flag: str = ""):
    
    est_out = []
    data_points = {}
    gamma_squared = []

    fig, ax = plt.subplots() 
    Sig_hat = cp.Variable((2,2), PSD=True)
    losses = [0,0,0,0]
    i = 0
    for _, details in data.items():

        gamma = details['gamma']
        if flag == "slow":
            gamma_squared.append(gamma**2)
        elif flag == "median":
            gamma_squared.append(4*gamma**2)
        else: 
            gamma_squared.append((1/0.33*gamma)**2)
        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])

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

    # print(data_points)
    # print(gammas_collect)
    # for ref_pt in center_points:
        
    #     gamma_values = gammas_collect[ref_pt]
    #     variance_collect.append(np.var(gamma_values))

    for j in range(4):
        obj = cp.Minimize(losses[j])
        prob = cp.Problem(obj)
        prob.solve(solver=cp.SCS, max_iters = max_iter)

        ct = 1
        while prob.status != cp.OPTIMAL and ct < len(solvers):
            prob.solve(solver=solvers[ct], max_iters = max_iter)
            ct += 1
        # print(f'Tried {ct + 1} solvers, status {prob.status}')
        est_out.append(Sig_hat.value)

    # # plotting
    # for center in center_points:
    #     ax.scatter(center[0], center[1], c='blue', marker='o', s=50, label='Center Points' if center == center_points[0] else "")

    # # Plot corresponding data points around each center point in red
    # for center, points in data_points.items():
    #     ax.scatter([p[0] for p in points], [p[1] for p in points], c='red', marker='x',label='Data Points' if center == center_points[0] else "")


    for i in range(len(center_points)):
        plot_cov_ellipse(est_out[i], center_points[i], data_points[tuple(center_points[i])], nstd=0.3, ax=ax, edgecolor='red', facecolor='none')

    # Set plot limits based on the range of data points and centers
    all_points = [point for sublist in data_points.values() for point in sublist] + center_points
    all_x = [point[0] for point in all_points]
    all_y = [point[1] for point in all_points]
    
    ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
    ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid()
    plt.show()

    return gamma_squared

# Main execution
if __name__ == "__main__":
    data = load_data('ui_local/data/color_data_lorraine3.json')
    fast_data, medium_data, slow_data = divide_data_by_flag(data)

    gamma_squared_1 = metric_est_and_ellipse_plot(fast_data, flag = "fast")
    gamma_squared_2 = metric_est_and_ellipse_plot(medium_data, flag = "median")
    gamma_squared_3 = metric_est_and_ellipse_plot(slow_data, flag = "slow")

    print("gamma_squared_1: ", gamma_squared_1)
    print("gamma_squared_2: ", gamma_squared_2)
    print("gamma_squared_3: ", gamma_squared_3)

    seq1 = subtract_lists(gamma_squared_1, gamma_squared_2)
    seq2 = subtract_lists(gamma_squared_2, gamma_squared_3)

    mean_value_1 = statistics.mean(seq1)
    variance_value1 = statistics.variance(seq1) 
    mean_value_2 = statistics.mean(seq2)
    variance_value2 = statistics.variance(seq2) 

    print("mean value for seq 1: ", mean_value_1)
    print("mean value for seq 2: ", mean_value_2)
    print("variance for seq 1: ", variance_value1)
    print("variance for seq 2: ", variance_value2)
    




