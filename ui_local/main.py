import json
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp



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
    """Divide data into two sets based on the 'flag' value."""
    long_data = {}
    short_data = {}

    for page_id, values in data.items():
        flag = values['endColor']['flag']
        if flag == 'long':
            if page_id not in long_data:
                long_data[page_id] = {}
            long_data[page_id] = values
        elif flag == 'short':
            if page_id not in short_data:
                short_data[page_id] = {}
            short_data[page_id] = values

    return long_data, short_data

def metric_est_and_ellipse_plot(data, thresh: float = 1e-3, num_meas: int = 10, max_iter: int = 10000):
    
    est_out = []
    data_points = {}

    fig, ax = plt.subplots() 
    Sig_hat = cp.Variable((2,2), PSD=True)
    losses = [0,0,0,0]
    loss_values = [[] for _ in range(4)]
    i = 0
    for _, details in data.items():

        gamma = details['gamma']
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

    for j in range(4):
        obj = cp.Minimize(losses[j])
        prob = cp.Problem(obj)
        prob.solve(solver=cp.SCS, max_iters = max_iter)

        ct = 1
        while prob.status != cp.OPTIMAL and ct < len(solvers):
            prob.solve(solver=solvers[ct], max_iters = max_iter)
            ct += 1
        print(f'Tried {ct + 1} solvers, status {prob.status}')
        est_out.append(Sig_hat.value)

        # Store the optimized loss value for each group
        optimized_loss = losses[j].value
        loss_values[j].append(optimized_loss)

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

    print(loss_values)

    return est_out

# Main execution
if __name__ == "__main__":
    data = load_data('ui_local/data/color_data_lorraine_2.json')
    long_data, short_data = divide_data_by_flag(data)
    metric_est_and_ellipse_plot(long_data)
    metric_est_and_ellipse_plot(short_data)

