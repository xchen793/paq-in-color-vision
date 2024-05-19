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


def subtract_lists(dict1, dict2):
    result = {}

    for key, value1 in dict1.items():
        if key in dict2:
            value2 = dict2[key]
            result[key] = value1 - value2

    return result


def merge_by_first_key_element(dictionary):
    aggregated_results = {}
    for key, values in dictionary.items():
        first_key_element = key[0]  

        if first_key_element not in aggregated_results:
            aggregated_results[first_key_element] = []

        aggregated_results[first_key_element].append(values)

    return aggregated_results

def metric_est_and_ellipse_plot(data, thresh: float = 1e-3, num_meas: int = 10, max_iter: int = 10000, flag: str = ""):
    
    est_out = []
    data_points = {}
    gamma_squared = {}

    fig, ax = plt.subplots() 
    Sig_hat = cp.Variable((2,2), PSD=True)
    losses = [0,0,0,0]
    i = 0
    for _, details in data.items():

        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        gamma = details['gamma']
        query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
        flag = details['endColor']['flag']
        key = (fixed, query)
        if key not in gamma_squared:
            gamma_squared[key] = None

        if flag == "slow":
            gamma_squared[key] = gamma**2
        elif flag == "medium":
            gamma_squared[key] = 4*gamma**2
        elif flag == "fast":
            gamma_squared[key] = (1/0.33*gamma)**2
        else:
            print(f"Unhandled flag {flag}")

        # for key in gamma_squared:
        #     print(f"Generated key: {key} for flag {flag}")
        
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
    data = load_data('ui_local/data/color_data.json')
    fast_data, medium_data, slow_data = divide_data_by_flag(data)
    # check_key_completeness(fast_data, medium_data, slow_data)


    gamma_squared_1 = metric_est_and_ellipse_plot(fast_data, flag = "fast")
    gamma_squared_2 = metric_est_and_ellipse_plot(medium_data, flag = "medium")
    gamma_squared_3 = metric_est_and_ellipse_plot(slow_data, flag = "slow")


    seq1 = subtract_lists(gamma_squared_1, gamma_squared_2)
    seq2 = subtract_lists(gamma_squared_2, gamma_squared_3)


    merged_seq1 = merge_by_first_key_element(seq1)
    merged_seq2 = merge_by_first_key_element(seq2)
    # print("seq1: ", merged_seq1)
    # print("seq2: ", merged_seq2)

    statistics1 = {key: {"mean": np.mean(values), "variance": np.var(values)} for key, values in merged_seq1.items()}
    statistics2 = {key: {"mean": np.mean(values), "variance": np.var(values)} for key, values in merged_seq2.items()}

    print("mean value for seq 1: ", statistics1)
    print("mean value for seq 2: ", statistics2)

    




