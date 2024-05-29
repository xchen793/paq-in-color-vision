import json
import numpy as np
import cvxpy as cp


solvers = [cp.SCS, cp.CVXOPT]
name = "L"

center_points = [(0.25, 0.34), (0.29, 0.40)]

def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def divide_data_by_flag(data):
    fast_data = {}
    slow_data = {}

    for page_id, details in data.items():
        flag = details['endColor']['flag']
        
        # Constructing a unique key from both fixedColor and query_vec
        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
        unique_key = (fixed, query)
        
        if flag == 'fast':
            fast_data[page_id] = details
        elif flag == 'slow':
            slow_data[page_id] = details
        else:
            print(f"Unexpected flag value: {flag} for page_id {page_id}")

    return fast_data, slow_data

# def check_key_completeness(fast_data, slow_data):
#     fast_keys = set(fast_data.keys())
#     slow_keys = set(slow_data.keys())

#     if not (fast_keys  == slow_keys):
#         print("Mismatch in keys across datasets")
#         if fast_keys != slow_keys:
#             print("Differences between fast and slow:", fast_keys.symmetric_difference(slow_keys))
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

def store_noise_by_keys(data, thresh: float = 1, num_meas: int = 10):
    gamma_dict = {} #store the gamma
    i = 0
    for _, details in data.items():
        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
        key = (fixed, query)
        
        if key not in gamma_dict:
            i += 1
            gamma_dict[key] = []

        gamma = details['gamma']
        gamma_dict[key].append(gamma)

    return gamma_dict


# Main execution
if __name__ == "__main__":
    data = load_data('ui_local/data/reject_null/color_data_lorraine.json')
    fast_data, slow_data = divide_data_by_flag(data)
    # check_key_completeness(fast_data, slow_data)

    fast_gamma = store_noise_by_keys(fast_data)
    slow_gamma = store_noise_by_keys(slow_data)

    mean_fast_gamma = {}
    std_fast_gamma = {}
    var_fast_gamma = {}
    for key, gamma_list in fast_gamma.items():
        mean_fast_gamma[key] = np.mean(gamma_list)
        std_fast_gamma[key] = np.std(gamma_list)
        var_fast_gamma[key] = np.var(gamma_list)


    mean_slow_gamma = {}
    std_slow_gamma = {}
    var_slow_gamma = {}
    for key, gamma_list in slow_gamma.items():
        mean_slow_gamma[key] = np.mean(gamma_list)
        std_slow_gamma[key] = np.std(gamma_list)
        var_slow_gamma[key] = np.var(gamma_list)

    print("------------------mean_fast_gamma-------------------") 
    for key, value in mean_fast_gamma.items():
        print("{}: {:.3f} x 10^-3".format(key, value * 1000))
    print("------------------mean_slow_gamma-------------------")
    for key, value in mean_slow_gamma.items():
        print("{}: {:.3f} x 10^-3".format(key, value * 1000))
    print("------------------std_fast_gamma-------------------")
    for key, value in std_fast_gamma.items():
        print("{}: {:.3f} x 10^-3".format(key, value * 1000))
    print("------------------std_slow_gamma-------------------")
    for key, value in std_slow_gamma.items():
        print("{}: {:.3f} x 10^-3".format(key, value * 1000))
    print("------------------var_fast_gamma-------------------")
    for key, value in var_fast_gamma.items():
        print("{}: {:.3f} x 10^-3".format(key, value * 1000))
    print("------------------var_slow_gamma-------------------")
    for key, value in var_slow_gamma.items():
        print("{}: {:.3f} x 10^-3".format(key, value * 1000))

    

    
    





    