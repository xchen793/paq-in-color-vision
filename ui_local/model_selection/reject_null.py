import json
import numpy as np
import cvxpy as cp
import scipy.stats as stats
import scipy.stats as norm
import matplotlib.pyplot as plt


solvers = [cp.SCS, cp.CVXOPT]

center_points = [(0.25, 0.34), (0.29, 0.40)]

def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def divide_data_by_flag(data):
    fast_data = {}
    slow_data = {}

    model_rejection = data.get("model_rejection", {})
    
    for key, value in model_rejection.items():
        if key.startswith("survey_page") and isinstance(value, dict):
            details = value
            if 'endColor' in details:
                flag = details['endColor'].get('flag')

                if flag == 'fast':
                    fast_data[key] = details
                elif flag == 'slow':
                    slow_data[key] = details
                else:
                    print(f"Unexpected flag value: {flag} for key {key}")
            else:
                print(f"No 'endColor' key found for key {key}")

    return fast_data, slow_data


# def subtract_lists(dict1, dict2):
#     result = {}

#     for key, value1 in dict1.items():
#         if key in dict2:
#             value2 = dict2[key]
#             result[key] = value1 - value2

#     return result


def merge_by_first_key_element(dictionary):
    aggregated_results = {}
    for key, values in dictionary.items():
        first_key_element = key[0]  

        if first_key_element not in aggregated_results:
            aggregated_results[first_key_element] = []

        aggregated_results[first_key_element].append(values)

    return aggregated_results

def store_noise_by_keys(data, scale_factor: float = 1):
    gamma_inv_dict = {} #store the gamma
    i = 0
    for _, details in data.items():
        fixed = (details['fixedColor']['x'], details['fixedColor']['y'])
        query = (details['query_vec']['x'], details['query_vec']['y'], details['query_vec']['Y'])
        key = (fixed, query)
        
        if key not in gamma_inv_dict:
            i += 1
            gamma_inv_dict[key] = []

        gamma = details['gamma']
        gamma_sq = (gamma * scale_factor) ** 2
        gamma_inv_dict[key].append(gamma_sq)

    return gamma_inv_dict

def scale_gamma(data, scale_factor: float):
    for _, details in data.items():
        details['gamma'] *= scale_factor


def f_test(val_1, val_2, df, alpha):
    F = val_1 / val_2
    df1 = df
    df2 = df
    p_value = 1 - stats.f.cdf(F, df1, df2)

    F_crit = stats.f.ppf(1 - alpha, df1, df2)
    print("p_value: ", p_value)

    if F > F_crit:
        print("Reject the null hypothesis: The variances are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The variances are not significantly different.")
    return p_value


def sweep_alpha_and_plot(data):

    alphas = np.arange(0, 1.01, 0.01)
    p_values = [f_test(var1, var2, 29, alpha) for alpha in alphas] # A numpy array ranging from 0 to 1 in steps of 0.01

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, p_values, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('P-value')
    plt.title('Alpha vs P-value')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # data = load_data('data/model_rejection_lorraine.json')
    data = load_data('data/prev/model_rejection_austin.json')
    fast_data, slow_data = divide_data_by_flag(data)

    fast_gamma = store_noise_by_keys(fast_data, 1/2)
    slow_gamma = store_noise_by_keys(slow_data)

    combined_gamma = {}

    # Combine fast_gamma and slow_gamma
    for key, gamma_list in fast_gamma.items():
        if key not in combined_gamma:
            combined_gamma[key] = []
        combined_gamma[key].extend(gamma_list)

    for key, gamma_list in slow_gamma.items():
        if key not in combined_gamma:
            combined_gamma[key] = []
        combined_gamma[key].extend(gamma_list)

    # Calculate statistics for combined data
    mean_gamma = {}
    std_gamma = {}
    var_gamma = {}

    for key, gamma_list in combined_gamma.items():
        mean_gamma[key] = np.mean(gamma_list)
        std_gamma[key] = np.std(gamma_list)
        var_gamma[key] = np.var(gamma_list, ddof=1)

    print("------------------mean(gamma^2)-------------------") 
    for key, value in mean_gamma.items():
        if key == ((0.32, 0.32), (1, 0, 0)):
            mean1 = value
        else:
            mean2 = value
        print("{}: {}".format(key, value))
    print("------------------var(gamma^2)-------------------")
    for key, value in var_gamma.items():
        if key == ((0.32, 0.32), (1, 0, 0)):
            var1 = value
        else:
            var2 = value
        print("{}: {}".format(key, value))


    f_test(var1, var2, 29, 0.05)
    # sweep_alpha_and_plot(data)



    


        

        
        





        