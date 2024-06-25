import json
import numpy as np
import cvxpy as cp
import scipy.stats as stats

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
        gamma_sq = gamma**2
        gamma_inv_dict[key].append(gamma_sq)

    return gamma_inv_dict


# Main execution
if __name__ == "__main__":
    data = load_data('ui_local/data/reject_null/color_data_lorraine.json')
    fast_data, slow_data = divide_data_by_flag(data)
    # check_key_completeness(fast_data, slow_data)

    fast_gamma = store_noise_by_keys(fast_data)
    slow_gamma = store_noise_by_keys(slow_data)

    fast_gamma_dict = {}
    mean_fast_gamma = {}
    std_fast_gamma = {}
    var_fast_gamma = {}

    for key, gamma_list in fast_gamma.items():
        fast_gamma_dict[key] = gamma_list
        mean_fast_gamma[key] = np.mean(gamma_list)
        std_fast_gamma[key] = np.std(gamma_list)
        var_fast_gamma[key] = np.var(gamma_list, ddof=1)

    mean_slow_gamma = {}
    std_slow_gamma = {}
    var_slow_gamma = {}
    for key, gamma_list in slow_gamma.items():
        mean_slow_gamma[key] = np.mean(gamma_list)
        std_slow_gamma[key] = np.std(gamma_list)
        var_slow_gamma[key] = np.var(gamma_list,ddof=1)

    print("------------------mean_fast(gamma^2)-------------------") 
    for key, value in mean_fast_gamma.items():
        if key == ((0.25, 0.34), (-1, 0, 0)):
            mean1 = value
            gamma_list1 = fast_gamma_dict[key]
        elif key == ((0.25, 0.34), (1, 0, 0)):
            mean2 = value
            gamma_list2 = fast_gamma_dict[key]
        elif key == ((0.25, 0.34), (0, 1, 0)):
            mean3 = value
            gamma_list3 = fast_gamma_dict[key]
        elif key == ((0.25, 0.34), (0, -1, 0)):
            mean4 = value
            gamma_list4 = fast_gamma_dict[key]
        print("{}: {}".format(key, value))
    # print("------------------mean_slow(gamma^2)-------------------")
    # for key, value in mean_slow_gamma.items():
    #     print("{}: {}".format(key, value))
    # print("------------------std_fast(gamma^2)-------------------")
    # for key, value in std_fast_gamma.items():
    #     print("{}: {}".format(key, value))
    # print("------------------std_slow(gamma^2)-------------------")
    # for key, value in std_slow_gamma.items():
    #     print("{}: {}".format(key, value))
    print("------------------var_fast(gamma^2)-------------------")
    for key, value in var_fast_gamma.items():
        if key == ((0.25, 0.34), (-1, 0, 0)):
            var1 = value
        elif key == ((0.25, 0.34), (1, 0, 0)):
            var2 = value
        elif key == ((0.25, 0.34), (0, 1, 0)):
            var3 = value
        elif key == ((0.25, 0.34), (0, -1, 0)):
            var4 = value
        print("{}: {}".format(key, value))
    # print("------------------var_slow(gamma^2)-------------------")
    # for key, value in var_slow_gamma.items():
    #     print("{}: {}".format(key, value))

####################### Tests ########################
        
def t_test(mean1, mean2):
    # Perform T-test    
    t_stat, p_value = stats.ttest_ind(mean1, mean2, equal_var=True) 

    # Significance level
    alpha = 0.05

    # Make a decision
    if p_value < alpha:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")

def levene_test(gamma_list1, gamma_list2, alpha):
    _, p_value = stats.levene(gamma_list1, gamma_list2)
    if p_value < alpha:
        print("Reject the null hypothesis: The variances are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The variances are not significantly different.")

def f_test(val_1, val_2, df, alpha):
    F = val_1 / val_2
    df1 = df
    df2 = df

    F_crit = stats.f.ppf(1 - alpha, df1, df2)

    # Make a decision
    if F > F_crit:
        print("Reject the null hypothesis: The variances are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The variances are not significantly different.")


if __name__ == "__main__":
    # check for mean: do we simply compare them or use t test?
    if mean1 == mean3:
        print("Means are not equal.")
    else:
        # Check normality using Shapiro-Wilk test
        _, p_value1 = stats.shapiro(gamma_list1)
        _, p_value2 = stats.shapiro(gamma_list3)

        # Significance level
        alpha = 0.05

        # Make a decision
        if p_value1 > alpha and p_value2 > alpha:
            print("Both samples are normally distributed. F test is used to compare the variances.")
            f_test(var1, var3, df = 9, alpha = alpha)
        else:
            print("At least one sample is not normally distributed. Levene's test(non-parametric) is used to compare the variances.")
            levene_test(gamma_list1, gamma_list3, alpha = alpha)

    # if not mean1 == mean3:
    #     f_test(var1, var3, df = 9, alpha = 0.05)
    # if not mean3 == mean4:
    #     f_test(var3, var4, df = 9, alpha = 0.05)


        

        
        





        