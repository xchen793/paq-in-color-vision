import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sp
import cvxpy as cp
import os
import json
import random

from datetime import datetime

from collections import defaultdict

from utils import *

font = {'size'   : 12}

matplotlib.rc('font', **font)

def plot_multiple(dirs: list, save_path: str, save_name: str, sweep_type: str, query_plot: list = []) -> None:
    save_fpath = os.path.join(save_path, save_name)

    plt.figure(0)
    plt.figure(figsize=(8, 6), dpi=80)
    for dir in dirs:
        result_fname = os.path.join(dir, f"{sweep_type}_sweep.json")
        config_fname = os.path.join(dir, f"config_{sweep_type}_sweep.json")

        with open(result_fname, 'r') as f:
            res = json.load(f)

        with open(config_fname, 'r') as fc:
            config = json.load(fc)

        for query_type in query_plot:
            results = res[query_type]
            sweep_param = list(results.keys())
            sweep_param = sorted([int(i) for i in sweep_param])
            results = {int(k):v for k, v in results.items()}

            err_mean = [results[n]['err'][0] for n in sweep_param]
            err_std = [results[n]['err'][1] for n in sweep_param]

            err_std_low = [err_mean[i] - err_std[i] for i in range(len(err_mean))]
            err_std_high = [err_mean[i] + err_std[i] for i in range(len(err_mean))]

            plt.loglog(sweep_param, err_mean, label = get_label_multiple(query_type, config['num_meas']))
            plt.fill_between(sweep_param, err_std_low, err_std_high, alpha = 0.5)
        
    plt.xlabel(get_x_axis_label(sweep_type), fontsize=20)
    plt.ylabel('Copunctal point estimation error', fontsize=20)
    plt.legend(loc='best', ncol=1) #'right', bbox_to_anchor=(1, 0.32))
    plt.savefig(save_fpath, bbox_inches = 'tight')
    #plt.show()


### PLOT reference sweeps, val err
dirs = [
    './sweep_results/sweep_ref_2024-06-25_20-24-49', # m = 10
    './sweep_results/sweep_ref_2024-06-25_20-22-21', # m = 25
    './sweep_results/sweep_ref_2024-06-25_20-22-59', # m = 50
    './sweep_results/sweep_ref_2024-06-25_20-24-33', # m = 75
    './sweep_results/sweep_ref_2024-06-25_21-59-16'  # m = 100
]
VIABLE_QUERY_TYPES = ['paq', 'paq_noisy_low', 'paq_noisy_med', 'paq_noisy_high']
for qt in VIABLE_QUERY_TYPES:
    plot_list = [qt]
    plot_list_name = '_'.join(plot_list)
    save_name = f'./ref_plots/sweep_ref_multiple_meas_{plot_list_name}.pdf'
    plot_multiple(dirs, '.', save_name, 'ref', plot_list)

### PLOT reference sweeps, val fails
dirs = [
    './sweep_results/sweep_ref_2024-06-26_09-28-38', # m = 75
    './sweep_results/sweep_ref_2024-06-26_09-28-47', # m = 50
    './sweep_results/sweep_ref_2024-06-26_09-29-03', # m = 10
    './sweep_results/sweep_ref_2024-06-26_09-29-12', # m = 100
    './sweep_results/sweep_ref_2024-06-26_09-31-34', # m = 25
]
dirs = dirs[::-1]
VIABLE_QUERY_TYPES = ['paq', 'paq_noisy_low', 'paq_noisy_med', 'paq_noisy_high']
for qt in VIABLE_QUERY_TYPES:
    plot_list = [qt]
    plot_list_name = '_'.join(plot_list)
    save_name = f'./ref_plots_val_fails/sweep_ref_multiple_meas_{plot_list_name}.pdf'
    plot_multiple(dirs, '.', save_name, 'ref', plot_list)

### PLOT gap sweeps, val err
dirs = [
    './sweep_results/sweep_gap_2024-06-26_07-32-56',  # m = 10
    './sweep_results/sweep_gap_2024-06-26_07-32-39',  # m = 25
    './sweep_results/sweep_gap_2024-06-25_22-03-15',  # m = 50
    './sweep_results/sweep_gap_2024-06-25_22-00-14',  # m = 75
    './sweep_results/sweep_gap_2024-06-25_22-00-06'   # m = 100
]
for qt in VIABLE_QUERY_TYPES:
    plot_list = [qt]
    plot_list_name = '_'.join(plot_list)
    save_name = f'./gap_plots/sweep_gap_multiple_meas_{plot_list_name}.pdf'
    plot_multiple(dirs, '.', save_name, 'gap', plot_list)

### PLOT gap sweeps, val fails
# dirs = [
#     '', # m = 100
#     '', # m = 75
#     '', # m = 50
#     '', # m = 25
#     '' # m = 10
# ]
# dirs = dirs[::-1]
# VIABLE_QUERY_TYPES = ['paq', 'paq_noisy_low', 'paq_noisy_med', 'paq_noisy_high']
# for qt in VIABLE_QUERY_TYPES:
#     plot_list = [qt]
#     plot_list_name = '_'.join(plot_list)
#     save_name = f'./gap_plots_val_fails/sweep_ref_multiple_meas_{plot_list_name}.pdf'
#     plot_multiple(dirs, '.', save_name, 'ref', plot_list)
