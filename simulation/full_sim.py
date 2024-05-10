import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cvxpy as cp
import os
import json
import random

#from typing import Callable
from collections import defaultdict

from utils import *

MAX_RETRIES = 100
STOP_THRESH = 3
NUM_TRIALS = 20
VIABLE_QUERY_TYPES = ['paq', 'paq_noisy_low', 'paq_noisy_med', 'paq_noisy_high', 'paired_comparison', 'triplet']

def run_one_exp() -> None:
    num_ref = 25
    eigenvalue_gap = 10

    num_meas = 20
    query_type = 'triplet'
    gt_out = generate_gt(num_ref, eigenvalue_gap)
    thresh = 5
    noise_frac = 0
    normalize = False
    square = False

    # From GT values, generate measurements
    meas_out = generate_measurements(num_meas, gt_out, query_type=query_type, thresh=thresh, noise_frac=noise_frac)
    print(meas_out)
    # Use queries to estimate metrics
    est_out = estimate_metric(meas_out, query_type=query_type, thresh=thresh)

    for k, v in est_out.items():
        print(f'n = {k} | sig_hat = {v}')

    const = 10
    w_star = gt_out['w_star']
    angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
    for a in angles_out:
        print(a*180/np.pi)

    w_hat, status = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

    print(w_hat)
    print(w_star)
    print(compute_err(w_star, w_hat, normalize=normalize, squared=square))

def run_sweep() -> None:
    num_ref = 25
    eigenvalue_gap = 10
    queries = ['paired_comparison', 'triplet', 'paq', 'paq_noisy_low', 'paq_noisy_med', 'paq_noisy_high']
    meas_v = [3, 5, 10, 25, 50, 75, 100, 125, 150, 200, 250]
    thresh = 5
    const_start = 1
    normalize = False
    square = False

    fname = 'results_baseline_nonnorm.json'

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            res = json.load(f)
    else:
        res = defaultdict(dict)


    for query_type in queries:
        if query_type in res or query_type not in VIABLE_QUERY_TYPES:
            continue

        res_query = defaultdict(list) # maps num_meas -> [avg err, std err, [trial err]]

        if 'low' in query_type:
            noise_frac = 0.1
        elif 'med' in query_type:
            noise_frac = 0.5
        elif 'high' in query_type:
            noise_frac = 1
        else:
            noise_frac = 0

        for num_meas in meas_v:
            if 'paq' not in query_type and num_meas < 10:
                continue

            print(f'Processing {query_type} at {num_meas} measurements')
            err_mc = []

            for mc in range(NUM_TRIALS):
                # Generate ground truth values
                gt_out = generate_gt(num_ref, eigenvalue_gap)

                # From GT values, generate measurements
                meas_out = generate_measurements(num_meas, gt_out, query_type=query_type, thresh=thresh, noise_frac=noise_frac)

                # Use queries to estimate metrics
                est_out = estimate_metric(meas_out, query_type=query_type, thresh=thresh)

                const = const_start
                w_star = gt_out['w_star']
                angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
                w_hat, status = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

                if w_hat is not None:
                    err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
                    err_best = err
                else: 
                    err_best = 1e9

                ct = 0
                stop_sign = 0
                while ct < MAX_RETRIES and stop_sign < STOP_THRESH: #status != cp.OPTIMAL and 
                    if const == 1:
                        const += 9
                    else:
                        const += 10
                    angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
                    w_hat, status = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star = w_star)
                    ct += 1

                    if w_hat is not None:
                        err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
                        if err < err_best:
                            err_best = err
                            stop_sign = 0
                        else:
                            stop_sign += 1

                
                err_mc.append(err_best)

            res_query[num_meas] = [np.average(err_mc), np.std(err_mc) / NUM_TRIALS, err_mc]

        res[query_type] = res_query
        with open(fname, 'w') as f:
            json.dump(res, f, indent=4)
                

if __name__ == '__main__':
    #run_one_exp()
    #run_sweep()
    fname = 'results_baseline_nonnorm.json'
    plot_figure(fname)