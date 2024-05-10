import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cvxpy as cp
import os
import json
import random
import argparse

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

    w_hat, status, num_fails = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

    print(w_hat)
    print(w_star)
    print(compute_err(w_star, w_hat, normalize=normalize, squared=square))

def run_sweep(config: dict, plot_only: bool = False) -> str:

    num_ref = config['num_ref']
    eigenvalue_gap = config['eigenvalue_gap']
    queries = config['queries']
    num_meas = config['num_meas']
    print(num_meas)
    thresh = config['thresh']
    const_start = config['const_start']
    normalize = config['normalize']
    square = config['square']

    sweep_type = config['sweep_type']
    if sweep_type == 'meas':
        fname = 'meas_sweep.json'
        sweep_range = num_meas
    elif sweep_type == 'gap':
        fname = 'gap_sweep.json'
        sweep_range = eigenvalue_gap
    elif sweep_type == 'ref':
        fname = 'ref_sweep.json'
        sweep_range = num_ref

    if plot_only:
        return fname

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            res = json.load(f)
    else:
        res = defaultdict(dict)

    print(res)
    for query_type in queries:
        if query_type in res or query_type not in VIABLE_QUERY_TYPES:
            continue

        res_query = defaultdict(list) 

        if 'low' in query_type:
            noise_frac = 0.1
        elif 'med' in query_type:
            noise_frac = 0.5
        elif 'high' in query_type:
            noise_frac = 1
        else:
            noise_frac = 0

        for sweep_var in sweep_range:
            if sweep_type == 'meas':
                num_meas = sweep_var
                if 'paq' not in query_type and num_meas < 10:
                    continue
                print(f'Processing {query_type} at {num_meas} measurements')
            elif sweep_type == 'gap':
                eigenvalue_gap = sweep_var
                print(f'Processing {query_type} at {eigenvalue_gap} eig gap')
            elif sweep_type == 'ref':
                num_ref = sweep_var
                print(f'Processing {query_type} w/ {num_ref} references')

            err_mc = []
            num_fails_mc = []

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
                w_hat, status, num_fails = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

                if w_hat is not None:
                    err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
                    err_best = err
                    num_fails_best = num_fails
                else: 
                    err_best = 1e9
                    num_fails_best = 0

                ct = 0
                stop_sign = 0
                while ct < MAX_RETRIES and stop_sign < STOP_THRESH: 
                    if const == 1:
                        const += 9
                    else:
                        const += 10
                    angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
                    w_hat, status, num_fails = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star = w_star)
                    ct += 1

                    if w_hat is not None:
                        err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
                        if err < err_best:
                            err_best = err
                            num_fails_best = num_fails
                            stop_sign = 0
                        else:
                            stop_sign += 1

                
                err_mc.append(err_best)
                num_fails_mc.append(num_fails_best)
                print(f'Best performance had {num_fails_best} / {2*num_ref} violated constraints')

            res_query[sweep_var] = [np.average(err_mc), np.std(err_mc) / NUM_TRIALS, err_mc, num_fails_mc]

        res[query_type] = res_query
        with open(fname, 'w') as f:
            json.dump(res, f, indent=4)

    return fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run different types of sweeps.")

    parser.add_argument("config", type=str, help="Filepath to JSON configuration file")
    parser.add_argument("-plot", action='store_true', help = "Bypass sweep, plot figure")
    
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config_data = json.load(file)

    fname = run_sweep(config_data, plot_only=args.plot)
    if config_data['sweep_type'] == "meas":
        save_name = 'meas_sweep.pdf'
    elif config_data['sweep_type'] == 'gap':
        save_name = 'gap_sweep.pdf'
    elif config_data['sweep_type'] == 'ref':
        save_name = 'ref_sweep.pdf'

    plot_figure(fname, save_name, config_data['sweep_type'])