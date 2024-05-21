import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cvxpy as cp
import os
import json
import random
import argparse
import datetime 

from collections import defaultdict

from utils import *

MAX_RETRIES = 30
STOP_THRESH = 5
NUM_TRIALS = 20
VIABLE_QUERY_TYPES = ['paq', 'paq_noisy_low', 'paq_noisy_med', 'paq_noisy_high', 'paired_comparison', 'triplet']

REF_SWEEP = 'ref'
GAP_SWEEP = 'gap'
MEAS_SWEEP = 'meas'

def run_one_exp() -> None:
    num_ref = 25
    eigenvalue_gap = 10

    num_meas = 250
    query_type = 'paq'
    gt_out = generate_gt(num_ref, eigenvalue_gap)
    print('Generated GT')
    thresh = 5
    noise_frac = 0
    normalize = False
    square = False

    # From GT values, generate measurements
    meas_out = generate_measurements(num_meas, gt_out, query_type=query_type, thresh=thresh, noise_frac=noise_frac)
    print('Generated metric')
    # Use queries to estimate metrics
    est_out = estimate_metric(meas_out, query_type=query_type, thresh=thresh)
    print('Estimated metrics')
    print(est_out)

    # for k, v in est_out.items():
    #     print(f'{k} : {v}')
    #est_out = estimate_metric(meas_out, query_type=query_type, thresh=thresh)

    # for k, v in est_out.items():
    #     print(f'n = {k} | sig_hat = {v}')

    const = 1
    w_star = gt_out['w_star']
    angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const, pca=True)
    angles_deg_out = [a*180/np.pi for a in angles_out]
    print(f'angles: {angles_deg_out}')

    w_hat, status, num_fails = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

    print(f'est: {w_hat} | true: {w_star} | fails: {num_fails}')
    if w_hat is not None:
        err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
        print(f'err: {err}')

def run_sweep(config: dict, dir: str, plot_only: bool = False) -> str:

    num_ref = config['num_ref']
    eigenvalue_gap = config['eigenvalue_gap']
    queries = config['queries']
    num_meas = config['num_meas']
    thresh = config['thresh']
    const_start = config['const_start']
    normalize = config['normalize']
    square = config['square']
    if 'pca' not in config:
        use_pca = False
    else:
        use_pca = config['pca']

    sweep_type = config['sweep_type']
    if sweep_type == MEAS_SWEEP:
        fname = 'meas_sweep.json'
        sweep_range = num_meas
    elif sweep_type == GAP_SWEEP:
        fname = 'gap_sweep.json'
        sweep_range = eigenvalue_gap
    elif sweep_type == REF_SWEEP:
        fname = 'ref_sweep.json'
        sweep_range = num_ref

    fpath = os.path.join(dir, fname)
    if plot_only:
        return fpath

    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            res = json.load(f)
    else:
        res = defaultdict(dict)

    for query_type in queries:
        if query_type in res or query_type not in VIABLE_QUERY_TYPES:
            continue

        res_query = defaultdict(list) 

        if 'low' in query_type:
            noise_frac = 0.1
        elif 'med' in query_type:
            noise_frac = 0.5
        elif 'high' in query_type:
            noise_frac = 0.75
        else:
            noise_frac = 0

        for sweep_var in sweep_range:
            if sweep_type == MEAS_SWEEP:
                num_meas = sweep_var
                if 'paq' not in query_type and num_meas < 10:
                    continue
                print(f'Processing {query_type} at {num_meas} measurements')
            elif sweep_type == GAP_SWEEP:
                eigenvalue_gap = sweep_var
                print(f'Processing {query_type} at {eigenvalue_gap} eig gap')
            elif sweep_type == REF_SWEEP:
                num_ref = sweep_var
                print(f'Processing {query_type} w/ {num_ref} references')

            err_mc = []
            num_fails_mc = []
            best_const_mc = []

            for mc in range(NUM_TRIALS):
                if (mc + 1) % 5 == 0:
                    print(f'Processed {mc + 1} / {NUM_TRIALS} \t | err so far = {np.average(err_mc):.4f} \t | fails so far = {np.average(num_fails_mc):.4f} \t | const so far = {np.average(best_const_mc):.4f}')

                # Generate ground truth values
                gt_out = generate_gt(num_ref, eigenvalue_gap)
                # From GT values, generate measurements
                meas_out = generate_measurements(num_meas, gt_out, query_type=query_type, thresh=thresh, noise_frac=noise_frac)

                # Use queries to estimate metrics
                if not use_pca:
                    est_out = estimate_metric(meas_out, query_type=query_type, thresh=thresh)
                else:
                    est_out = estimate_metric_pca(meas_out, gt_out, thresh=thresh, query_type=query_type)

                const = const_start
                w_star = gt_out['w_star']
                angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const, pca=use_pca)
                w_hat, status, num_fails = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

                if w_hat is not None:
                    err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
                    err_best = err
                    num_fails_best = num_fails
                else: 
                    err_best = 1e9
                    num_fails_best = 1e9

                #print(f'errors: {err} | num_fails: {num_fails}')
                ct = 0
                stop_sign = 0
                best_const = const
                #const_mult = const_start
                while ct < MAX_RETRIES and stop_sign < STOP_THRESH: 
                    if const == 1:
                        const += 4
                    elif const < 1:
                        const += 0.1
                    else:
                        const += 5
                    #const_mult += 0.1
                    #const = const_mult * num_meas
                    angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
                    w_hat, status, num_fails = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star = w_star)
                    ct += 1
                    

                    if w_hat is not None:
                        err = compute_err(w_star, w_hat, normalize=normalize, squared=square)
                        #print(f'errors: {err} | num_fails: {num_fails}')
                        if num_fails < num_fails_best:
                            err_best = err
                            num_fails_best = num_fails
                            stop_sign = 0
                            best_const = const
                        elif num_fails == num_fails_best:
                            if err < err_best:
                                err_best = err
                                best_const = const
                                stop_sign = 0
                            else:
                                stop_sign += 1
                        else:
                            stop_sign += 1

                #print(f'best const: {best_const} | num violations: {num_fails_best} | err: {err_best}')
                err_mc.append(err_best)
                num_fails_mc.append(num_fails_best)
                best_const_mc.append(best_const)
                #print(f'Best performance had {num_fails_best} / {2*num_ref} violated constraints')

            print(f'Finished sweep, avg err = {np.average(err_mc):.4f} |\t avg fails = {np.average(num_fails_mc):.4f} |\t best const: {np.average(best_const_mc):.4f} |\t stop: {stop_sign}')
            print('--------')
            res_query[sweep_var] = {
                'err': [np.average(err_mc), np.std(err_mc) / NUM_TRIALS, err_mc],
                'fails' : [np.average(num_fails_mc), np.std(num_fails_mc) / NUM_TRIALS, num_fails_mc],
                'const' : [best_const_mc]
            }

        res[query_type] = res_query
        with open(fpath, 'w') as f:
            json.dump(res, f, indent=4)

    return fpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run different types of sweeps.")

    parser.add_argument("config", type=str, default='', help="Filepath to JSON configuration file")
    parser.add_argument("-test_sweep", action='store_true', help = "Bypass sweep, run test sweep")
    parser.add_argument("-plot", action='store_true', help = "Bypass sweep, plot figure")
    parser.add_argument("-plot_file", type=str, default='', help='Filepath to sweep output JSON to plot')
    
    args = parser.parse_args()

    if args.test_sweep:
        run_one_exp()
    else:
        with open(args.config, 'r') as file:
            config_data = json.load(file)
            
        sweep_type = config_data['sweep_type']
        if not args.plot:
            dir = create_directory(sweep_type)
            fpath = run_sweep(config_data, dir, plot_only=args.plot)
        else:
            fpath = args.plot_file

        if sweep_type == "meas":
            save_name = 'meas_sweep.pdf'
        elif sweep_type == 'gap':
            save_name = 'gap_sweep.pdf'
        elif sweep_type == 'ref':
            save_name = 'ref_sweep.pdf'

        plot_figure(fpath, save_name, sweep_type)