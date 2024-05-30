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

from sklearn.decomposition import PCA

np.random.seed(2024)
solvers = [cp.SCS]

font = {'size'   : 14}

matplotlib.rc('font', **font)

# forward model
def generate_gt(n_ref: int, eigenvalue_gap: int) -> dict:

    gt_out = {}

    area_part1 = 0.8 * 1   # Area of (0.2, 1) x (0, 1)
    area_part2 = 0.2 * 0.8 # Area of (0, 0.2) x (0.2, 1)
    total_area = area_part1 + area_part2
    refs = []

    w_star = (random.uniform(0, 0.15), random.uniform(0, 0.15))   # Generate the random point w_star
    gt_out['w_star'] = w_star

    for _ in range(n_ref):  # generate reference points
        ref = np.zeros(2,)
        if np.random.rand() < area_part1 / total_area:
            # Sample from Part 1
            ref[0] = np.random.uniform(0.2, 1)
            ref[1] = np.random.uniform(0, 1)
        else:
            # Sample from Part 2
            ref[0] = np.random.uniform(0, 0.2)
            ref[1] = np.random.uniform(0.2, 1)
        refs.append(ref)

    gt_out['refs'] = refs
    

    ############  Part 2: Get u2 star, compute u1 star  ############
    u2_stars = [] # store u2 vectors
    u1_stars = [] # store u1 vectors
    for ref in refs:
        dx = w_star[0] - ref[0]
        dy = w_star[1] - ref[1]
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            u2_star = np.array([dx / magnitude, dy / magnitude])
            u1_star = np.array([-u2_star[1], u2_star[0]])
        else:
            u2_star = np.array([0, 0])
            u1_star = np.array([0, 0])

        u2_stars.append(u2_star)
        u1_stars.append(u1_star)

    gt_out['eigenvectors'] = [u1_stars, u2_stars]
    ############  Part 3: Compute true metrics  ############
    sigma_stars = []
    for u2_star, u1_star in zip(u2_stars, u1_stars):
        lambda1 = random.uniform(eigenvalue_gap, 2*eigenvalue_gap) # randomly generate lambda1
        lambda2 = lambda1 - eigenvalue_gap 
        U = np.column_stack((u1_star, u2_star))
        Lambda = np.diag([lambda1, lambda2])
        sigma_star = U @ Lambda @ U.T
        sigma_stars.append(sigma_star)

    gt_out['sigma_stars'] = sigma_stars
    

    return gt_out
    

# step 1: Generate measurements
# For a fixed query type, returns a dict:
#   [key] n : [value] responses
#
#   responses is a list of tuples (feature_vector, response)
#   feature_vector changes for query_type
#       For paq, it's the direction
#       For paired comparison, its the item compared against the reference
#       For triplets, its 2 vectors that are compared against the reference


def generate_measurements(num_meas: int, gt: dict, query_type: str = 'paq', thresh: float = 1, noise_frac: float = 0.5) -> dict:
    # iterate over each reference
    num_ref = len(gt['refs'])
    responses = {}
    
    for n in range(num_ref):
        responses_n = []
        feat_matrix_n = np.zeros((2, 2, num_meas))
        item_vectors = []
        Sig_n = gt['sigma_stars'][n]
        if 'paq' in query_type:      
            for i in range(num_meas):
                a_i = np.random.normal(size=(2))
                quad_i = a_i.T @ Sig_n @ a_i
                if 'noisy' in query_type:
                    gamma_i = (thresh + np.random.uniform(low=-noise_frac*thresh, high=noise_frac*thresh)) / quad_i
                else:
                    gamma_i = thresh / quad_i

                feat_matrix_n[:,:,i] = np.outer(a_i, a_i)
                responses_n.append( gamma_i )
                item_vectors.append(a_i)
        
        elif query_type == 'triplet':
            for i in range(num_meas):
                delta_i = np.random.normal(size=(2,2))
                feat_matrix_n[:,:,i] = np.outer(delta_i[0] - delta_i[1], delta_i[0] - delta_i[1])

                dist = np.trace(feat_matrix_n[:,:,i] @ Sig_n)

                if dist > thresh:
                    outcome = 1
                else:
                    outcome = -1

                responses_n.append(outcome)
                item_vectors.append(delta_i)

        elif query_type == 'paired_comparison':
            for i in range(num_meas):
                delta_i = np.random.normal(size=(2,))
                feat_matrix_n[:,:,i] = np.outer(delta_i, delta_i)
                
                dist = np.trace(feat_matrix_n[:,:,i] @ Sig_n)

                if dist > thresh:
                    outcome = 1
                else:
                    outcome = -1

                responses_n.append(outcome)
                item_vectors.append(delta_i)

        responses[n] = [feat_matrix_n, responses_n, item_vectors]
    return responses


# step 1: estimate metric
def estimate_metric(responses: dict, thresh: float = 1, query_type: str = 'paq', max_iter: int = 10000) -> dict:
    est_out = {}
    n_ref = len(responses)
    for n in range(n_ref):
        feat_matrix_n, labels, _ = responses[n]
        #feat_flat_n = feat_matrix_n.reshape(4, -1)
        num_meas = len(labels)

        # estimation
        Sig_hat = cp.Variable((2,2), PSD=True)
        loss = 0
        for i in range(num_meas):
            feat_matrix = feat_matrix_n[:,:,i]
            if 'paq' in query_type:
                meas = thresh / labels[i] 
                loss += (meas - cp.trace(feat_matrix @ Sig_hat))**2 / num_meas

                #loss = cp.sum_squares( thresh - (cp.vec(Sig_hat) @ feat_flat_n) ) / num_meas #+ 0.1 * cp.norm(Sig_hat, 'fro')

            elif query_type == 'triplet':
                loss += cp.pos(1 - cp.multiply(labels[i], cp.trace(feat_matrix @ Sig_hat) - thresh)) / num_meas
                #loss = cp.sum(cp.pos( 1 - cp.multiply(labels, (cp.vec(Sig_hat) @ feat_flat_n)) ) ) / num_meas #+ 0.1 * cp.norm(Sig_hat, 'fro')

            elif query_type == 'paired_comparison':
                loss += cp.pos(1 - cp.multiply(labels[i], cp.trace(feat_matrix @ Sig_hat) - thresh)) / num_meas
                #loss = cp.sum(cp.pos( thresh - cp.multiply(labels, (cp.vec(Sig_hat) @ feat_flat_n))) ) / num_meas #+ 0.1 * cp.norm(Sig_hat, 'fro')


        obj = cp.Minimize(loss)
        prob = cp.Problem(obj)

        prob.solve(solver=cp.SCS, max_iters = max_iter)#, verbose=True)
        ct = 1
        while prob.status != cp.OPTIMAL and ct < len(solvers):
            prob.solve(solver=solvers[ct], max_iters = max_iter)
            ct += 1
        
        #print(f'Tried {ct + 1} solvers, status {prob.status}')
        
        est = Sig_hat.value
        est_out[n] = est
    
    return est_out

def estimate_metric_pca(responses: dict, gt: dict, thresh : float = 1, query_type: str = 'paq') -> dict:
    est_out = {}
    n_ref = len(responses)
    num_meas = len(responses[0][1])
    refs = gt['refs']
    #plt.figure(figsize=(8, 6))
    for n in range(n_ref):
        response_items = []
        _, labels, query_dirs = responses[n]
        labels = [np.sqrt(l) for l in labels]
        for i in range(num_meas):
            response_item = (labels[i] * query_dirs[i].reshape(1,-1))
            response_items.append(response_item)

        response_items = np.concatenate(response_items, axis=0)

        XtX = response_items.T @ response_items / (num_meas - 1)
        
        U, Sig, _ = np.linalg.svd(XtX)
        Sig_est = U @ np.diag(Sig) @ U.T

        #scales = [1]
        # for i in range(num_meas):
        #     ga_i = response_items[i]
        #     scale = thresh / (ga_i.T @ Sig_est_unscaled @ ga_i)
        #     scales.append(scale)

        #Sig_est = np.mean(scales) * Sig_est_unscaled
        #print(f'mean scale: {np.mean(scales)} | Sig: {Sig}')

        # sorted_indices = np.argsort(Sig)[::-1]
        # Sig = Sig[sorted_indices]
        # U = U[:, sorted_indices]

        # # Extract principal components
        # pc1 = U[0]  # First principal component
        # pc2 = U[1]  # Second principal component

        # #Plot the data and principal components
        
        # #plt.scatter(response_items[:,0] + refs[n][0], response_items[:,1] + refs[n][1], alpha=0.6, label='Data')

        # plt.quiver(refs[n][0], refs[n][1], pc1[0], pc1[1], angles='xy', scale_units='xy', scale=0.1, color='r', label='Principal Component 1')
        # #plt.quiver(0, 0, pc2[0], pc2[1], angles='xy', scale_units='xy', scale=2, color='g', label='Principal Component 2')


        est_out[n] = Sig_est

    # print(gt['w_star'])
    # plt.scatter(gt['w_star'][0], gt['w_star'][1])
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.axhline(0, color='black',linewidth=0.5)
    # plt.axvline(0, color='black',linewidth=0.5)
    # plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    # plt.axis('equal')
    # plt.title('Principal Components of 2D PCA')
    # plt.show()
    return est_out

# step 2: compute cones
def compute_cones(est_out: dict, gt: dict, num_meas: int, cheat: bool = False, const: float = 10, pca: bool = False) -> tuple[list, list]:
    alphas = []
    major_axes = []

    for n, sigma_hat in est_out.items():
        # compute svd of Sig_hat
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_hat)
        for i, e in enumerate(eigenvalues):
            if e < 0:
                eigenvalues[i] = -e
                eigenvectors[i] = -eigenvectors[i]

        if not pca:
            indices = np.argsort(eigenvalues)[::-1]
        else:
            indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        lambda_1_hat_n, lambda_2_hat_n = eigenvalues[0], eigenvalues[1]
        u1_hat, u2_hat = eigenvectors[:, 0], eigenvectors[:, 1]
        
        if cheat:
            tau = np.linalg.norm(sigma_hat - gt['sigma_stars'][n], 2)
        else:
            tau = const * 2 / num_meas

        alpha = 4 * tau / np.abs(lambda_1_hat_n - lambda_2_hat_n)
        alphas.append(alpha)
        major_axes.append(u2_hat)

    return alphas, major_axes

def feas_check(w_star: np.ndarray, zn: np.ndarray, u2_hat: np.ndarray, alpha: float) -> bool:
    u21_hat, u22_hat = u2_hat[0], u2_hat[1]

    cos_alpha_2 = np.cos(alpha / 2)
    sin_alpha_2 = np.sin(alpha / 2)

    cond_one = (w_star[1] - zn[1]) * (cos_alpha_2 * u21_hat - sin_alpha_2 * u22_hat) <= (w_star[0] - zn[0]) * (sin_alpha_2 * u21_hat + cos_alpha_2 * u22_hat)
    cond_two = (w_star[1] - zn[1]) * (cos_alpha_2 * u21_hat + sin_alpha_2 * u22_hat) >= (w_star[0] - zn[0]) * (-u21_hat * sin_alpha_2 + u22_hat * cos_alpha_2)

    return (cond_one and cond_two)

# step 3: Estimate copunct
def estimate_copunctal(alphas: dict, major_axes: dict, refs: list, w_star: np.ndarray = None) -> tuple[np.ndarray, str, int]:
    w = cp.Variable(2) # Decision variable for the copunctal point w
    constraints = []
    num_fails = 0

    # Add constraints for each reference point and its corresponding cone
    num_constraints = 0
    for zn, u2_hat, alpha in zip(refs, major_axes, alphas):
        alpha = min(alpha, np.pi)
        #print(alpha)
        #u2_hat = 100*u2_hat
        # Unpack the eigenvalue vector for readability
        u21_hat, u22_hat = u2_hat[0], u2_hat[1]

        cos_alpha_2 = np.cos(alpha / 2)
        sin_alpha_2 = np.sin(alpha / 2)

        
        if w_star is not None and not feas_check(w_star, zn, u2_hat, alpha):
            if not feas_check(w_star, zn, -u2_hat, alpha):
                # print(f'Error cone alpha: {alpha}')
                # print(feas_check(w_star, zn, u2_hat, np.pi))
                # print('failed')
                # print('-----')
                num_fails += 2
                continue
            else:
                num_constraints += 2
                #print(f'Use neg, {num_constraints} added')
                u21_hat *= -1
                u22_hat *= -1
        else:
            num_constraints += 2
            #print(f'{num_constraints} constraints added')
        #print('-----')

        # Upper side of the cone
        constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat - sin_alpha_2 * u22_hat) <= (w[0] - zn[0]) * (sin_alpha_2 * u21_hat + cos_alpha_2 * u22_hat))
        # Lower side of the cone
        constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat + sin_alpha_2 * u22_hat) >= (w[0] - zn[0]) * (-u21_hat * sin_alpha_2 + u22_hat * cos_alpha_2))
    
    #print(f'{num_fails}----')
    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)#, verbose=True)
    ct = 0
    while problem.status != cp.OPTIMAL and ct < len(solvers):
        problem.solve(solver=solvers[ct])
        ct += 1

    return w.value, problem.status, num_fails

def compute_err(w_star: np.ndarray, w_hat: np.ndarray, normalize: bool = True, squared: bool = True) -> float:
    
    err = np.linalg.norm(w_star - w_hat)
    if normalize:
        err /= np.linalg.norm(w_star)
    if squared:
        err = err**2
    return err

def get_label(query_name: str) -> str:
    if query_name == 'paq':
        return 'PAQ (noiseless)'
    elif query_name == 'paq_noisy_low':
        return 'PAQ (low noise)'
    elif query_name == 'paq_noisy_med':
        return 'PAQ (medium noise)'
    elif query_name == 'paq_noisy_high':
        return 'PAQ (high noise)'
    elif query_name == 'paired_comparison':
        return 'Paired comparison\n(noiseless)'
    elif query_name == 'triplet':
        return 'Triplet (noiseless)'
    
def get_x_axis_label(sweep_type: str) -> str:
    if sweep_type == 'meas':
        return 'Number of responses per reference'
    elif sweep_type == 'gap':
        return 'Eigenvalue gap'
    elif sweep_type == 'ref':
        return 'Number of reference points'

def plot_figure(fname: str, save_path: str, save_name: str, sweep_type: str) -> None:
    with open(fname, 'r') as f:
        res = json.load(f)
    
    save_fpath = os.path.join(save_path, save_name)
    plt.figure(0)
    plt.figure(figsize=(8, 6), dpi=80)
    for query_type, results in res.items():
        sweep_param = list(results.keys())
        sweep_param = sorted([int(i) for i in sweep_param])
        results = {int(k):v for k, v in results.items()}

        err_mean = [results[n]['err'][0] for n in sweep_param]
        err_std = [results[n]['err'][1] for n in sweep_param]

        err_std_low = [err_mean[i] - err_std[i] for i in range(len(err_mean))]
        err_std_high = [err_mean[i] + err_std[i] for i in range(len(err_mean))]

        plt.loglog(sweep_param, err_mean, label = get_label(query_type))
        plt.fill_between(sweep_param, err_std_low, err_std_high, alpha = 0.5)
    
    plt.xlabel(get_x_axis_label(sweep_type), fontsize=20)
    plt.ylabel('Copunctal point estimation error', fontsize=20)
    #plt.legend()
    plt.legend(loc='best')#'right', bbox_to_anchor=(1, 0.32))
    plt.savefig(save_fpath, bbox_inches = 'tight')
    plt.show()

    save_name = f'failure_rate_{save_name}'
    save_fpath = os.path.join(save_path, save_name)
    plt.figure(1)
    for query_type, results in res.items():
        sweep_param = list(results.keys())
        sweep_param = sorted([int(i) for i in sweep_param])
        results = {int(k):v for k, v in results.items()}

        err_mean = [results[n]['fails'][0] / 50 for n in sweep_param]
        err_std = [results[n]['fails'][1] / 50 for n in sweep_param]

        err_std_low = [err_mean[i] - err_std[i] for i in range(len(err_mean))]
        err_std_high = [err_mean[i] + err_std[i] for i in range(len(err_mean))]

        plt.plot(sweep_param, err_mean, label = get_label(query_type))
        plt.fill_between(sweep_param, err_std_low, err_std_high, alpha = 0.5)
    
    plt.xlabel(get_x_axis_label(sweep_type), fontsize=24)
    plt.ylabel('Number of constraint violations', fontsize=24)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(save_fpath, bbox_inches = 'tight')
    plt.show()


def create_directory(sweep_type:str) -> str:
    # Get today's date and time
    now = datetime.now()
    
    # Format the date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create directory name using formatted date and time
    directory_name = f"./sweep_results/sweep_{sweep_type}_{date_time}"
    
    # Create the directory
    try:
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")

    return directory_name