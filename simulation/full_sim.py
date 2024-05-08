import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cvxpy as cp
import random

#from typing import Callable
from collections import defaultdict

np.random.seed(2024)
solvers = [cp.SCS, cp.CVXOPT]

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


def generate_measurements(num_meas: int, gt: dict, query_type: str = 'paq', thresh: float = 1) -> dict:
    # iterate over each reference
    num_ref = len(gt['refs'])
    responses = {}
    
    for n in range(num_ref):
        responses_n = []
        feat_matrix_n = np.zeros((2, 2, num_meas))
        Sig_n = gt['sigma_stars'][n]
        if 'paq' in query_type:      
            for i in range(num_meas):
                a_i = np.random.normal(size=(2))
                quad_i = a_i.T @ Sig_n @ a_i
                if 'noisy' in query_type:
                    gamma_i = (thresh + np.random.uniform(low=-thresh, high=thresh)) / quad_i
                else:
                    gamma_i = thresh / quad_i

                feat_matrix_n[:,:,i] = gamma_i * np.outer(a_i, a_i)
                responses_n.append( gamma_i )
        
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

        responses[n] = [feat_matrix_n, responses_n]
    return responses


# step 1: estimate metric
def estimate_metric(responses: dict, thresh: float = 1, query_type: str = 'paq', max_iter: int = 10000) -> dict:
    est_out = {}
    n_ref = len(responses)
    for n in range(n_ref):
        feat_matrix_n, labels = responses[n]
        feat_flat_n = feat_matrix_n.reshape(4, -1)
        num_meas = len(labels)

        # estimation
        Sig_hat = cp.Variable((2,2), PSD=True)

        if 'paq' in query_type:
            loss = cp.sum_squares( thresh - (cp.vec(Sig_hat) @ feat_flat_n) ) / num_meas #+ 0.1 * cp.norm(Sig_hat, 'fro')

        elif query_type == 'triplet':
            loss = cp.sum(cp.pos( thresh - cp.multiply(labels, (cp.vec(Sig_hat) @ feat_flat_n)) ) ) / num_meas #+ 0.1 * cp.norm(Sig_hat, 'fro')

        elif query_type == 'paired_comparison':
            loss = cp.sum(cp.pos( thresh - cp.multiply(labels, (cp.vec(Sig_hat) @ feat_flat_n))) ) / num_meas #+ 0.1 * cp.norm(Sig_hat, 'fro')


        obj = cp.Minimize(loss)
        prob = cp.Problem(obj)

        prob.solve(solver=cp.SCS, max_iters = max_iter)
        ct = 1
        while prob.status != cp.OPTIMAL and ct < len(solvers):
            prob.solve(solver=solvers[ct], max_iters = max_iter)
            ct += 1
        
        #print(f'Tried {ct + 1} solvers, status {prob.status}')
        
        est = Sig_hat.value
        est_out[n] = est
    
    return est_out


# step 2: compute cones
def compute_cones(est_out: dict, gt: dict, num_meas: int, cheat: bool = False, const: float = 10) -> tuple[list, list]:
    alphas = []
    major_axes = []

    for n, sigma_hat in est_out.items():
        # compute svd of Sig_hat
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_hat)
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        lambda_1_hat_n, lambda_2_hat_n = eigenvalues[0], eigenvalues[1]
        u1_hat, u2_hat = eigenvectors[:, 0], eigenvectors[:, 1]
        
        if cheat:
            tau = np.linalg.norm(sigma_hat - gt['sigma_stars'][n], 2)
        else:
            tau = const * 2 / num_meas

        alpha = 2 * tau / np.abs(lambda_1_hat_n - lambda_2_hat_n)
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
def estimate_copunctal(alphas: dict, major_axes: dict, refs: list, w_star: np.ndarray = None) -> tuple[np.ndarray, str]:
    w = cp.Variable(2) # Decision variable for the copunctal point w
    constraints = []
    # Add constraints for each reference point and its corresponding cone
    for zn, u2_hat, alpha in zip(refs, major_axes, alphas):
        # Unpack the eigenvalue vector for readability
        u21_hat, u22_hat = u2_hat[0], u2_hat[1]

        cos_alpha_2 = np.cos(alpha / 2)
        sin_alpha_2 = np.sin(alpha / 2)

        if w_star is not None and not feas_check(w_star, zn, u2_hat, alpha):
            continue

        # Upper side of the cone
        constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat - sin_alpha_2 * u22_hat) <= (w[0] - zn[0]) * (sin_alpha_2 * u21_hat + cos_alpha_2 * u22_hat))
        # Lower side of the cone
        constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat + sin_alpha_2 * u22_hat) >= (w[0] - zn[0]) * (-u21_hat * sin_alpha_2 + u22_hat * cos_alpha_2))
    objective = cp.Minimize(0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    ct = 0
    while problem.status != cp.OPTIMAL and ct < len(solvers):
        problem.solve(solver=solvers[ct])
        ct += 1

    return w.value, problem.status


MAX_RETRIES = 50
TRIALS = 20

num_ref = 25
eigenvalue_gap = 10
query_type = 'triplet'
num_meas = 3
thresh = 5
const_start = 10

for mc in range(TRIALS):
    gt_out = generate_gt(num_ref, eigenvalue_gap)
    meas_out = generate_measurements(num_meas, gt_out, query_type=query_type, thresh=thresh)
    est_out = estimate_metric(meas_out, query_type=query_type, thresh=thresh)

    const = const_start
    w_star = gt_out['w_star']
    angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
    w_hat, status = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star=w_star)

    ct = 0
    while status != cp.OPTIMAL and ct < MAX_RETRIES:
        const += 10
        angles_out, major_axes_out = compute_cones(est_out, gt_out, num_meas, const=const)
        w_hat, status = estimate_copunctal(angles_out, major_axes_out, gt_out['refs'], w_star = w_star)
        ct += 1
    print(const)

    
    print(f'w_hat: {w_hat} | status: {status}')
    print('-------------')