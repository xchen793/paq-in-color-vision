import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import random

#from typing import Callable
from collections import defaultdict

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
        if query_type == 'paq':      
            for i in range(num_meas):
                a_i = np.random.normal(size=(2))
                quad_i = a_i.T @ Sig_n @ a_i
                gamma_i = thresh / quad_i

                feat_matrix_n[:,:,i] = gamma_i * np.outer(a_i, a_i)
                responses_n.append( gamma_i )

        elif query_type == 'paired_comp':
            for i in range(num_meas):
                ref = gt['refs'][n]
                delta_i = np.random.uniform(low=-1, high=1, size=(2))
                y_i = int((delta_i.T @ Sig_n @ delta_i - thresh) > 0)
 
                feat_matrix_n[:,:,i] = np.outer(delta_i[0] - delta_i[1], delta_i[0] - delta_i[1])
                responses_n.append( y_i )

        elif query_type == 'triplet':
            for i in range(num_meas):
                deltas_i = np.random.uniform(low=-1, high=1, size = (2,2))
                dist1 = deltas_i[:,0].T @ Sig_n @ deltas_i[:,0]
                dist2 = deltas_i[:,1].T @ Sig_n @ deltas_i[:,1]
                
                feat_matrix_n[:,:,i] = 2*np.outer(deltas_i[0], deltas_i[2] - deltas_i[1]) + np.outer(deltas_i[1], deltas_i[1]) - np.outer(deltas_i[2], deltas_i[2])
                if dist1 < dist2:
                    responses_n.append( 1 )
                else:
                    responses_n.append( 0 )

        responses[n] = [feat_matrix_n, responses_n]
    return responses


# step 1: estimate metric
def estimate_metric(responses: dict, thresh: float = 1, query_type: str = 'paq', max_iter: int = 10000) -> dict:
    est_out = {}
    n_ref = len(responses)
    for n in range(n_ref):
        if query_type == 'paq':
            feat_matrix_n, y = responses[n]
            feat_flat_n = feat_matrix_n.reshape(4, -1)

            Sig_hat = cp.Variable((2,2), PSD=True)

            loss = cp.sum_squares( thresh - (cp.vec(Sig_hat) @ feat_flat_n) )
            obj = cp.Minimize(loss)
            prob = cp.Problem(obj)

            prob.solve(solver=cp.SCS, max_iters = max_iter)

            est = Sig_hat.value

        # elif query_type == 'paired_comp':
        #     est = estimate_metric_pc(responses, thresh)
        # elif query_type == 'triplet':
        #     est = estimate_metric_triplet(responses)
        est_out[n] = est
    
    return est_out


# step 2: compute cones

# step 3: Estimate copunct



# gt_out = generate_gt(2, 10)
# meas_out = generate_measurements(4, gt_out)

# feat_matrix_n, _ = meas_out[0]
# feat_flat_n = feat_matrix_n.reshape(4, -1)

# est_out = estimate_metric(meas_out)

# print(est_out[0])
# print(gt_out['sigma_stars'][0])