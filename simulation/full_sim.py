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
        Sig_n = gt['sigma_stars'][n]
        if query_type == 'paq':      
            for _ in range(num_meas):
                a_i = np.random.normal(size=(2))
                quad_i = a_i.T @ Sig_n @ a_i
                gamma_i = thresh / quad_i

                responses_n.append( (a_i, gamma_i) )

        elif query_type == 'paired_comp':
            for _ in range(num_meas):
                ref = gt['refs'][n]
                delta_i = np.random.uniform(low=-1, high=1, size=(2))
                y_i = int((delta_i.T @ Sig_n @ delta_i - thresh) > 0)
 

                responses_n.append( (delta_i, y_i))

        elif query_type == 'triplet':
            for _ in range(num_meas):
                deltas_i = np.random.uniform(low=-1, high=1, size = (2,2))
                dist1 = deltas_i[:,0].T @ Sig_n @ deltas_i[:,0]
                dist2 = deltas_i[:,1].T @ Sig_n @ deltas_i[:,1]

                if dist1 < dist2:
                    responses_n.append( (deltas_i, 1) )
                else:
                    responses_n.append( (deltas_i, 0) )

        responses[n] = responses_n
    return responses



n_ref, eigenvalue_gap = 25, 10
gt_out = generate_gt(n_ref, eigenvalue_gap)
responses = generate_measurements(10, gt_out, query_type='triplet')
print(responses)
# step 1: estimate metric

# step 2: compute cones

# step 3: Estimate copunct