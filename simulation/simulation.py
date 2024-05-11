import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import random

'''
This method is to generate the peturbation matrix with its operator norm bounded by k and estimated metric as a psd matrix.
It return value error if it fails to find such a perturbation matrix after 100 attempts.
@param: 
    sigma - true metric
    k - operator norm bound
@output: 
    perturbation - 2x2 perturbation matrix 
    sigmahat - estimated metric (2x2 psd matrix)
'''
def generate_psd_perturbation(sigma, k):
    attempts = 0
    while True:    
        perturbation = np.random.randn(2, 2) # Step 1: Generate a random 2x2 matrix from a Gaussian distribution
        perturbation = (perturbation + perturbation.T) / 2 # Step 2: Ensure symmetry by averaging with its transpose
        perturbation = np.dot(perturbation, perturbation.T) # Step 3: Make the perturbation matrix positive semi-definite

        norm = np.linalg.norm(perturbation, ord=2) # compute the operator norm of the perturbation 
        if norm > k:
            perturbation *= (k / norm)

        sigmahat = sigma + perturbation
        if np.all(np.linalg.eigvals(sigmahat) >= 0) and np.allclose(sigmahat, sigmahat.T):
            return perturbation, sigmahat
        attempts += 1
        if attempts >= 100:
            raise ValueError("Failed to generate a suitable perturbation that keeps the matrix PSD")

'''
This method is Simulation Process (Algorithm 1, second step)
@param: 
    n_ref - number of reference points
    eigenvalue_gap - lambda1 - lambda2
    area_part1 - specific area to generate reference points
    area_part2 - specific area to generate reference points
@output: 
    avg_distance - the average distances of ||wstar - what||_2 over 20 trials 
'''
def solve(n_ref, s, eigenvalue_gap, area_part1, area_part2):

    total_area = area_part1 + area_part2
    distance_sum = 0 # compute average distances of ||wstar - what||_2 over 20 trials 
    taus = np.random.uniform(1e-1, 1, size=n_ref)
    taus = taus / s # scale taus
    refs = [] # store n ref points 

    infeas_num = 0

    for i in range(20): # we have 20 trials in total
        ############  Part 1: Generate wstar and reference points ############
        w_star = (random.uniform(0, 0.15), random.uniform(0, 0.15))   # Generate the random point w_star
        for _ in range(n_ref):  # generate reference points
            if np.random.rand() < area_part1 / total_area:
                # Sample from Part 1
                x = np.random.uniform(0.2, 1)
                y = np.random.uniform(0, 1)
            else:
                # Sample from Part 2
                x = np.random.uniform(0, 0.2)
                y = np.random.uniform(0.2, 1)
            refs.append((x, y))

        # fig, ax = plt.subplots()
        # ax.scatter(*zip(*refs),  label='$\mathcal{z}$', color='blue')  
        # ax.add_patch(plt.Rectangle((0,0), 0.15, 0.15, edgecolor='red', fill=None, linestyle='--'))
        # ax.plot(w_star[0], w_star[1], '^', color='red', label='w*', markersize=10)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_title(f'True copunctal point and {n_ref} reference points')
        # fig.savefig(f'refpt_wstar_generation_plot_{i}.png', dpi=500, bbox_inches='tight')
        

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
            lambda1 = random.uniform(10, 16) # randomly generate lambda1
            lambda2 = lambda1 - eigenvalue_gap 
            U = np.column_stack((u1_star, u2_star))
            Lambda = np.diag([lambda1, lambda2])
            sigma_star = U @ Lambda @ U.T
            sigma_stars.append(sigma_star)


        ###### Part 4: Obtain estimated metrics from small peruturbation #######
        sigma_hats = []
        perturbations = []

        for sigma_star, tau in zip(sigma_stars, taus):
            try:
                perturbation, sigma_hat = generate_psd_perturbation(sigma_star, tau)
                sigma_hats.append(sigma_hat)
                perturbations.append(perturbation)
            except ValueError as e:
                print(e)
                continue

            # # Print results
            # print("Perturbed Σ* Matrices and Corresponding Taus:")
            # for i, (sigma_hat, tau_value) in enumerate(zip(sigma_hats, taus)):
            #     print(f"Matrix {i+1} with tau={tau_value:.4f}:\n{sigma_hat}\n")

        #################### Perform our algorithm ######################
        ###### Step 1:  Get eigenvectors from estimated metrics #######
        svd_results = []
        u2_hats = []

        for sigma_hat in sigma_hats:
            # Compute the SVD of the perturbed matrix
            eigenvalues, eigenvectors = np.linalg.eigh(sigma_hat)
            indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]

            eigval_1_hat, eigval_2_hat = eigenvalues[0], eigenvalues[1]
            u1_hat, u2_hat = eigenvectors[:, 0], eigenvectors[:, 1]
            u2_hats.append(u2_hat)

            # Store the SVD results
            svd_results.append({
                'eigval_1': eigval_1_hat,
                'eigval_2': eigval_2_hat,
                'u1_hat': u1_hat,
                'u2_hat': u2_hat
            })

        ###### Step 2:  Computes cone angles (alpha's) #######
        alphas = []

        for result, tau in zip(svd_results, taus):
            lambda_1_hat_n = result['eigval_1']
            lambda_2_hat_n = result['eigval_2']
            alpha = 2 * tau / np.abs(lambda_1_hat_n - lambda_2_hat_n)
            alphas.append(alpha)


        ###### Step 3:  Obtain the cone sides and do linear programming #######

        ### Linear programming
        w = cp.Variable(2) # Decision variable for the copunctal point w
        constraints = []
        # Add constraints for each reference point and its corresponding cone
        for zn, u2_hat, alpha in zip(refs, u2_hats, alphas):
            # Unpack the eigenvalue vector for readability
            u21_hat, u22_hat = u2_hat
            cos_alpha_2 = np.cos(alpha / 2)
            sin_alpha_2 = np.sin(alpha / 2)
            # Upper side of the cone
            constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat - sin_alpha_2 * u22_hat) <= (w[0] - zn[0]) * (sin_alpha_2 * u21_hat + cos_alpha_2 * u22_hat))
            # Lower side of the cone
            constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat + sin_alpha_2 * u22_hat) >= (w[0] - zn[0]) * (-u21_hat * sin_alpha_2 + u22_hat * cos_alpha_2))

        objective = cp.Minimize(0)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except cp.error.SolverError:
            print("Trying with SCS solver.")
            problem.solve(solver=cp.SCS)
        # ? Try CVSOPT, SCS and ECOS as solver. 
        # ECOS performs better in num_refs experiments, but it fails in threshold experiment
        # CVSOPT is better in threshold experiment, but it fails in eigenvalue gap experiment
        # SCS can run all experiments

        # Print the solution
        # print("The estimated copunctal point w is:", w.value)
        # print("The true copunctal point w is:", w_star)
        if problem.status == cp.OPTIMAL:
            distance = np.sqrt(np.sum((w.value - w_star)**2))
        else:
            print(f"The problem is {problem.status} and does not have an optimal solution.")
            distance = 0  
            infeas_num +=1
        distance_sum += distance
        
    avg_distance = distance_sum/20
    print(f"Infeasible solutions: {infeas_num}/20")
    print("The avg distance between the estimated and true copunctal points over 20 trials is: ", avg_distance)
   
    return avg_distance

ss = np.logspace(0, 6, num=100) 
eigenvalue_gaps = np.linspace(0, 10, num=100, endpoint=False) #[0,10)
area_part1 = 0.8 * 1   # Area of (0.2, 1) x (0, 1)
area_part2 = 0.2 * 0.8 # Area of (0, 0.2) x (0.2, 1)


# # different number of reference points 
# print("Test different number of reference points.")

# s = 10
# eigenvalue_gap = 4
# avg_distance_list_1 = []

# # Define the ranges and number of points for each segment
# # More points between 1 and 10
# start_1 = np.log10(1)   # This is 10^0 = 1
# stop_1 = np.log10(10)   # This is 10^1 = 10
# num_points_1 = 20       # Number of points in this segment

# # Fewer points between 10 and 100
# start_2 = np.log10(10)  # This is 10^1 = 10
# stop_2 = np.log10(1000)  # This is 10^2 = 100
# num_points_2 = 20        # Number of points in this segment


# # Generate sample points using logspace for each segment
# points_segment_1 = np.logspace(start_1, stop_1, num=num_points_1, endpoint=False)  # Exclude endpoint to avoid duplication at 10
# points_segment_2 = np.logspace(start_2, stop_2, num=num_points_2)

# # Combine the arrays and sort (although they should already be in order)
# n_refs = np.concatenate((points_segment_1, points_segment_2))
# n_refs = np.unique(n_refs).astype(int)  # Ensure unique values and convert to integer

# n_refs = sorted(set(n_refs))

# for n_ref in n_refs:
#     print(f"The number of reference points is {n_ref}.")
#     avg_distance_list_1.append(solve(n_ref, s, eigenvalue_gap, area_part1, area_part2))
# print(avg_distance_list_1)
# plt.figure(figsize=(10, 5))
# plt.loglog(n_refs, avg_distance_list_1, marker='o', markersize = 4, linestyle='-')  # Line plot with log-scale on x-axis
# plt.legend(fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xlabel('Number of reference points')
# plt.ylabel('Averaged estimation error')
# plt.title('Estimation error vs. number of reference points')
# plt.grid(True)
# plt.savefig('est_error_vs_number_refpoints.png', dpi=500, bbox_inches='tight')

# different thresholds (i.e., operator norm bound for metrics) 
print("Test different thresholds.")

n_ref = 25
eigenvalue_gap = 4
avg_distance_list_2 = []
ss = np.logspace(0, 5, num=50)

for s in ss:
    print(f"The scaling factor is {s}.")
    avg_distance_list_2.append(solve(n_ref, s, eigenvalue_gap, area_part1, area_part2))
    
plt.figure(figsize=(10, 5))
plt.loglog(ss, avg_distance_list_2, marker='o', markersize = 4, linestyle='-')  # Line plot with log-scale on x-axis
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel(r'Thresholds $\sigma$')
plt.ylabel('Averaged estimation error')
plt.title('Estimation error vs. Thresholds')
plt.grid(True)
plt.savefig('est_error_vs_thresholds.png', dpi=500, bbox_inches='tight')


# print("Test different eigenvalue gaps.")

# s = 10
# n_ref = 25
# avg_distance_list_3 = []

# for eigenvalue_gap in eigenvalue_gaps:
#     print(f"The eigenvalue gap is {eigenvalue_gap}.")
#     avg_distance_list_3.append(solve(n_ref, s, eigenvalue_gap, area_part1, area_part2))
    
# plt.figure(figsize=(10, 5))
# plt.semilogy(eigenvalue_gaps, avg_distance_list_3, marker='o', markersize = 4, linestyle='-')  # Line plot with log-scale on x-axis
# print(eigenvalue_gaps)
# print(avg_distance_list_3)
# input()
# plt.legend(fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xlabel('Eigenvalue gap')
# plt.ylabel('Averaged estimation error')
# plt.title('Estimation error vs. eigenvalue gaps')
# plt.grid(True)
# plt.savefig('est_error_vs_eiggap.png', dpi=500, bbox_inches='tight')

