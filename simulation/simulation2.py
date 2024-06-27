import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import random

NUM_TRIALS = 20

area_part1 = 0.8 * 1   # Area of (0.2, 1) x (0, 1)
area_part2 = 0.2 * 0.8 # Area of (0, 0.2) x (0.2, 1)

s = 10

avg_distance_list_1 = []
n_refs = 25
# [1, 5, 10, 20, 40, 80, 140, 200, 320, 450, 600, 800]

total_area = area_part1 + area_part2

#tau = np.random.uniform(1e-1, 1, size=n_refs)
tau = 1
tau = tau / s # scale tau
refs = [] # store n ref points 

infeas_num = 0

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

eigenvalue_gaps = np.linspace(1, 20, num=20, endpoint=True) #(1,10)
max_e_gap = eigenvalue_gaps[-1]


def gen_gt(n_refs):
    ############  Part 1: Generate wstar and reference points ############
    w_star = (random.uniform(0, 0.15), random.uniform(0, 0.15))   # Generate the random point w_star
    for _ in range(n_refs):  # generate reference points
        if np.random.rand() < area_part1 / total_area:
            # Sample from Part 1
            x = np.random.uniform(0.2, 1)
            y = np.random.uniform(0, 1)
        else:
            # Sample from Part 2
            x = np.random.uniform(0, 0.2)
            y = np.random.uniform(0.2, 1)
        refs.append((x, y))

    return w_star, refs


def gen_sigma_stars(w_star, refs, eigenvalue_gap, lambda1_min = 20):
    ############  Part 2: Get u2 star, compute u1 star  ############
    u2_stars = [] # store u2 vectors
    u1_stars = [] # store u1 vectors
    sigma_stars = []
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

        lambda1 = random.uniform(lambda1_min, lambda1_min + 10) # randomly generate lambda1
        lambda2 = lambda1 - eigenvalue_gap 
        U = np.column_stack((u1_star, u2_star))
        Lambda = np.diag([lambda1, lambda2])
        sigma_star = U @ Lambda @ U.T

        sigma_stars.append(sigma_star)
        u2_stars.append(u2_star)
        u1_stars.append(u1_star)

    return sigma_stars, u2_stars, u1_stars

    

avg_distances = []
std_distances = []
for eigenvalue_gap in eigenvalue_gaps:
    distances_mc = [] # store distance of every trial
    
    trials = 0
    while trials < NUM_TRIALS:
        ############  Part 3: Compute true metrics  ############
        w_star, refs = gen_gt(n_refs)
        sigma_stars, _, _ = gen_sigma_stars(w_star, refs, eigenvalue_gap, lambda1_min=max_e_gap)

        ###### Part 4: Obtain estimated metrics from small peruturbation #######
        sigma_hats = []
        perturbations = []

        for sigma_star in sigma_stars:
            try:
                perturbation, sigma_hat = generate_psd_perturbation(sigma_star, tau)
                sigma_hats.append(sigma_hat)
                perturbations.append(perturbation)
            except ValueError as e:
                print(e)
                continue


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

        for result in svd_results:
            lambda_1_hat_n = result['eigval_1']
            lambda_2_hat_n = result['eigval_2']
            alpha = (2 * np.pi * tau) / np.abs(lambda_1_hat_n - lambda_2_hat_n)
            alphas.append(alpha)

        ###### Step 3:  Obtain the cone sides and do linear programming #######

        ### Linear programming
        w = cp.Variable(2) # Decision variable for the copunctal point w
        constraints = []
        # Add constraints for each reference point and its corresponding cone
        for zn, u2_hat, alpha in zip(refs, u2_hats, alphas):
            # Unpack the eigenvalue vector for readability
            u21_hat, u22_hat = u2_hat

            if u21_hat > 0 and u22_hat > 0:
                print('Wrong dir')
                u21_hat, u22_hat = -u21_hat, -u22_hat

            cos_alpha_2 = np.cos(alpha / 2)
            sin_alpha_2 = np.sin(alpha / 2)
            # Upper side of the cone
            constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat - sin_alpha_2 * u22_hat) <= (w[0] - zn[0]) * (sin_alpha_2 * u21_hat + cos_alpha_2 * u22_hat))
            # Lower side of the cone
            constraints.append((w[1] - zn[1]) * (cos_alpha_2 * u21_hat + sin_alpha_2 * u22_hat) >= (w[0] - zn[0]) * (-u21_hat * sin_alpha_2 + u22_hat * cos_alpha_2))

        objective = cp.Minimize(0)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, warm_start = False) 

        if problem.status == cp.OPTIMAL:
            distance = np.sqrt(np.sum((w.value - w_star)**2))
            distances_mc.append(distance)
            trials += 1
        else:
            print(f"The problem is {problem.status} and does not have an optimal solution.")
            continue
        

    avg_distances.append(np.average(distances_mc))
    std_distances.append(np.std(distances_mc) / NUM_TRIALS) # std err
    print(f"Infeasible solutions: {infeas_num}/20")
    print(f"Gap = {eigenvalue_gap}: the avg distance between the estimated and true copunctal points over {NUM_TRIALS} trials is: {avg_distances[-1]}")


plt.figure(figsize=(10, 5))
plt.loglog(eigenvalue_gaps, avg_distances, marker='o', markersize = 4, linestyle='-')  # Line plot with log-scale on x-axis

err_plus_std = [avg_distances[i] + std_distances[i] for i in range(len(avg_distances))]
err_minus_std = [avg_distances[i] - std_distances[i] for i in range(len(avg_distances))]
plt.fill_between(eigenvalue_gaps, err_minus_std, err_plus_std, alpha = 0.5)

#plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Eigenvalue gap')
plt.ylabel('Log of averaged estimation error')
plt.title('Estimation error vs. eigenvalue gaps')
plt.grid(True)
#plt.show()
plt.savefig('est_error_vs_eiggap.png', dpi=500, bbox_inches='tight')
