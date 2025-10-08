
from scipy.linalg import fractional_matrix_power

def cum_matrix(A):
# input: MxN; output: Mx(N-1)
    M, N = A.shape
    B = np.zeros((M, N - 1))
    B[:, -1] = A[:, -1]  # last column in B
    for j in range(N - 3, -1, -1):
        B[:, j] = A[:, j+1] + B[:, j + 1]
    return B

def unravel(A):
# input: MxN; output: Mx(N+1)
    M, N = A.shape
    B = np.zeros((M, N + 1))
    B[:, 0] = 1 - A[:, 0]  # first column in B
    B[:, -1] = A[:, -1]
    for j in range(N - 1, 0, -1):
        B[:, j] = A[:, j-1] - np.sum(B[:, j+1:N+1], axis=1)
    return B


def compute_EndCQS(transition_cube, random_numbers, StartCQS_vector):
    """
    written by NK; June 2025
    
    this code takes as inputs S simulated cummulative transition matrices and S x b random numbers (where b = # bonds at start), as well as the starting CQS of the b bonds. 
    it outputs the end_CQS matrix for all S simulations accordingly. 

    transition_cube: shape (1000, 7, 7) – 1000 simulation matrices
    random_numbers: shape (1000, 100) – one random number per bond per simulation
    StartCQS_vector: shape (100,) – initial CQS per bond (value 0 to 6)
    
    Returns:
        EndCQS_array: shape (1000, 100) – new CQS values (0–6 or 999 for default)
    """
    sims, M, N = transition_cube.shape  # sims = 1000, M = N = 7
    bonds = len(StartCQS_vector)

    # Allocate result
    EndCQS_array = np.full((sims, bonds), 999)  # default = 999

    for b in range(bonds):
        cqs = StartCQS_vector[b]
        t_matrix = transition_cube[:, cqs, :]  # (1000, 7)
        rand_vals = random_numbers[:, b]       # (1000,)

        # Compare random values against thresholds
        mask = rand_vals[:, None] > t_matrix   # (1000, 7), boolean

        # Get the first True from left (i.e. highest index where rand > threshold)
        new_cqs = np.argmax(mask, axis=1)

        # If no True found, bond defaults → keep 999
        has_movement = mask.any(axis=1)
        EndCQS_array[has_movement, b] = new_cqs[has_movement]

    return EndCQS_array



import numpy as np
from scipy.optimize import minimize

def reconstruct_migration_matrix(K):
    """
    Reconstructs full 8x8 migration matrix M from K (30x7),
    assuming last column of M is known and last row is absorbing.
    """
    n_states = 8
    n_non_default = 7
    n_steps = K.shape[0]

    # Extract known default column
    default_col = np.append(K[0], 1.0)

    # Objective: minimize error between M^t[:, -1] and K[t-1]
    def objective(flat_M):
        M = flat_M.reshape((n_states, n_states))
        M[:, -1] = default_col  # enforce known default column
        M[-1, :] = np.zeros(n_states)
        M[-1, -1] = 1.0  # absorbing state

        loss = 0.0
        Mt = np.eye(n_states)
        for t in range(1, n_steps + 1):
            Mt = Mt @ M
            pred = Mt[:n_non_default, -1]
            loss += np.sum((pred - K[t - 1])**2)
        return loss

    # Initial guess: uniform transitions excluding default
    M0 = np.ones((n_states, n_states)) / n_states
    M0[:, -1] = default_col
    M0[-1, :] = np.zeros(n_states)
    M0[-1, -1] = 1.0

    # Constraints: rows sum to 1 (excluding last row), non-negative entries
    constraints = [
        {'type': 'eq', 'fun': lambda x, i=i: np.sum(x.reshape((n_states, n_states))[i, :]) - 1}
        for i in range(n_non_default)
    ]
    bounds = [(0.0, 1.0)] * (n_states * n_states)

    result = minimize(objective, M0.flatten(), method='SLSQP',
                      bounds=bounds, constraints=constraints)

    M_est = result.x.reshape((n_states, n_states))
    M_est[:, -1] = default_col
    M_est[-1, :] = np.zeros(n_states)
    M_est[-1, -1] = 1.0

    return M_est

# Example usage:
# K = np.array(...)  # shape (30, 7)
# M = reconstruct_migration_matrix(K)


def matpower(P,n):
    if float(n).is_integer():
        return np.linalg.matrix_power(P, int(n))
    else:
        return fractional_matrix_power(P, n)
