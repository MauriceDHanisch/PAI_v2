"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, DotProduct, ConstantKernel
from scipy.stats import norm

# global variables
# only need to edit in here
DOMAIN = np.array([[0, 8]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
BETA = 2 # safety parameter
KAPPA = 50 # exploration parameter
LAMBDA = 10 # weight of constraint violation
LENGTH_F = 1  # length scale of f
LENGTH_V = 1  # length scale of v
NU_F = 2.5  # smoothness of f
NU_V = 2.5  # smoothness of v
SIGMA_F = 0.15  # noise level of f
SIGMA_V = 1e-4  # noise level of v

# MATERN KERNEL
KERNEL_F = Matern(length_scale=LENGTH_F, nu=NU_F) + WhiteKernel(noise_level=SIGMA_F**2)
KERNEL_V = DotProduct(sigma_0=0) + Matern(length_scale=LENGTH_V, nu=NU_V) + WhiteKernel(noise_level=SIGMA_V**2)

# # RBF KERNEL
# KERNEL_F = np.sqrt(2) * RBF(length_scale=LENGTH_F) + WhiteKernel(noise_level=SIGMA_F**2)
# KERNEL_V = 0.5 * RBF(length_scale=LENGTH_V) + WhiteKernel(noise_level=SIGMA_V**2)

# # fix random seed for reproducibility while tuning hyperparameters
np.random.seed(0)

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.gaussian_process_f = GaussianProcessRegressor(kernel=KERNEL_F)
        self.gaussian_process_v = GaussianProcessRegressor(kernel=KERNEL_V)
        self.X = None
        self.Y_f = None
        self.Y_v = None
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        
        x_opt = self.optimize_acquisition_function()

        # # reduce domain to the range of possible x
        # global DOMAIN
        # DOMAIN = np.array([[
        #     np.min(self.X, DOMAIN[:, 0], axis=0),
        #     np.max(self.X, DOMAIN[:, 1])
        # ]])
        # print(f"DOMAIN: {DOMAIN}")

        return np.atleast_2d(x_opt)

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        # Get the predictions from your Gaussian process model
        mu_f, sigma_f = self.gaussian_process_f.predict(x, return_std=True)
        mu_v, sigma_v = self.gaussian_process_v.predict(x, return_std=True)
        

        ######################################
        # # EI method
        # f_best = np.max(self.Y_f)
        # Z_f = (mu_f - f_best) / sigma_f
        # ei_f = (mu_f - f_best) * norm.cdf(Z_f) + sigma_f * norm.pdf(Z_f)

        # # method from hint
        # # Penalty for constraint violation
        # penalty = LAMBDA * np.max(mu_v + sigma_v, 0)
        # af_value = ei_f - penalty
        ######################################

        ######################################
        # # Safe opt method
        # # Probability of being safe, incorporating beta
        # Z_v = (SAFETY_THRESHOLD - mu_v) / np.sqrt(sigma_v**2 + BETA)
        # prob_safe = norm.cdf(Z_v)

        # # Incorporating both objective improvement and safety
        # af_value = ei_f * prob_safe
        ######################################

        ######################################
        # # UCB method
        penalty = LAMBDA * np.maximum(0, mu_v + sigma_v)
        af_value = mu_f + KAPPA * sigma_f - penalty
        ######################################

        return af_value.squeeze()

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        new_point = np.atleast_2d(x)
        new_f = np.atleast_2d(f)
        new_v = np.atleast_2d(v)

        # Append the new data point to the existing dataset
        if self.X is None:
            self.X = new_point
            self.Y_f = new_f
            self.Y_v = new_v
        else:
            self.X = np.vstack([self.X, new_point])
            self.Y_f = np.vstack([self.Y_f, new_f])
            self.Y_v = np.vstack([self.Y_v, new_v])

        # Re-train both models
        self.gaussian_process_f.fit(self.X, self.Y_f)
        self.gaussian_process_v.fit(self.X, self.Y_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        possible_index = np.where(self.Y_v < SAFETY_THRESHOLD)[0]

        if self.X.size > 0:
            best_index = np.argmax(self.Y_f)
            return self.X[best_index].item()
        else:
            raise ValueError("No data points available")

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    print(f'Initial safe point: {x_init}')
    obj_val = f(x_init)
    print(f'Initial objective value: {obj_val}')
    cost_val = v(x_init)
    print(f'Initial constraint value: {cost_val}')
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
