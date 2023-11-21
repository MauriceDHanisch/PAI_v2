"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import norm
import random

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

# Set a fixed seed for reproducibility
np.random.seed(0)
random.seed(0)


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # self.gaussian_process_f = GaussianProcessRegressor(
        #     kernel=Matern(length_scale=10.0, nu=2.5)
        # )
        self.gaussian_process_f = GaussianProcessRegressor(
            kernel=RBF(length_scale=10.0)
        )
        self.gaussian_process_v = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
        )
        self.X = np.array([[0]])
        self.Y_f = np.array([[0]])
        self.Y_v = np.array([[0]])
        self.lambda_ = 0.1
        pass

    def current_best(self):
        if self.Y_f.size > 0:
            return np.max(self.Y_f) - self.lambda_ * np.max(self.Y_v)  # Assuming maximization
        else:
            return 0  # Default value if no data is available


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
        x_opt = np.array(self.optimize_acquisition_function())
        return x_opt.reshape(1, -1)

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
        
        f_best = self.current_best()

        # Expected Improvement
        Z_f = (mu_f - f_best) / sigma_f
        ei_f = (mu_f - f_best) * norm.cdf(Z_f) + sigma_f * norm.pdf(Z_f)

        # Penalty for constraint violation
        penalty = np.maximum(0, mu_v - SAFETY_THRESHOLD)
        weighted_penalty = np.exp(-self.lambda_ * penalty)  # Adjust lambda as needed

        # Modified Expected Improvement
        modified_ei = ei_f * weighted_penalty

        return modified_ei

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
        new_point = np.array([[x.squeeze()]])
        new_f = np.array([[f.squeeze()]])
        new_v = np.array([[v]] if np.isscalar(v) else [v.squeeze()])

        # Append the new data point to the existing dataset
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
        print(f'Iteration {j + 1}: x={x}')
        print(f'Domain shape: {DOMAIN.shape[0]}')
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
