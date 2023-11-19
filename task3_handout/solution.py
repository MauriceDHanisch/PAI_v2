"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, WhiteKernel
from scipy.stats import norm
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.data = []
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.15**2))
        self.gp_v = GaussianProcessRegressor(
            kernel=(DotProduct(sigma_0=0) + Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=1e-8)))
        self.X_sample = None
        self.Y_sample = None
        self.V_sample = None

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

        # raise NotImplementedError

        # Optimize the acquisition function
        x_opt = self.optimize_acquisition_function()

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
            return -self.acquisition_function(x)[0]

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
            shape (N, 1)nh
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        #  Predict the mean and standard deviation of each point x
        # Expected Improvement for f
        mu, sigma = self.gp.predict(x, return_std=True)
        mu_sample_opt = np.max(self.Y_sample)
        imp = mu - mu_sample_opt
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        # Probability of satisfying the constraint
        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)
        prob_feasibility = norm.cdf((SAFETY_THRESHOLD - mu_v) / sigma_v)

        # Adjust EI by the probability of feasibility
        af_value = ei * prob_feasibility
        # print(af_value.shape)

        return af_value.reshape(-1, 1)

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
        # raise NotImplementedError
        # Convert x, f, and v to np.ndarray and reshape for consistency
        x = np.atleast_2d(x)
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)

        # Update the sample datasets
        if self.X_sample is None:
            self.X_sample = x
            self.Y_sample = f
            self.V_sample = v
        else:
            self.X_sample = np.vstack([self.X_sample, x])
            self.Y_sample = np.vstack([self.Y_sample, f])
            self.V_sample = np.vstack([self.V_sample, v])

        # Retrain the Gaussian Process models
        self.gp.fit(self.X_sample, self.Y_sample)
        self.gp_v.fit(self.X_sample, self.V_sample)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # Filter the samples that satisfy the constraint
        feasible_indices = np.where(self.V_sample < SAFETY_THRESHOLD)[0]
        feasible_X = self.X_sample[feasible_indices]
        feasible_Y = self.Y_sample[feasible_indices]

        if len(feasible_Y) == 0:
            raise ValueError(
                "No feasible solution found under the given constraint.")

        # Find the index of the maximum value in feasible_Y
        max_index = np.argmax(feasible_Y)

        # The optimal solution is the corresponding x value
        solution = feasible_X[max_index].item()

        return solution

        # raise NotImplementedError

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
    obj_val = f(x_init)
    cost_val = v(x_init)
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
