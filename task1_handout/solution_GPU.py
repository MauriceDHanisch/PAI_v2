import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import Nystroem
import matplotlib.pyplot as plt
from matplotlib import cm
# os.chdir('C:\\Users\\MOUms\\VS Projects\\PAI_v2\\task1_handout')

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

import torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x_tensor, train_y_tensor, likelihood):
        super(GPRegressionModel, self).__init__(train_x_tensor, train_y_tensor, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self, train_x, train_y):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        #self.gp = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = GPRegressionModel(train_x, train_y, self.likelihood)

        # TODO: Add custom initialization for your model here if necessary

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Set device and model to evaluation mode
        device = torch.device("mps")
        self.gp.eval()
        self.gp.likelihood.eval()

        # Convert test data to PyTorch tensors and move to device
        test_x_tensor = torch.tensor(test_x_2D, dtype=torch.float32).to(device)
        test_x_AREA_tensor = torch.tensor(test_x_AREA, dtype=torch.bool).to(device)

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.gp.likelihood(self.gp(test_x_tensor))
            gp_mean = observed_pred.mean
            gp_std = observed_pred.stddev

        # Convert predictions back to numpy
        gp_mean = gp_mean.cpu().numpy()
        gp_std = gp_std.cpu().numpy()

        # Apply adjustment based on test_x_AREA
        adjustment = np.where(test_x_AREA, gp_std, 0)
        predictions = gp_mean + adjustment

        return predictions, gp_mean, gp_std

    

    def fitting_model(self, train_y_tensor, train_x_tensor):
        
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

        device = torch.device("mps")
        print(f"Using device: {device}")

        self.gp.to(device)
        self.likelihood.to(device)

        self.gp.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        
        n_epochs = 50

        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.gp(data)
                loss = -mll(output, target)
                loss.backward()
                print(f"Iter {epoch + 1}/{n_epochs} - Loss: {loss.item()}")
                optimizer.step()



# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [
        bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function


def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                        [0.79915856, 0.46147936, 0.1567626],
                        [0.26455561, 0.77423369, 0.10298338],
                        [0.6976312,  0.06022547, 0.04015634],
                        [0.31542835, 0.36371077, 0.17985623],
                        [0.15896958, 0.11037514, 0.07244247],
                        [0.82099323, 0.09710128, 0.08136552],
                        [0.41426299, 0.0641475,  0.04442035],
                        [0.09394051, 0.5759465,  0.08729856],
                        [0.84640867, 0.69947928, 0.04568374],
                        [0.23789282, 0.934214,   0.04039037],
                        [0.82076712, 0.90884372, 0.07434012],
                        [0.09961493, 0.94530153, 0.04755969],
                        [0.88172021, 0.2724369,  0.04483477],
                        [0.9425836,  0.6339977,  0.04979664]])

    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i, coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any(
            [is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA


# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1,
                    num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1,
                    num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack(
        (grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(
        visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(
        gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = train_x[:, :2]
    train_x_AREA = train_x[:, 2].astype(bool)
    test_x_2D = test_x[:, :2]
    test_x_AREA = test_x[:, 2].astype(bool)

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

# you don't have to change this function


def main():
    # Load the training dateset and test features
    print('Loading data')
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Take a random subset of the training data
    print('Taking a random subset of the training data')
    percentage = 1
    random_indices = np.random.choice(train_y.shape[0], int(
        percentage/100 * train_y.shape[0]), replace=False)
    train_x = train_x[random_indices]
    train_y = train_y[random_indices]

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(
        train_x, test_x)
    
    train_x_tensor = torch.tensor(train_x_2D, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

    # Fit the model
    print('Fitting model')
    model = Model(train_x_tensor, train_y_tensor)
    model.fitting_model(train_y_tensor, train_x_tensor)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
