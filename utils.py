import random
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def evaluate_xgb_model(model, X_test, y_test):
    """Evaluate an XGBoost model on the provided test dataset.

    Args:
        model: A trained XGBoost regressor used for prediction.
        X_test (array-like): The input feature matrix for the test set.
        y_test (array-like): The true target values corresponding to X_test.

    Returns:
        tuple: A tuple containing three evaluation metrics:
            - rmse (float): Root Mean Squared Error.
            - mae (float): Mean Absolute Error.
            - r2 (float): R-squared score.
    """
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rmse, mae, r2


def grid_downsample(points, grid_size=10, sample_ratio=0.5, random_state=0):
    """Downsamples 2D points using a spatial grid strategy.

    Divides the data space into a grid and randomly samples a specified 
    ratio of points from each non-empty cell to preserve spatial distribution.

    Args:
        points (np.ndarray): Input 2D data points with shape (N, 2).
        grid_size (int, optional): Number of grid divisions per dimension. 
            Defaults to 10.
        sample_ratio (float, optional): Fraction of points to sample from 
            each grid cell. Must be in the range (0, 1]. Defaults to 0.5.
        random_state (int, optional): Random seed for reproducibility. 
            Defaults to 0.

    Returns:
        np.ndarray: Downsampled 2D data points with shape (M, 2) and dtype float32.
    """
    
    # Set random seed to ensure reproducibility
    if random_state is not None:
        random.seed(random_state)
        
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Initialize a dictionary to store points in each grid cell
    grid = {}
    
    # 1. Assign data points to their corresponding grid cells
    for point in points:
        x, y = point
        # Calculate grid coordinates; prevent division by zero and out-of-bounds errors
        grid_x = int((x - min_x) / width * grid_size) if width > 0 else 0
        grid_y = int((y - min_y) / height * grid_size) if height > 0 else 0
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)
        
        key = (grid_x, grid_y)
        if key not in grid:
            grid[key] = []
        grid[key].append(point.tolist())
        
    # 2. Randomly sample points from each cell based on the specified ratio
    sampled_points = []
    for cell_points in grid.values():
        if not cell_points:
            continue
        # Calculate the number of points to sample; keep at least 1 if the cell is non-empty
        sample_count = max(1, int(len(cell_points) * sample_ratio))
        # If the requested count exceeds actual points, take all available points
        sample_count = min(sample_count, len(cell_points))
        
        # Perform random sampling
        sampled = random.sample(cell_points, sample_count)
        sampled_points.extend(sampled)
        
    return np.array(sampled_points, dtype=np.float32)


def make_regression_data(n_samples=10000, random_state=0):
    """Generates a synthetic regression dataset with complex non-linear relationships.

    Args:
        n_samples (int, optional): The number of samples to generate. Defaults to 1000.
        random_state (int, optional): The random seed used to fix the state of the 
            random number generator to ensure reproducibility. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Input feature matrix with shape (n_samples, 5).
            - y (np.ndarray): Target variable vector with shape (n_samples,).
    """
    rng = np.random.RandomState(random_state)
    
    # Configurations for input feature ranges
    feature_ranges = [
        (0, 100),           # Feature 1: Similar to age or percentage
        (-50, 50),          # Feature 2: Similar to temperature difference
        (1000, 50000),      # Feature 3: Similar to amount/income
        (0, 1),             # Feature 4: Similar to probability or ratio
        (10, 200),          # Feature 5: Similar to length or weight
    ]
    
    # Dimension and noise configurations
    n_features = 5          # Number of input features
    noise_level = 0.3       # Standard deviation of Gaussian noise
    
    # Generate raw feature data based on specified ranges
    X = np.zeros((n_samples, n_features))
    for i, (min_val, max_val) in enumerate(feature_ranges):
        X[:, i] = rng.uniform(min_val, max_val, size=n_samples)
        
    # Normalize to prevent target variable overflow caused by extreme values
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    # Base linear response
    linear_weights = rng.uniform(-2, 2, size=n_features)
    y = np.dot(X_normalized, linear_weights)
    
    # Overlay complex non-linear relationships and feature interactions
    y += 0.5 * np.sum(X_normalized[:, :3] ** 2, axis=1)       # Squared term effects for the first 3 features
    y += 1.5 * X_normalized[:, 0] * X_normalized[:, 1]        # Cross-multiplication between Feature 1 and Feature 2
    y += 2.0 * np.sin(X_normalized[:, 2])                     # Periodic fluctuation of Feature 3
    y += 0.1 * np.exp(X_normalized[:, 3] / 2)                 # Exponential transformation of Feature 4
    y += 3.0 * (X_normalized[:, 4] > 0.5).astype(float)       # Threshold trigger effect of Feature 5
            
    # Add bias term and Gaussian noise
    bias = 5.0
    noise = rng.normal(0, noise_level, size=n_samples)
    y = y + bias + noise
    
    return X, y