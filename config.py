from sklearn.model_selection import train_test_split
from utils import make_regression_data
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    """Load data."""

    # Randomly generated dataset for testing the program
    feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]
    X, y = make_regression_data(n_samples=60000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    return X_train, X_test, y_train, y_test, feature_names


def initialize():
    """Initialize global parameters and plot configurations."""
    
    # Set random seed for reproducibility
    random_state = 0

    # SBOA initialization parameters
    pop_size = 30
    max_iter = 20

    # The hyperparameters to be optimized and their corresponding search spaces
    pbounds = {
        "n_estimators": [1, 300],
        "max_depth": [1, 10],
        "learning_rate": [0.01, 1],
    }

    # Configure matplotlib to use Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # Set font sizes for axes titles, labels, and ticks
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    return pop_size, max_iter, pbounds, random_state