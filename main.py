import os
import shap
import time
import numpy as np
import pandas as pd
from sboa import sboa
from config import initialize, load_data
from functools import partial
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from utils import evaluate_xgb_model, grid_downsample
from sklearn.model_selection import cross_val_score

def xgb_objective(n_estimators, max_depth, learning_rate, X_train, y_train, random_state):

    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    learning_rate = round(learning_rate, 2)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1
    )
    
    neg_score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    # SBOA minimizes the objective function
    return -neg_score


def optimization(pbounds, pop_size, max_iter, X_train, y_train, random_state):

    keys = []
    lower_bounds = []
    upper_bounds = []
    for k, v in pbounds.items():
        keys.append(k)
        lower_bounds.append(v[0])
        upper_bounds.append(v[1])
    
    wrapped_func = partial(xgb_objective, X_train=X_train, y_train=y_train, random_state=random_state)

    start_time = time.perf_counter()
    best_params, optim_curve = sboa(pop_size, max_iter, lower_bounds, upper_bounds, wrapped_func)
    end_time = time.perf_counter()

    _best_params = {}
    for idx, value in enumerate(best_params):
        _best_params[keys[idx]] = value

    minutes = (end_time - start_time) // 60
    seconds = (end_time - start_time) % 60
    print(f"[ SBOA Optimization ] Hyperparameter optimization required {minutes:>8.4f} min {seconds:>8.4f} s")

    return _best_params, optim_curve


if __name__ == "__main__":

    # Initialize parameters: population size, max iterations, parameter bounds, and random seed
    pop_size, max_iter, pbounds, random_state = initialize()


    # Load the dataset, obtaining training set, test set, and feature names
    X_train, X_test, y_train, y_test, feature_names = load_data()
    

    # Perform hyperparameter optimization
    best_params, optim_curve = optimization(
        pbounds=pbounds,
        pop_size=pop_size,
        max_iter=max_iter,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state
    )


    # Format and print the best parameters
    n_estimators = int(best_params["n_estimators"])
    max_depth = int(best_params["max_depth"])
    learning_rate = round(best_params["learning_rate"], 2)
    print(f"[ Best Params ] n_estimators: {n_estimators:>8}, max_depth: {max_depth:>8}, learning_rate: {learning_rate:>8}")


    # Create an output directory for saving results
    os.makedirs("output", exist_ok=True)


    # Plot the optimization convergence curve
    plt.figure(figsize=(6, 4), layout="constrained")
    plt.plot(range(1, max_iter+1), optim_curve)
    plt.title("Optim Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Cross Validation Score")
    optim_curve_df = pd.DataFrame({"optim_curve": optim_curve})
    optim_curve_df.to_csv("output/optim_curve.csv", index=False) # Save the optimization curve data to CSV


    # Train the model using the optimal parameters and evaluate the trained model
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1 # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) # Save prediction result
    df_pred = pd.DataFrame({"y_pred": y_pred})
    df_pred.to_csv("output/y_pred.csv", index=False)
    rmse, mae, r2 = evaluate_xgb_model(model, X_test, y_test) # Save evaluation metrics 
    df_metrics = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2]
    })
    df_metrics.to_csv("output/metrics.csv")
    print(f"[ Evaluate Model ] RMSE: {rmse:>8.4f}, MAE: {mae:>8.4f}, R2: {r2:>8.4f}")


    # Plot XGBoost feature importance bar chart
    plt.figure(figsize=(6, 4), layout="constrained")
    plt.bar(feature_names, model.feature_importances_)
    plt.title("XGBoost Feature Importance")
    plt.ylabel("Feature Importance")
    
    
    # Plot scatter plot of predicted vs. true values
    plt.figure(figsize=(6, 4), layout="constrained")
    points = np.column_stack((y_test, y_pred))
    points = grid_downsample(points) # Reduce overplotting and improve visualization clarity
    plt.scatter(points[:, 0], points[:, 1], color='#8887cb', edgecolors="k")
    plt.title("Scatter Plot")
    plt.xlabel("True Value")
    plt.ylabel("Pred Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(
        "output/scatter_plot.tif",
        format="tiff",           
        dpi=300,            
        bbox_inches='tight',
    )


    # Plot SHAP summary plot
    plt.figure(layout="constrained")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.plots.violin(shap_values, X_train, feature_names, plot_size=(6, 4), color_bar=True, show=False)
    plt.title("Summary Plot")
    plt.savefig(
        "output/summary_plot.tif",
        format="tiff",           
        dpi=300,            
        bbox_inches='tight',
    )


    # Plot SHAP dependence plots for each feature
    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    expl = explainer(X_train)
    for feature_name in feature_names:
        fig = plt.figure(figsize=(6, 4), layout="constrained")
        shap.plots.scatter(expl[:, feature_name], color=expl, ax=fig.gca(), show=False)
        plt.xlabel(feature_name)
        plt.ylabel(f"SHAP Value for {feature_name}")
        plt.savefig(
            f"output/{feature_name}_dependence_plot.tif",
            format="tiff",           
            dpi=300,            
            bbox_inches='tight',
        )

    plt.show()
    

