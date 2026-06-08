# SBOA-XGBoost and SHAP

This repository provides the implementation of SBOA-XGBoost with SHAP-based interpretability analysis. The project code can be used for synthetic well log generation or other regression tasks.

## Features

* XGBoost hyperparameter optimization using SBOA (Secretary Bird Optimization Algorithm)
* Model performance evaluation (RMSE, MAE, R²)
* XGBoost feature importance visualization
* SHAP explainability analysis

## Installation

Clone the repository:

```bash
git clone https://github.com/yourname/SBOA-XGBoost.git
cd SBOA-XGBoost
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

> Note: Python 3.14 or higher is recommended.

## Usage

A randomly generated dataset is included in this project for demonstration purposes. After installing the required dependencies, the code can be executed directly:

```bash
python main.py
```

The program will automatically perform hyperparameter optimization, model evaluation, and SHAP-based interpretability analysis.

To use your own dataset, modify the `load_data()` function in `config.py` to return:

```python
X_train, X_test, y_train, y_test, feature_names
```

Optimization settings and hyperparameter search ranges can also be adjusted in `config.py`.

## Outputs

The program generates:

* Optimization convergence curve
* Prediction results and evaluation metrics
* XGBoost feature importance bar plot
* SHAP summary plot
* SHAP dependence plots

The program will automatically create a `output/` directory, all outputs will be saved in the `output/` directory.

## Citation

If you use this code in your research, please cite the corresponding paper.

```bibtex
@article{zeng2026synthetic,
  title={Synthetic well log generation based on SBOA-XGBoost and SHAP},
  author={Zeng, Weijie and Zhang, Zhansong and Guo, Jianhong and Xu, Wenbin and Meng, Sihai and Lv, Hengyang and Yang, Hang},
  journal={Journal of Geophysics and Engineering},
  volume={23},
  number={2},
  pages={353--372},
  year={2026},
  publisher={Oxford University Press}
}
```

## License

MIT License
