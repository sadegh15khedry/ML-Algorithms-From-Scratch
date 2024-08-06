# Linear Regression From The Scratch

## Introduction

This project utilizes a Linear Regressor model that has been built from the scratch to predict housing prices using the Boston Housing. The main reason behind this project was to get a better understanding of how the Linear Regression works.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation Guide](#installation-guide)
- [Acknowledgments](#acknowledgments)
- [Further Improvements](#further-improvements)
- [License](#license)

## Overview

This project is focused on implementing a custom linear regression model and comparing it with the standard scikit-learn linear regression model. The project includes data preprocessing, model training, and evaluation phases using the Boston Housing dataset.

## Directory Structure
```
├── src
│ ├── utils.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── linear_regression.py
│ ├── data_preprocessing.py
│ └── data_exploration.py
├── notebooks
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ ├── model_training.ipynb
│ └── model_evaluation.ipynb
├── models
│ ├── model_custom.joblib
│ └── model_sklearn.joblib
├── environment.yml
└── README.md
```
## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `model_training.py` : Functions for training the model.
- `model_evaluation.py` : Functions for evaluating the model.
- `linear_regression.py` : Custom implementation of linear regression.
- `data_preprocessing.py` : Functions for data preprocessing.
- `data_exploration.py` : Functions for data exploration.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `model_training.ipynb`: Notebook for model training.
- `model_evaluation.ipynb`: Notebook for model evaluation.


### utils.py

Utility functions for various tasks.

- `save_confution_matrix(cm, file_path)`: Saves a confusion matrix as a heatmap.
- `save_report(report, file_path)`: Saves a classification report to a file.
- `save_dataframe_as_csv(df, file_path)`: Saves a pandas DataFrame to a CSV file.
- `save_model(model, path)`: Saves a trained model to a file.
- `load_model(path)`: Loads a model from a file.
- `load_data(path)`: Loads a CSV file into a pandas DataFrame.
- `set_pandas_options()`: Sets display options for pandas.
- `get_error(y_train, y_pred_train)`: Calculates mean squared error.

### model_training.py

Functions for training the model.

- `train_model(x_train, y_train, number_of_iterations, learning_rate, model_type='custom')`: Trains a linear regression model, either custom or sklearn's.

### model_evaluation.py

Functions for evaluating the model.



### linear_regression.py

Custom implementation of linear regression.

- `CustomLinearRegression`: Class for custom linear regression.
  - `__init__(self, learning_rate=0.001, number_of_iterations=1000)`: Initializes the model.
  - `fit(self, x, y)`: Trains the model.
  - `predict(self, x)`: Makes predictions.

### data_preprocessing.py

Functions for data preprocessing.

- `split_data(df, feature_column, label_column, test_size=0.2, random_state=50)`: Splits the data into training and testing sets.
- `normalize_data(df, method, normalization_columns)`: Normalizes the data using various methods.

### data_exploration.py

Functions for data exploration.

- `plot_correlation_matrix(correlation_matrix)`: Plots and saves the  correlation matrix heatmap.

### Notebooks

- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `model_training.ipynb`: Notebook for model training.
- `model_evaluation.ipynb`: Notebook for model evaluation.

## Dataset

The dataset used is the Boston Housing dataset.

## Model Performance


- custom model Training MSE: 74.31886724749972
- custom model Test MSE: 86.51035738530226

- sklearn model Training MSE: 18.361320694695735
- sklearn model Test MSE: 24.969578107874188

## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/ML-Algorithms-From-Scratch.git
    cd ML-Algorithms-From-Scratch/Linear-Regression
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate your-env-name
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```


## Acknowledgments
- This project is based AssemblyAI video on linear regression implementation from scratch. You can use the fallowing link (https://www.youtube.com/watch?v=ltXSoduiVwY) to see their video on this project. 
- Special thanks to the developers and contributors the libraries used in this project, including NumPy, pandas, scikit-learn, Seaborn, and Matplotlib.
- Huge thaks to  contributors of the  boston housing dataset.

## Further Improvements
- Implement cross-validation to improve model performance.
- Explore other regression models for comparison, such as Ridge and Lasso regression.
- Add hyperparameter tuning to optimize the model parameters.
- Enhance data preprocessing techniques, such as handling missing values and feature engineering.
- Extend the evaluation metrics to include R^2 score and MAE (Mean Absolute Error).

  
## License
This project is licensed under the MIT License. See the LICENSE file for details.
