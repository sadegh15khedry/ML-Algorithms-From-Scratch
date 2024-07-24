# Random Forest Regressor for Boston Housing Dataset

## Introduction

This project utilizes a Random Forest Regressor model to predict housing prices using the Boston Housing dataset. It includes various components such as data preprocessing, model training, evaluation, and utility functions for saving and loading models and data.

## Table of Contents

- [Random Forest Regressor](#random-forest-regressor)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Folder Structure](#folder-structure)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Utils](#utils)
- [Contributing](#contributing)
- [License](#license)

## Random Forest Regressor

The Random Forest Regressor is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees for regression tasks. Here’s a brief overview of its functioning:

1. **Ensemble Method**: It combines multiple decision trees to improve generalizability and robustness over a single decision tree model.

2. **Tree Construction**: Each tree is built using a subset of the training data and a random selection of features. This randomness helps to reduce overfitting.

3. **Prediction**: For regression tasks, predictions are made by averaging the predictions of all the individual trees in the forest.

4. **Hyperparameters**: Important hyperparameters include the number of trees (n_estimators), maximum depth of each tree (max_depth), and the number of features considered for splitting at each node (max_features).

Random Forests are widely used due to their ability to handle large datasets with high dimensionality and noisy data, while also providing good accuracy and robustness.

For more details on the implementation and parameters, refer to the `model_training.ipynb` notebook and the scikit-learn documentation on [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).


## Installation

Ensure Python 3.x is installed along with the dependencies listed in `requirements.txt`. Install them using pip:
```bash
pip install -r requirements.txt
```

## Usage

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/sadegh15khedry/Housing-Prices-Prediction-Using-RandomForest.git
cd random-forest-boston-housing
```

Run the Jupyter notebooks for different aspects of the project:

- data_exploration.ipynb: Explore the dataset and visualize correlations.
- data_preprocessing.ipynb: Preprocess the dataset by removing duplicates and null values, and split into training and test sets.
- model_training.ipynb: Train the Random Forest Regressor model.
- model_evaluation.ipynb: Evaluate the trained model and calculate Mean Squared Error (MSE).

## Dataset

The Boston Housing dataset contains various factors that might influence housing prices in Boston suburbs. Features include crime rate, property tax rate, and accessibility to highways. The target variable is the median value of owner-occupied homes (MEDV).

## Folder Structure

The project follows a standard folder structure convention:

- **datasets/**: Contains dataset files.
- **models/**: Stores trained machine learning models.
- **notebooks/**: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **src/**: Source code directory containing Python scripts for data processing, model training, evaluation, and utility functions.


## Data Exploration

Explore the dataset to understand its structure and statistical summaries:
```code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Load dataset
df = pd.read_csv('../datasets/housing_prices_boston.csv')

Display information about columns
df.info()

Describe statistical summary
df.describe()

Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()
```

## Data Preprocessing

Prepare the data by preprocessing steps such as removing duplicates and null values, and splitting into training and test sets:
```code
from data_prepocessing import load_data, split_data, preprocess_data
from utils import save_dataframe_as_csv

#Load dataset
df = load_data("housing_prices_boston.csv")

#Preprocess data
df = preprocess_data(df)

#Select feature columns
feature_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

#Select label column
label_column = 'MEDV'

#Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

#Split data into training and test sets
x_train, x_test, y_train, y_test = split_data(df, feature_columns, label_column)

#Save dataframes as CSV
save_dataframe_as_csv(x_train, "../datasets/x_train.csv")
save_dataframe_as_csv(y_train, "../datasets/y_train.csv")
save_dataframe_as_csv(x_test, "../datasets/x_test.csv")
save_dataframe_as_csv(y_test, "../datasets/y_test.csv")
```

## Model Training

Train the Random Forest Regressor model:

```code
from sklearn.ensemble import RandomForestRegressor
from model_training import train_model

#Train model
model = train_model(x_train, y_train, estimators=100)

#Save trained model
from utils import save_model
save_model(model, '../models/random_forest.joblib')
```

## Model Evaluation

Evaluate the trained model using Mean Squared Error (MSE):

```code
from sklearn.metrics import mean_squared_error
from model_evaluation import evaluate_model
from data_prepocessing import load_data

#Load trained model
model = load_model('../models/random_forest.joblib')

#Load test data
x_test = load_data("x_test.csv")
y_test = load_data("y_test.csv")

#Evaluate model
mse_test = evaluate_model(model, x_test, y_test)

print(f"Test MSE: {mse_test}")
```

## Utils

Utility functions for saving and loading dataframes, models, confusion matrices, and reports:

```code
from utils import save_confusion_matrix, save_report, save_dataframe_as_csv, save_model, load_model

#Example: Save confusion matrix
save_confusion_matrix(cm, "confusion_matrix.png")

#Example: Save classification report
save_report(report, "classification_report.txt")

#Example: Save dataframe as CSV
save_dataframe_as_csv(df, "data.csv")

#Example: Save trained model
save_model(model, "model.joblib")

#Example: Load trained model
loaded_model = load_model("model.joblib")
```

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
