# decision Tree From The Scatch


## Introduction

This project utilizes a decision Tree model that has been built from the scratch to classify Iris flowers. The main reason behind this project was to get a better understanding of how the decision tree works.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
  - [utils.py](#utilspy)
  - [model_training.py](#model_trainingpy)
  - [model_evaluation.py](#model_evaluationpy)
  - [linear_regression.py](#linear_regressionpy)
  - [data_preprocessing.py](#data_preprocessingpy)
  - [data_exploration.py](#data_explorationpy)
  - [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation Guide](#installation-guide)
- [Acknowledgments](#acknowledgments)
- [Further Improvements](#further-improvements)
- [License](#license)

## Overview

This project is focused on implementing a custom decision tree model and comparing it with the standard scikit-learn decision tree model. The project includes data exploration, data preprocessing, model training, and evaluation phases using the Spam email Dataset.

## Directory Structure
```
├── src
│ ├── utils.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── decision_tree.py
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

### utils.py

Utility functions for various tasks.



### model_training.py

Functions for training the model.


### model_evaluation.py

Functions for evaluating the model.



### decision_tree.py

Custom implementation of decision tree.


    

### data_preprocessing.py

Functions for data preprocessing.


### data_exploration.py

Functions for data exploration.



### Notebooks

- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `model_training.ipynb`: Notebook for model training.
- `model_evaluation.ipynb`: Notebook for model evaluation.

## Dataset

The dataset used is the Iris Flower Dataset. get the dataset using the fallowing link [kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

## Model Performance

this section will be added

## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/ML-Algorithms-From-Scratch.git
    cd ML-Algorithms-From-Scratch/Decision-Tree
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate decision-tree
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```


## Acknowledgments

- This project is based AssemblyAI video on decision tree implementation from scratch. You can use the fallowing link ([AssemblyAI](https://www.youtube.com/watch?v=NxEHSAfFlK8&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=5) to see their video on this project. 
- Special thanks to the developers and contributors the libraries used in this project, including NumPy, pandas, scikit-learn, Seaborn, and Matplotlib.
- Huge thaks to contributors of the  Spam email Dataset.

## Further Improvements

- Add hyperparameter tuning to optimize the model parameters.


  
## License
This project is licensed under the MIT License. See the LICENSE file for details.


