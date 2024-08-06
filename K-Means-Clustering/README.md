# K-Means Clustering From The Scratch

## Introduction

This project utilizes a K-Means Clustering that has been built from the scratch to cluster boston houses. The main reason behind this project was to get a better understanding of how the K-Means Clustering works.

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

This project is focused on implementing a custom K-Means Clustering model and comparing it with the standard scikit-learn K-Means Clustering model. The project includes data exploration, data preprocessing and data clustering.

## Directory Structure
```
├── src
│ ├── utils.py
│ ├── data_clustering.py
│ ├── k_means.py
│ ├── data_preprocessing.py
│ └── data_exploration.py
├── notebooks
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ └── data_clustering.ipynb
├── models
│ ├── model_custom.joblib
│ └── model_sklearn.joblib
├── environment.yml
└── README.md
```
## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `data_clustering.py` : Functions for clustering data.
- `k_means.py` : Custom implementation of K-Means Clustering.
- `data_preprocessing.py` : Functions for data preprocessing.
- `data_exploration.py` : Functions for data exploration.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `data_clustering.ipynb`: Notebook for clustering data


## Dataset

The dataset used is the Boston Housing dataset. you can get the dataset using the following link:
https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset

## Results

This section will be added.

## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/ML-Algorithms-From-Scratch.git
    cd ML-Algorithms-From-Scratch/K-Means-Clustering
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate k-means
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```


## Acknowledgments
- This project is based AssemblyAI video on K-Means Clustering implementation from scratch. You can use the fallowing link ([https://www.youtube.com/watch?v=ltXSoduiVwY](https://www.youtube.com/watch?v=6UF5Ysk_2gk&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=11)) to see their video on this project. 
- Special thanks to the developers and contributors the libraries used in this project, including NumPy, pandas, scikit-learn, Seaborn, and Matplotlib.
- Huge thaks to  contributors of the  boston housing dataset.

## Further Improvements
- Add hyperparameter tuning to optimize the model parameters.


  
## License
This project is licensed under the MIT License. See the LICENSE file for details.
