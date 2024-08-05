# Principal Component Analysis From The Scratch

## Introduction

This project utilizes a Principal Component Analysis that has been built from the scratch to reduce the dimentionalty of the data.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Files and Functions](#files-and-functions)
- [Dataset](#dataset)
- [Installation Guide](#installation-guide)
- [Results](results)
- [Acknowledgments](#acknowledgments)
- [Further Improvements](#further-improvements)
- [License](#license)

## Overview

This project is focused on implementing a custom Principal Component Analysis and comparing it with the standard scikit-learn PCA.

## Directory Structure
```
├── src
│ ├── utils.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── principal_component_analysis.py
│ ├── data_preprocessing.py
│ └── data_exploration.py
├── notebooks
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ └── principal_component_analysis.ipynb
├── models
│ ├── model_custom.joblib
│ └── model_sklearn.joblib
├── environment.yml
└── README.md
```
## Files and Functions

- `utils.py` : Utility functions for various tasks.
- `model_training.py` : Functions for training the model.
- `principal_component_analysis.py` : Custom implementation of PCA.
- `data_preprocessing.py` : Functions for data preprocessing.
- `data_exploration.py` : Functions for data exploration.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `data_exploration.ipynb`: Notebook for data exploration.
- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `principal_component_analysis.ipynb`: Notebook for principal component analysis


## Dataset

The dataset used is the Boston Housing dataset. you can get it using the following link: https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset


## Installation Guide

To set up the project environment, use the `environment.yml` file to create a conda environment.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sadegh15khedry/ML-Algorithms-From-Scratch.git
    cd ML-Algorithms-From-Scratch/Principal-Component-Analysis
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**

    ```bash
    conda activate pca
    ```

4. **Verify the installation:**

    ```bash
    python --version
    ```
## Results 

sklearn PCA:

![sklearn_scatter_plot](https://github.com/user-attachments/assets/02ab8e6e-210e-4d98-b0b4-a70d4a9576a3)


Cusom PCA:

![custom_scatter_plot](https://github.com/user-attachments/assets/225f00d7-d5ff-4dfe-8876-90c87fa089f3)


## Acknowledgments
- This project is based AssemblyAI video on PCA from scratch. You can use the fallowing link (https://www.youtube.com/watch?v=Rjr62b_h7S4&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=8) to see their video on this project. 
- Special thanks to the developers and contributors the libraries used in this project, including NumPy, pandas, scikit-learn, Seaborn, and Matplotlib.
- Huge thaks to  contributors of the  boston housing dataset.

## Further Improvements
- parameters hupertuning for reachin better results.

  
## License
This project is licensed under the MIT License. See the LICENSE file for details.
