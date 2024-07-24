import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Heatmap')
    plt.show()