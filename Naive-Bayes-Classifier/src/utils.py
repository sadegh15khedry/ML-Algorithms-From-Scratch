import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

def save_confution_matrix(cm, file_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(file_path)
    
    
def save_report(report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
        
def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def save_model(model, path):
    joblib.dump(model, path)
    

def load_model(path):
    loaded_model = joblib.load(path)
    return loaded_model

def load_data(path):
    return pd.read_csv(path)

def set_pandas_options():
    #setting the maximum number of printing columns 
    pd.set_option('display.max_columns', 20)
    # Increase the maximum width of the display
    pd.set_option('display.width', 1000)
    
def get_error(y_train, y_pred_train):
    return mean_squared_error(y_train, y_pred_train)   

def get_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.2f}")
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.2f}")
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.2f}")
    
def convert_dataframe_to_numpy_array(dataframe):
    return np.array(dataframe)  
