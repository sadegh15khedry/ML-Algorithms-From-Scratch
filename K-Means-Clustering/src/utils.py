import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
    
def load_dataset(datset_path):
    df = pd.read_csv(datset_path)
    #setting the maximum number of printing columns 
    pd.set_option('display.max_columns', 20)
    # Increase the maximum width of the display
    pd.set_option('display.width', 1000)
    return df
    
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

 
def save_report(report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
        
def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def save_model(model, path):
    model.save(path)

    
def display_plot(hist, title, color, xlabel, ylabel):
    plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def display_scatter(x_data, y_data, labels, title, xlabel, ylabel):
    plt.scatter(x_data,y_data, c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def display_correlation_matrix_heatmap(title,correlation_matrix):
    #visualizing the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()


def save_datest(df, save_path):
    df.to_csv(save_path, index=False)
          
          
