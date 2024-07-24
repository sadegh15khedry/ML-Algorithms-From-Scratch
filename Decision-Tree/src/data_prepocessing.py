import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_name):
    return pd.read_csv('../datasets/'+file_name)

def split_data(df, feature_column, label_column, test_size=.2, random_state=50):
    # x = df.drop(columns= [feature_column])
    # y = df.drop(columns= [label_column])
    return train_test_split(df[feature_column], df[label_column], test_size=test_size, random_state=random_state)

def preprocess_data(df):
    df = df.dropna()
    df.drop_duplicates(keep=False) 
    return df    