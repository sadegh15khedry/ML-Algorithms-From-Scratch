import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def split_data(x, y, test_size=.2):
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def convert_splitted_data_to_dataframe(X_train_dense, y_train, X_test_dense, y_test):
    train_df = pd.DataFrame(X_train_dense)
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_dense)
    test_df['label'] = y_test
    return train_df, test_df

# # Save the DataFrames to CSV files
# train_df.to_csv('train_dataset.csv', index=False)
# test_df.to_csv('test_dataset.csv', index=False)
  
# def normalize_data(df, method, normalization_columns):
#     if method == 'max_abs':
#         for column in normalization_columns:
#             df[column] = df[column] / df[column].abs().max()
#     elif method == 'min_max':
#         for column in normalization_columns: 
#             df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())     
#     elif method == 'z_score':
#         for column in normalization_columns: 
#             df[column] = (df[column] - df[column].mean()) / df[column].std()
#     elif method == 'robust':
#         for column in normalization_columns:
#             df[column] = (df[column] - df[column].median()) / (df[column].quantile(0.75) - df[column].quantile(0.25))
#     elif method == 'log':
#         for column in normalization_columns:
#             df[column] = np.log1p(df[column])
#     elif method == 'l2':
#         df = df.apply(lambda x: x / np.sqrt(np.sum(np.square(x))), axis=1)
    
#     return df


def encode_labels(df, label):
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label])
    return y


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def stem_words(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def claen_data(df, column):
    df[column] = df[column].apply(remove_punctuation).str.lower().apply(remove_stopwords).apply(stem_words)
    return df

def extract_features(df, text_column):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df[text_column])
    return x