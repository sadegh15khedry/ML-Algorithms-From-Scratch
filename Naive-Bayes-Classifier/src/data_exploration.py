import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textblob import TextBlob

# def plot_correlation_matrix(correlation_matrix):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#     plt.title('Correlation Matrix Heatmap')
#     plt.savefig('../results/correlation_matrix.png')
#     plt.show()
    
def display_text_size(df):
    # Histogram of text length
    plt.hist(df['text_length'], bins=50)
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Length')
    plt.savefig('../results/text_length.png')
    plt.show()
    
def display_target_values(df, target):
    # Bar plot of target classes
    df[target].value_counts().plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Classes')
    plt.savefig('../results/target_values.png')
    plt.show()

def vocab_analysis(df, column):    
    # Download stopwords
    nltk.download('stopwords')

    # Function to preprocess and tokenize text
    def preprocess(text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return tokens

    # Apply preprocessing and flatten the list of tokens
    all_tokens = df[column].apply(preprocess).sum()

    # Get the most common words
    word_freq = Counter(all_tokens)
    print(word_freq.most_common(20))

    # Bar plot of the most common words
    common_words = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
    common_words.plot(kind='bar', x='word', y='count', legend=False)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Most Common Words')
    plt.savefig('../results/vocal_analysis.png')
    plt.show()
    
    return all_tokens
    
def n_gram_analysis(df, text_column, n=2):
    # Initialize the CountVectorizer for n-grams
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(df[text_column])
    
    # Get the feature names (n-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum up the counts of each n-gram
    n_gram_counts = X.sum(axis=0).A1
    n_gram_freq = pd.DataFrame({'ngram': feature_names, 'count': n_gram_counts})
    
    # Sort the DataFrame by count
    n_gram_freq = n_gram_freq.sort_values(by='count', ascending=False).head(20)
    
    # Plot the top n-grams
    n_gram_freq.plot(kind='bar', x='ngram', y='count', legend=False)
    plt.xlabel(f'{n}-gram')
    plt.ylabel('Frequency')
    plt.title(f'Most Common {n}-grams')
    plt.savefig('../results/ngram.png')
    plt.show()
    
    
    
def get_word_cloud(all_tokens):

    # Generate word cloud
    wordcloud = WordCloud(stopwords=stopwords.words('english'), background_color='white').generate(' '.join(all_tokens))

    # Display the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig('../results/wordcloud.png')
    plt.show()
 
def get_polarity(text):
    return TextBlob(text).sentiment.polarity 
  
def explore_sentiment(df, text_column):
    # Apply the function to get polarity scores
    df['polarity'] = df[text_column].apply(get_polarity)
    
    # Summary statistics of polarity
    print(df['polarity'].describe())
    
    # Histogram of polarity
    plt.hist(df['polarity'], bins=50)
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Polarity')
    plt.savefig('../results/sentiment.png')
    plt.show()
    
#Correlation matrix
def display_correlation_matrix(df):
    correlation_matrix = df.corr()
    print(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig
    plt.show()