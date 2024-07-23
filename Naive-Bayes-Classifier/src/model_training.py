from sklearn.linear_model import LogisticRegression

from naive_bayes import CustomNaiveBayesClassifier


def train_model(x_train, y_train, number_of_iterations, learning_rate, model_type='custom'):
    if model_type == 'custom':
        model = CustomNaiveBayesClassifier(learning_rate, number_of_iterations)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = LogisticRegression ()
        model.fit(x_train, y_train)
        return model




