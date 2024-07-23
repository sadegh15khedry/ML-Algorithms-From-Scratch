from sklearn.naive_bayes import GaussianNB

from naive_bayes import CustomNaiveBayesClassifier


def train_model(x_train, y_train, model_type='custom'):
    if model_type == 'custom':
        model = CustomNaiveBayesClassifier()
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = GaussianNB()
        model.fit(x_train, y_train)
        return model




