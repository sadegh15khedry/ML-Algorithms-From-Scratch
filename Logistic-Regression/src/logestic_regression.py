import numpy as np
import pandas as pd


def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result

class CustomLogesticRegression():
    
    def __init__(self, learning_rate, number_of_iterations):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
        
    def fit(self, x, y):  
        number_of_samples, number_of_features = x.shape
        # self.weights = np.zeros((number_of_features, 1))
        self.weights = np.zeros(number_of_features)
        self.bias = 0
        
        y = y.values if isinstance(y, pd.Series) else y
        
        for i in range(self.number_of_iterations):
            
            linear_prdictions = np.dot(x, self.weights) + self.bias
            predictions = sigmoid(linear_prdictions)

            dw = (1 / number_of_samples) * np.dot(x.T, (predictions - y))
            db = (1/number_of_samples) * np.sum(predictions -y)
            
            self.weights = self.weights - (dw * self.learning_rate)
            self.bias = self.bias - (db * self.learning_rate)


    def predict(self, x):
        # Ensure x is a numpy array
        x = np.array(x)
        # Print shape of x
        print("Shape of x in predict:", x.shape)
        # Compute linear predictions
        linear_predictions = np.dot(x, self.weights) + self.bias
        # Print shape of linear_predictions
        print("Shape of linear_predictions:", linear_predictions.shape)
        # Apply sigmoid function to get probabilities
        y_pred = sigmoid(linear_predictions)
        # Print shape of y_pred
        print("Shape of y_pred:", y_pred.shape)
        # Convert probabilities to class labels using numpy's where function
        class_pred = np.where(y_pred > 0.5, 1, 0)
        # Print shape of class_pred
        print("Shape of class_pred:", class_pred.shape)
        return class_pred
                
        

        
        
    