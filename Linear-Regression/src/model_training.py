from sklearn.linear_model import LinearRegression 

from linear_regression import CustomLinearRegression


def train_model(x_train, y_train, number_of_iterations, learning_rate, model_type='custom'):
    if model_type == 'custom':
        model = CustomLinearRegression(learning_rate, number_of_iterations)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = LinearRegression ()
        model.fit(x_train, y_train)
        return model




