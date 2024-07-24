from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return report, cm

def get_error(y_test, y_pred_test):
    mse_train = mean_squared_error(y_test, y_pred_test)
    print(f"Training MSE: {mse_train}")
