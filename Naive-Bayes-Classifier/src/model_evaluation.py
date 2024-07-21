from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return report, cm
