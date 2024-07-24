from sklearn.ensemble import RandomForestRegressor 


def train_model(x_train, y_train, estimators=100):
    model = RandomForestRegressor (n_estimators=estimators, random_state=50)
    model.fit(x_train, y_train)
    return model

