from sklearn import preprocessing
import pandas as pd


def gws(data_train_rs, data_train, data_test):
    # Male
    data_train_rs_m = data_train_rs[data_train_rs.gender == 1]
    data_train_m = data_train[data_train.gender == 1]
    data_test_m = data_test[data_test.gender == 1]
    X_train_rs = data_train_rs_m.iloc[:, :-6]
    X_train = data_train_m.iloc[:, :-6]
    X_test = data_test_m.iloc[:, :-6]
    scaler_m = preprocessing.RobustScaler(quantile_range=(25, 75))
    x = X_train_rs.values  # returns a numpy array
    x_trains = X_train.values
    x_tests = X_test.values
    scaler_m.fit_transform(x)
    x_trains = scaler_m.transform(x_trains)
    x_tests = scaler_m.transform(x_tests)
    X_train = pd.DataFrame(x_trains, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(x_tests, index=X_test.index, columns=X_test.columns)
    data_train_m.iloc[:, :-6] = X_train
    data_test_m.iloc[:, :-6] = X_test

    # Female
    data_train_rs_f = data_train_rs[data_train_rs.gender == 0]
    data_train_f = data_train[data_train.gender == 0]
    data_test_f = data_test[data_test.gender == 0]
    X_train_rs = data_train_rs_f.iloc[:, :-6]
    X_train = data_train_f.iloc[:, :-6]
    X_test = data_test_f.iloc[:, :-6]
    scaler_f = preprocessing.RobustScaler(quantile_range=(25, 75))
    x = X_train_rs.values  # returns a numpy array
    x_trains = X_train.values
    x_tests = X_test.values
    scaler_f.fit_transform(x)
    x_trains = scaler_f.transform(x_trains)
    x_tests = scaler_f.transform(x_tests)
    X_train = pd.DataFrame(x_trains, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(x_tests, index=X_test.index, columns=X_test.columns)
    data_train_f.iloc[:, :-6] = X_train
    data_test_f.iloc[:, :-6] = X_test

    data_train = pd.concat([data_train_f, data_train_m])
    data_test = pd.concat([data_test_f, data_test_m])

    return data_train, data_test, scaler_f, scaler_m
