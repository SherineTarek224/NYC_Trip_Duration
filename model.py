from joblib import dump

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
import os
from preprocessing import *

target = ["log_trip_duration"]
numerical_feature = ["haversine_distance", "manhattan_distance",
                     "direction", "dropoff_longitude", "dropoff_latitude"]

categorical_feature = ["vendor_id", "day", "day_of_year", "day_of_week"
    , "hour", "month", "quarter", "passenger_count", "store_and_fwd_flag", "period"]


train_features = numerical_feature + categorical_feature
r_s = 42
np.random.seed(r_s)


# Evaluation function
def evaluate(model, x_train, y_train, x_val, y_val, name, degree=1):
    y_pre_train = model.predict(x_train)

    error_train = np.sqrt(mean_squared_error(y_train, y_pre_train))
    r2_train = r2_score(y_train, y_pre_train)

    print(f"Training RMSE {name} of degree {degree} = {error_train}")
    print(f"Training R2 {name} of degree {degree} = {r2_train}")

    y_pre_val = model.predict(x_val)

    error_val = np.sqrt(mean_squared_error(y_val, y_pre_val))
    r2_val = r2_score(y_val, y_pre_val)

    print(f"Val RMSE {name} of degree {degree} = {error_val}")
    print(f"Val R2 {name} of degree {degree} = {r2_val}")

    return error_train, r2_train, error_val, r2_val


# Linear Regression Function
def simple_linear_regression(x_train, y_train, regulized=True):

    if not regulized:
        model = LinearRegression(fit_intercept=True)
    else:
       model=Ridge( alpha=1.0,fit_intercept=True,random_state=r_s)

    model.fit(x_train, y_train)
    abs_mean_cof = abs(model.coef_).mean()
    intercept = model.intercept_

    print("Model abs mean of coefficients", abs_mean_cof)
    print("Model intercept", intercept)

    return model


# Polynomial Regression Function
def polynomial_regression(x_train,x_val,train_target,val_target, degrees, preprocessing_option, regulized=False):
    if not degrees:
        raise ValueError("The degree list is empty .Please provide at least 1 degree")

    error_train_every_deg = []
    error_test_every_deg = []
    model=None

    print(f"x_train shape {x_train.shape}")
    y_train=train_target.values
    print(f"y_train shape {y_train.shape}")

    print(f"x_val shape {x_val.shape}")
    y_val=val_target.values
    print(f"y_val shape{y_val.shape}")

    for deg in degrees:
        x_train_p, x_val_p= preprocessing_data(x_train, x_val, deg, preprocessing_option)
        model = simple_linear_regression(x_train_p, y_train, regulized)

        rmse_train, r2_train, rmse_val, r2_val = evaluate(
            model, x_train_p, y_train, x_val_p, y_val,
            "polynomial_regression", deg
        )

        error_train_every_deg.append(rmse_train)
        error_test_every_deg.append(rmse_val)
    return model


def save_model(model,root_path):# root path where to save the model

    file_name = "model.pkl"
    file_path = os.path.join(root_path, file_name)

    dump(model, file_path)
    print(f"Model saved temporarily at {file_path}")
if __name__=="__main__":

        train = load_data(training=True)
        val = load_data(training=False)
        df_train = prepare(train)
        df_val=prepare(val)
        model = polynomial_regression(df_train[train_features], df_val[train_features],df_train[target],df_val[target], degrees=[6], preprocessing_option=2, regulized=True)
        save_model(model,root_path=r"F:\Machine_Learning\projects\NYC_Trip_Duration")

