from joblib import dump

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from preprocessing import *
from prepare import *
import os

seed = 42
np.random.seed(seed)


# Evaluation function
def evaluate(model, X_train, t_train, X_test, t_test, name, degree=1):
    t_pre_train = model.predict(X_train)

    error_train = np.sqrt(mean_squared_error(t_train, t_pre_train))
    r2_train = r2_score(t_train, t_pre_train)

    print(f"Training RMSE {name} of degree {degree} = {error_train}")
    print(f"Training R2 {name} of degree {degree} = {r2_train}")

    t_pre_test = model.predict(X_test)

    error_test = np.sqrt(mean_squared_error(t_test, t_pre_test))
    r2_test = r2_score(t_test, t_pre_test)

    print(f"Test RMSE {name} of degree {degree} = {error_test}")
    print(f"Test R2 {name} of degree {degree} = {r2_test}")

    return error_train, r2_train, error_test, r2_test


# Linear Regression Function
def simple_linear_regression(X_train, t_train, regulized=True):
    if not regulized:
        model = LinearRegression(fit_intercept=True)
    else:
        model = Ridge(alpha=1.0, random_state=seed)

    model.fit(X_train, t_train)
    abs_mean_cof = abs(model.coef_).mean()
    intercept = model.intercept_

    print("Model abs mean of coefficients", abs_mean_cof)
    print("Model intercept", intercept)

    return model


# Polynomial Regression Function
def polynomial_regression(X_train, t_train, X_test, t_test, degrees, preprocessing_option, regulized=False):
    if not degrees:
        raise ValueError("The degree list is empty .Please provide at least 1 degree")

    error_train_every_deg = []
    error_test_every_deg = []
    model=None

    for deg in degrees:
        X_train_p, X_test_p= preprocessing_data(X_train, X_test, deg, preprocessing_option)
        model = simple_linear_regression(X_train_p, t_train, regulized)

        rmse_train, r2_train, rmse_test, r2_test = evaluate(
            model, X_train_p, t_train, X_test_p, t_test,
            "polynomial_regression", deg
        )

        error_train_every_deg.append(rmse_train)
        error_test_every_deg.append(rmse_test)
    return model


# Lasso Function for Feature Selection and Linear Regression
def lasso(X_train, t_train, X_test, t_test, deg, preprocessing_option, regulized=True):
    # Preprocess data
    X_train_p, X_test_p= preprocessing_data(X_train, X_test, deg, preprocessing_option)

    # Fit Lasso model for feature selection
    lasso = Lasso(alpha=0.001, fit_intercept=True, max_iter=5000, random_state=seed)

    lasso.fit(X_train_p, t_train)

    # Get indices and names of selected features
    selected_feature_indices = [i for i, coef in enumerate(lasso.coef_) if coef != 0]


    print("Selected features:", selected_feature_indices)
    print("length of selected features",len(selected_feature_indices))

    # Select columns based on Lasso-selected indices
    X_train_selected = X_train_p[:, selected_feature_indices]
    X_test_selected = X_test_p[:, selected_feature_indices]

    # Fit Ridge (or Linear Regression) on selected features
    model = simple_linear_regression(X_train_selected, t_train, regulized)

    # Evaluate the model
    rmse_train, r2_train, rmse_test, r2_test = evaluate(
        model, X_train_selected, t_train, X_test_selected, t_test,
        name="Lasso then Ridge", degree=deg
    )
    return model


# Main Execution Block
if __name__ == "__main__":
    df = load_data()

    df = prepare(df)

    x_train, x_test, y_train, y_test = split_data(df)

    print(f"Training_Shape {x_train.shape}")
    print(f"Training_Target-Shape {y_train.shape}")
    print(f"Test_Shape {x_test.shape}")
    print(f"Test_Target_Shape {y_test.shape}")

    x_train, y_train, x_test, y_test = remove_outliers(x_train, y_train, x_test, y_test, clip=True, threshold=3,
                                                       option=2)
    
    model=polynomial_regression(x_train, y_train, x_test, y_test, degrees=[6], preprocessing_option=2, regulized=True)
    print("/////////// /???????????????????????????//////////////////")

    #lasso(X_train, t_train, X_test, t_test, deg=4, preprocessing_option=1, regulized=True)

    #save model which use the sample data  in pkl file
    root_path=r"F:\Machine_Learning\projects\Trip_duration_predict\project"
    file_name="sample_data_model.pkl"
    file_path=os.path.join(root_path,file_name)
    dump(model,file_path)
    print(f"Sample_Model saved Successfully at {file_path}")

    #save model which use the whole dataset
    # root_path = r"F:\Machine_Learning\projects\Trip_duration_predict\project"
    # file_name = "model.pkl"
    # file_path = os.path.join(root_path, file_name)
    # dump(model, file_path)
    # print(f"Model saved Successfully at {file_path}")
