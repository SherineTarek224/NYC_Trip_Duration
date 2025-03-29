from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from prepare import *

target = ["log_trip_duration"]
numerical_feature = [
    "pickup_longitude", "pickup_latitude",
    "log_haversine_distance", "log_manhattan_distance",
    "log_direction", "dropoff_longitude", "dropoff_latitude"]

categorical_feature = ["vendor_id", "day", "day_of_year", "day_of_week", "hour", "month", "quarter", "passenger_count",
                       "store_and_fwd_flag", "period"]
train_features = numerical_feature + categorical_feature
cols_with_outliers = ["log_haversine_distance"
    , "log_manhattan_distance", "log_direction", "pickup_longitude", "pickup_latitude"
    , "dropoff_longitude", "dropoff_latitude"]

r_s = 42
np.random.seed(r_s)


def split_data(df):
    y = df[target].astype('float32')
    x = df[train_features]  # keep dataframe
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=r_s)

    return x_train, x_test, y_train, y_test  # x_test,x_train,y_train,y_test are dataframe


def remove_outliers(x_train, y_train, x_test, y_test, clip=False, threshold=3, option=1):
    bounds = {}
    if option == 1:
        for col in cols_with_outliers:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            bounds[col] = (lower, upper)

    elif option == 2:

        means = x_train[cols_with_outliers].mean()
        stds = x_train[cols_with_outliers].std()
        for col in cols_with_outliers:
            lower = means[col] - threshold * stds[col]
            upper = means[col] + threshold * stds[col]
            bounds[col] = (lower, upper)

    if clip:
        # Clip outliers in-place using bounds
        x_train_clipped = x_train.copy()
        x_test_clipped = x_test.copy()
        for col in cols_with_outliers:
            lower, upper = bounds[col]
            x_train_clipped[col] = x_train[col].clip(lower, upper)
            x_test_clipped[col] = x_test[col].clip(lower, upper)
        return x_train_clipped, y_train, x_test_clipped, y_test
    else:
        # Filter outlier rows
        mask_train = pd.concat(
            [x_train[col].between(*bounds[col]) for col in cols_with_outliers],
            axis=1
        ).all(axis=1)
        mask_test = pd.concat(
            [x_test[col].between(*bounds[col]) for col in cols_with_outliers],
            axis=1
        ).all(axis=1)

        return (
            x_train[mask_train], y_train[mask_train],
            x_test[mask_test], y_test[mask_test]
        )


def choose_scaling_method(preprocessing_option):
    if preprocessing_option == 1:
        return MinMaxScaler()
    elif preprocessing_option == 2:
        return StandardScaler()
    elif preprocessing_option == 3:
        return RobustScaler()
    elif preprocessing_option == 4:
        return None
    else:
        raise ValueError("Invalid preprocessing option:choose 1,2,3 or 4")


def preprocessing_function(preprocessing_option, degree):
    scaler = choose_scaling_method(preprocessing_option)

    numerical_transformer = Pipeline(steps=[
        ("poly", PolynomialFeatures(degree, include_bias=True, interaction_only=False)),
        ("scaler", scaler),

    ])
    categorical_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    col_transformer = ColumnTransformer(  # using pipeline and column Transformer
        # convert data# from DataFrame to numpy array
        transformers=[
            ("numerical", numerical_transformer, numerical_feature),
            ("categorical", categorical_transformer, categorical_feature)
        ],
        remainder='passthrough'
    )

    preprocessing_pipeline = Pipeline(steps=[
        ("preprocessing", col_transformer)
    ])
    return preprocessing_pipeline


def preprocessing_data(train, test, degree, preprocessing_option=1):
    preprocessing_pipeline = preprocessing_function(preprocessing_option, degree)
    # prevent Data Leakage
    train_preprocessed = preprocessing_pipeline.fit_transform(train)
    test_preprocessed = preprocessing_pipeline.transform(test)

    # train_features_after_preprocessing=preprocessing_pipeline.named_steps["preprocessing"].get_feature_names_out(train_features)

    print(f"Shape of preprocessed training data of {degree} = {train_preprocessed.shape}")
    print(f"Shape of preprocessed test data of {degree} = {test_preprocessed.shape}")
    # print("Number of feature names:", len(train_features_after_preprocessing))

    return train_preprocessed, test_preprocessed


if __name__=="__main__":

        df = load_data()

        df = prepare(df)


        x_train, x_test, y_train, y_test = split_data(df)

        print(f"Training_Shape {x_train.shape}")
        print(f"Training_Target-Shape {y_train.shape}")
        print(f"Test_Shape {x_test.shape}")
        print(f"Test_Target_Shape {y_test.shape}")

        x_train, y_train, x_test, y_test = remove_outliers(x_train, y_train, x_test, y_test, clip=False, threshold=3,
                                                           option=2)

        X_train_prepr, X_test_prepr = preprocessing_data(x_train, x_test, degree=1, preprocessing_option=2)
        print(f"X_train after preprocessing {X_train_prepr.shape}")
        print(f"X_test after preprocessing {X_test_prepr.shape}")



