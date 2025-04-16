from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures, \
    FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from prepare import *

target = ["log_trip_duration"]
numerical_feature = [

    "haversine_distance", "manhattan_distance",
    "direction", "dropoff_longitude", "dropoff_latitude"]

categorical_feature = ["vendor_id", "day", "day_of_year", "day_of_week"
    , "hour", "month", "quarter", "passenger_count", "store_and_fwd_flag", "period"]


train_features = numerical_feature + categorical_feature

cols_with_outliers = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
r_s = 42
np.random.seed(r_s)


def remove_outlier(df_train, cols_with_outliers, option=1, threshold=3, clip=False):
    upper = None
    lower = None
    for col in cols_with_outliers:
        if option == 1:
            q1 = df_train[col].quantile(0.25)
            q3 = df_train[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
        elif option == 2:
            mean = df_train[col].mean()
            std = df_train[col].std()
            upper = mean + std * threshold
            lower = mean - std * threshold
        else:
            raise ValueError("Option must be 1 (IQR) or 2 (Z-score)")
        if clip:
            df_train[col] = df_train[col].clip(lower, upper)
        else:
            df_train = df_train[(df_train[col] >= lower) & (df_train[col] <= upper)]

    return df_train



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


def log_function(x):
    return np.log1p(np.maximum(x, 0))


def new_feature_name(_, names: list[str]):
    return [name + '_log' for name in names]


def preprocessing_function(preprocessing_option, degree):
    scaler = choose_scaling_method(preprocessing_option)
    log_features = FunctionTransformer(log_function, feature_names_out=new_feature_name)

    numerical_transformer = Pipeline(steps=[

        ('scaler',scaler),
        ('poly', PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False)),
        ('log', log_features)

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


def preprocessing_data(train_features, val_features, degree, preprocessing_option=2):
    preprocessing_pipeline = preprocessing_function(preprocessing_option, degree)
    # prevent Data Leakage
    train_preprocessed = preprocessing_pipeline.fit_transform(train_features)
    val_preprocessed = preprocessing_pipeline.transform(val_features)

    # train_features_after_preprocessing=preprocessing_pipeline.named_steps["preprocessing"].get_feature_names_out(train_features)

    print(f"Shape of preprocessed training data of {degree} = {train_preprocessed.shape}")
    print(f"Shape of preprocessed test data of {degree} = {val_preprocessed.shape}")
    # print("Number of feature names:", len(train_features_after_preprocessing))

    return train_preprocessed, val_preprocessed


if __name__=="__main__":
    train=load_data(training=True)
    val=load_data(training=False)
    df_train=prepare(train)
    df_val=prepare(val)
    preprocessing_data(df_train[train_features], df_val[train_features], degree=1, preprocessing_option=2)
