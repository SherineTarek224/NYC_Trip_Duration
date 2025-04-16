import pandas as pd
import numpy as np
from geopy import distance
import math


def load_data(training=True):

    if training:
        train = pd.read_csv(r"Dataset/Data/train.csv")
        # may need to extract train.zip, val.zip files
        return train
    # for sample data
    # val=pd.read_csv("val_sample.csv")
    # train=pd.concat([val,train])
    else:
        val = pd.read_csv(r"Dataset/Data/val.csv")
        return val


def hour_period(hour):
    if 12 <= hour <= 17:
        return "afternoon"
    elif 5 <= hour < 12:
        return "morning"
    else:
        return "Night"


def haversine_dist(row):
    pick = (row["pickup_latitude"], row["pickup_longitude"])
    drop = (row["dropoff_latitude"], row["dropoff_longitude"])
    dist = distance.geodesic(pick, drop).km
    return dist


def manhattan_dis(row):
    lat_dis = abs(row["pickup_latitude"] - row["dropoff_latitude"])
    long_dis = abs(row["pickup_longitude"] - row["dropoff_longitude"])
    return lat_dis + long_dis


def direction(row):
    pick = (row['pickup_latitude'], row['pickup_longitude'])
    drop = (row['dropoff_latitude'], row['dropoff_longitude'])

    delta_longitude = drop[1] - pick[1]  # difference in longitude
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(drop[0]))
    x = math.cos(math.radians(pick[0])) * math.sin(math.radians(drop[0])) - \
        math.sin(math.radians(pick[0])) * math.cos(math.radians(drop[0])) * \
        math.cos(math.radians(delta_longitude))

    # Calculate the bearing in degrees
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)

    # Adjust the bearing to be in the range [0, 360)
    bearing = (bearing + 360) % 360

    return bearing


def prepare(df):

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["period"] = df["hour"].apply(hour_period)
    df["month"] = df["pickup_datetime"].dt.month
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["quarter"] = df["pickup_datetime"].dt.quarter

    df["haversine_distance"] = df.apply(haversine_dist, axis=1)

    df["manhattan_distance"] = df.apply(manhattan_dis, axis=1)

    df["direction"] = df.apply(direction, axis=1)

    df["log_trip_duration"] = np.log1p(df["trip_duration"])  # ln(x+1)

    df.drop(columns=['id', 'pickup_datetime'], inplace=True)

    return df

if __name__=="__main__":
    train=load_data(training=True)
    val=load_data(training=False)
    df_train=prepare(train)
    print(f"Training Shape {df_train.shape}")
    df_val=prepare(val)
    print(f"Val shape {df_val.shape}")




