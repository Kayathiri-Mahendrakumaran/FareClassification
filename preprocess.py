import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt
import datetime


def findDistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c

    return km


def preProcess_X(df):
    df = df.replace("nan", np.NaN)
    df.dropna(inplace=True)
    y = []
    if 'label' in df.columns:
        y = df.iloc[:, -1].values
        df = df.drop("label", axis=1)

    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    df['drop_time'] = pd.to_datetime(df['drop_time'])
    # df['day'] = df['pickup_time'].dt.dayofweek
    # df['hour'] = df['pickup_time'].dt.hour
    # tem = pd.get_dummies(df.day, prefix='day')
    # tem = tem.drop("day_6", axis=1)
    # df = pd.concat([df, tem], axis=1, sort=False)

    # tem = pd.get_dummies(df.hour, prefix='hour')
    # tem = tem.drop("hour_23", axis=1)
    # df = pd.concat([df, tem], axis=1, sort=False)

    df['distance'] = df.apply(lambda row: findDistance(row.pick_lon, row.pick_lat, row.drop_lon, row.drop_lat), axis=1)
    df['effective_time'] = df.apply(lambda row: row.duration - row.meter_waiting, axis=1)
    df['effective_fare'] = df.apply(lambda row: row.fare - row.additional_fare - row.meter_waiting_fare, axis=1)
    df['total_duration'] = df.apply(lambda row: (row.duration + row.meter_waiting_till_pickup)/row.distance, axis=1)
    df = df.drop("pickup_time", axis=1)
    df = df.drop("drop_time", axis=1)
    df = df.drop("tripid", axis=1)

    print(df.columns)

    X = df.iloc[:, :].values
    # print (X[0,-1])
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    return X, y




#
# train_filename = "Data/train.csv"
# df_train = pd.read_csv(train_filename)
# X = preProcess_X(df_train)
# y = preProcess_y(df_train)
