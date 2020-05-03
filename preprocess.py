import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt


def findDistance(lon1,lat1, lon2,lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    # km = 6371 * c

    return c
def preProcess_X(df):
    df = df.replace("nan", np.NaN)
    # df['additional_fare'].fillna(df['additional_fare'].mode().iloc[0], inplace=True)
    df.fillna(df.mean(), inplace=True)
    # print(df.isnull().sum())

    # df['distance'] = df.apply(lambda row: findDistance(row.pick_lon,row.pick_lat, row.drop_lon, row.drop_lat), axis=1)
    # df['dlon'] = df.apply(lambda row: abs(row.drop_lon - row.pick_lon), axis=1)
    # df['dlat'] = df.apply(lambda row: abs(row.drop_lat - row.pick_lat), axis=1)
    print(df.head())
    X = df.iloc[:, [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]].values
    # print (X[0,-1])
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X


def preProcess_y(df):
    y = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    return y

#
# train_filename = "Data/train.csv"
# df_train = pd.read_csv(train_filename)
# X = preProcess_X(df_train)
# y = preProcess_y(df_train)

