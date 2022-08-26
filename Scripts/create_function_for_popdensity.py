import pandas as pd
import numpy as np
import boto3
    
from helpers import *

client = boto3.client('s3')
bucket='asdi-hackathon'
file_key1 = 'population-data/popcsv/longitude0.csv'
file_key2 = 'population-data/popcsv/longitudeNeg10.csv'
obj1 = client.get_object(Bucket=bucket, Key=file_key1)
obj2 = client.get_object(Bucket=bucket, Key=file_key2)
df1 = pd.read_csv(obj1['Body'])
df2 = pd.read_csv(obj2['Body'])

df = pd.concat([df1, df2], ignore_index=True)
df = df.drop('Unnamed: 0', axis = 1)

df_filter = df[(df['latitude'] >= 51.239405) & (df['latitude'] <= 51.737184)]
df_filter = df_filter[(df_filter['longitude'] >= -0.625211) & (df_filter['longitude'] <= 0.328289)]
df = df_filter

df['population'] = df['population'].astype(np.float32)

import math
df['latitude'] = df['latitude'].apply(math.radians)
df['longitude'] = df['longitude'].apply(math.radians)

from sklearn.neighbors import KNeighborsRegressor

#use all of data in train
X_train = df[['latitude', 'longitude']]
y_train = df['population'].ravel()

k = 1
model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'ball_tree', metric = 'haversine', n_jobs = -1)
model.fit(X_train, y_train)

upload_pickle_to_s3('asdi-hackathon', model, 'pickles/popdensity_model.pkl')