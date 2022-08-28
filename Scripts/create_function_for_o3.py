import pandas as pd
import numpy as np
import boto3
    
from helpers import *

client = boto3.client('s3')
bucket='asdi-hackathon'
file_key = 'aq-sentinel/ozone/o3_sentinel_01-16june.csv'
obj = client.get_object(Bucket=bucket, Key=file_key)
df = pd.read_csv(obj['Body'])

df['ozone_total_vertical_column'] = abs(df['ozone_total_vertical_column'])

df = df[df['qa_value'] >= 1]

import math
df['latitude'] = df['latitude'].apply(math.radians)
df['longitude'] = df['longitude'].apply(math.radians)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

df_train, df_test = train_test_split(df, random_state = 0, test_size = 0.01)

X_train = df_train[['latitude', 'longitude']]
y_train = df_train['ozone_total_vertical_column'].ravel()
X_test = df_test[['latitude', 'longitude']]
y_test = df_test['ozone_total_vertical_column'].ravel()

cv = LeaveOneOut()

parameters = {'n_neighbors': np.arange(start=1, stop=200, step=2)}
grid_search = GridSearchCV(KNeighborsRegressor(weights = 'distance', algorithm = 'brute', metric = 'haversine'), 
                           param_grid = parameters, 
                           scoring='neg_root_mean_squared_error', 
                           n_jobs = -1,
                           cv = cv,
                           error_score = 'raise')
grid_search.fit(X_train, y_train)

X_train = df[['latitude', 'longitude']]
y_train = df['ozone_total_vertical_column'].ravel()

k = grid_search.best_params_['n_neighbors']
model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'brute', metric = 'haversine', n_jobs = -1)
model.fit(X_train, y_train)

upload_pickle_to_s3('asdi-hackathon', model, 'pickles/o3_model.pkl')