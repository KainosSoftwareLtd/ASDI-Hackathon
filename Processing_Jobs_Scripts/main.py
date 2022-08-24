import pickle
import pandas as pd
import numpy as np
import math
import pathlib
from haversine import *
from tqdm import tqdm
import boto3
from io import StringIO # python3; python2: BytesIO 
import pickle
import time
import math
from multiprocessing import Pool
from multiprocessing import cpu_count
#for Jupyter Notebooks parallel processing
#multiprocess works better within Jupyter Notebooks than multiprocessing package
#from multiprocess import Pool
#from multiprocess import cpu_count

#for using locally
ROOT_FOLDER_PATH = pathlib.Path().absolute().parent.as_posix()
PICKLE_FOLDER_PATH = ROOT_FOLDER_PATH + '/Pickles/'

def co_function(lat, lon, bucket = '', key = ''):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    if bucket == '':
        #load model from pickle locally
        co_model = pickle.load(open(PICKLE_FOLDER_PATH + 'co_model.pkl', 'rb'))
    else:
        #load model from pickle in s3 bucket
        s3 = boto3.resource('s3')
        co_model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
    
    #predict with model
    preds = co_model.predict(input)
    return preds[0]

def no2_function(lat, lon, bucket = '', key = ''):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    if bucket == '':
        #load model from pickle locally
        no2_model = pickle.load(open(PICKLE_FOLDER_PATH + 'no2_model.pkl', 'rb'))
    else:
        #load model from pickle in s3 bucket
        s3 = boto3.resource('s3')
        no2_model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
    
    #predict with model
    preds = no2_model.predict(input)
    return preds[0]

def o3_function(lat, lon, bucket = '', key = ''):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    if bucket == '':
        #load model from pickle locally
        o3_model = pickle.load(open(PICKLE_FOLDER_PATH + 'o3_model.pkl', 'rb'))
    else:
        #load model from pickle in s3 bucket
        s3 = boto3.resource('s3')
        o3_model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
    
    #predict with model
    preds = o3_model.predict(input)
    return preds[0]

def so2_function(lat, lon, bucket = '', key = ''):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    if bucket == '':
        #load model from pickle locally
        so2_model = pickle.load(open(PICKLE_FOLDER_PATH + 'so2_model.pkl', 'rb'))
    else:
        #load model from pickle in s3 bucket
        s3 = boto3.resource('s3')
        so2_model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
    
    #predict with model
    preds = so2_model.predict(input)
    return preds[0]

def ai_function(lat, lon, bucket = '', key = ''):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    if bucket == '':
        #load model from pickle locally
        ai_model = pickle.load(open(PICKLE_FOLDER_PATH + 'ai_model.pkl', 'rb'))
    else:
        #load model from pickle in s3 bucket
        s3 = boto3.resource('s3')
        ai_model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
    
    #predict with model
    preds = ai_model.predict(input)
    return preds[0]

def popdensity_function(lat, lon, bucket = '', key = ''):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    if bucket == '':
        #load model from pickle locally
        popdensity_model = pickle.load(open(PICKLE_FOLDER_PATH + 'popdensity_model.pkl', 'rb'))
    else:
        #load model from pickle in s3 bucket
        s3 = boto3.resource('s3')
        popdensity_model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
    
    #predict with model
    preds = popdensity_model.predict(input)
    return preds[0]

def get_aqs_function(lat, lon, bucket = '', key = ''):
    """Takes a latitude and longitude coordinate and calculates the Air Quality Index at this point.

    Args:
        lat, lon (float): single float value for each. It is required that these lat, lon values exist in the referenced CSV file.
        bucket (string): name of s3 bucket where data stored; if empty string, read from local data file
        key (string): path of data file in s3 bucket; cannot be empty string unless bucket also empty string

    Returns:
        The Air Quality Index at the specified coordinates. Smaller value = higher air quality.
    """
    if bucket == '':
        #read locally
        df = pd.read_csv(ROOT_FOLDER_PATH + '/Spikes/Dash/data/final_df.csv', index_col = 0)
    else:
        #read from s3 bucket
        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        
        df = df[np.isclose(df['Latitude'], lat)]
        df = df[np.isclose(df['Longitude'], lon)]
        return df['AQ_score'].item()
  
def convert_point_list_to_df(points):
    """Converts list of points to a dataframe with Latitude and Longitude columns

    Args:
        points (list of float tuples): List of float tuples representing lat/long points

    Returns:
        DataFrame: A DataFrame with latitude and longitude points
    """
    latitude = []
    longitide = []

    for point in points:
        latitude.append(point[0])
        longitide.append(point[1])
      
      
    points_df = pd.DataFrame({'Latitude': latitude, 'Longitude' : longitide})
    return points_df


def get_spaced_point_set_in_bbox(d, bottom_left, top_right):
    """ Get an evenly spaced set of points from a given bounding box. Uses the haversine formula.

    Args:
        d (float): the diameter of the circular space around a point in km (the resolution wanted)
        bottom_left (float tuple): lat/long
        top_right (float tuple): lat/long

    Returns:
        DataFrame: points in the bounding box ordered by row left to right, top to bottom
    """
    top_left = (top_right[0], bottom_left[1])

    # 2. Divide vertical length by diameter to work out number of rows
    length = haversine(top_left, bottom_left)
    num_rows = int(length // d)
    print(num_rows)

    # 3. iterate through rows
    points = []
    for r in tqdm(range(num_rows)):
        # i. Get row start point and row end point moving right to left by moving r*diameter down from top left and top right
        row_start_point = inverse_haversine(top_left, r * d, Direction.SOUTH)
        row_end_point = inverse_haversine(top_right, r * d, Direction.SOUTH)
      
        # ii. calcualte horizontal distance to right line
        row_width = haversine(row_start_point, row_end_point)

        #iii. Divide by diameter of circle to get circles on the row
        number_points_on_row = int(row_width // d)

        # iv. If there's a remainder, divide by 2 and shift along to the right
        shift = ((row_width / d) % 1) / 2
        row_start_point = inverse_haversine(row_start_point, shift, Direction.EAST)

        # v. create calcuated number of circles on the row
        points_in_row_list = []
        for i in range(number_points_on_row):
            points_in_row_list.append(inverse_haversine(row_start_point, i*d, Direction.EAST))

        points = points + points_in_row_list
      
    point_set_df = convert_point_list_to_df(points)

    return point_set_df

def parallelise(df, func):
    n_cores = cpu_count()
    df_splits = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    results = pool.map(func, df_splits)
    pool.close()
    pool.join()
    df = pd.concat(results)
    return df

def apply_aq_functions(df):
    #molar mass constants
    co_molar_mass = 28.01
    no2_molar_mass = 46.0055
    o3_molar_mass = 48
    so2_molar_mass = 64.066

    #apply aq functions to each row (using latitude and longitude columns) and multiply by associated molar mass to give g/m2
    #axis = 1, apply function to each row
    
    df['Value_co'] = df.apply(lambda row : co_function(row['Latitude'], row['Longitude']) * co_molar_mass, axis=1)
    #print('co_function complete')
    df['Value_no2'] = df.apply(lambda row : no2_function(row['Latitude'], row['Longitude']) * no2_molar_mass, axis=1)
    #print('no2_function complete')
    df['Value_o3'] = df.apply(lambda row : o3_function(row['Latitude'], row['Longitude']) * o3_molar_mass, axis=1)
    #print('o3_function complete')
    df['Value_so2'] = df.apply(lambda row : so2_function(row['Latitude'], row['Longitude']) * so2_molar_mass, axis=1)
    #print('so2_function complete')
    df['Value_ai'] = df.apply(lambda row : ai_function(row['Latitude'], row['Longitude']), axis=1)
    #print('ai_function complete')
    return df

def normalise(df):
    norm_cols = ['Value_co', 'Value_no2', 'Value_o3', 'Value_so2', 'Value_ai']
    #normalise each aq metric value set between 1 and 0 where 0 = 0% and 1 = 20%
    for i in df[norm_cols]:   #normalise aq value columns
        df['norm_' + i]=(df[i]-df[i].min())/(df[i].max()-df[i].min())
    return df

def aqs_function(aq1, aq2, aq3, aq4, aq5):
    #smaller value = better air quality
    aqs = (aq1 * (20/100)) + (aq2 * (20/100)) + (aq3 * (20/100)) + (aq4 * (20/100)) + (aq5 * (20/100))
    return aqs

def apply_aqs_function(df):
    #assumption: each metric is worth 20% of AQS, 100 / 5 metrics
    #apply calculate_aqi function to each row of the 5 aq columns
    df['AQ_score'] = df.apply(lambda row : aqs_function(row['norm_Value_co'], 
                                                            row['norm_Value_no2'], 
                                                            row['norm_Value_o3'], 
                                                            row['norm_Value_so2'], 
                                                            row['norm_Value_ai']), axis=1)
    
    #drop normalised columns
    df = df.drop(['norm_Value_co', 'norm_Value_no2', 'norm_Value_o3', 'norm_Value_so2', 'norm_Value_ai'], axis = 1)
    return df

def apply_popd_function(df):
    #same as above apply aq functions but with...
    #popdensity_function
    df['Pop_density'] = df.apply(lambda row : popdensity_function(row['Latitude'], row['Longitude']), axis=1)
    return df

def greenspace_score_function(df, aqs, pop_density, land_type):
    #Population Density
    popd_pct = 50/100
    #50m2 per capita according to WHO standards or 100m2 (our resolution) per 2 people
    standard_gs_per_pop_m2 = 50
    sum_df_popd = df['Pop_density'].sum()
    sum_df_greenspace_m2 = df[df['Land_type'] == 'greenspace'].sum() * 100   #sum of greenspace multiplied by resolution
    gs_per_capita = sum_df_greenspace_m2 / sum_df_popd
    #if current greenspace per capita is BETTER than WHO standards, it is LESS likely greenspace is required so PENALISE lower weighting
    #weight <1 will decrease contribution of pop density to greenspace score
    #if current greenspace per capita is WORSE than WHO standards, it is MORE likely greenspace is required so REWARD higher weighting
    #weight >1 will increase contribution of pop density to greenspace score
    #weight =1 means weighting is essentially cancelled out
    popd_weight = gs_per_capita / standard_gs_per_pop_m2
    
    #Land Type
    if (land_type == 'hospital'):
        penalty_reward = 0
    elif (land_type == 'hospital'):
        penalty_reward = 0
    elif land_type in ['hospital', 'hospital']:
        penalty_reward = 0
    else:
        penalty_reward = 1   #no penalty_reward
        
    #Air Quality Score
    aqs_pct = 100 - sum(popd_weight * popd_pct)
    aqs_weight = 1
        
    Greenspace_score = (aqs * (aqs_weight * aqs_pct)) + (pop_density * (popd_weight * popd_pct)) * penalty_reward
    return Greenspace_score
    
def apply_greenspace_score_function(df):
    return df
    df['Greenspace_score'] = df.apply(lambda row : greenspace_score_function(row['Land_type'], 
                                                                                    row['AQ_score'], 
                                                                                    row['Pop_density']), axis=1)

def fill_points_df(bucket = '', key = ''):
    
    if bucket == '':
        #read locally
        df = pd.read_csv(ROOT_FOLDER_PATH + '/Spikes/Dash/data/points_df.csv', index_col = 0)
    else:
        #read from s3 bucket
        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
    
    start = time.time()
    df = parallelise(df, apply_aq_functions)
    end = time.time()
    time_taken1 = round(end - start, 2)
    print('apply_aq_functions complete')
    print('Time taken:', time_taken1)
    
    start = time.time()
    df = parallelise(df, apply_popd_function)
    df = normalise(df)
    print('normalisation of aq values complete')
    df = apply_aqs_function(df)
    print('apply_aqs_function complete')
    end = time.time()
    time_taken2 = round(end - start, 2)
    print('apply_popd_function complete')
    print('Time taken:', time_taken2)
    
    start = time.time()
    df = apply_greenspace_score_function(df)
    end = time.time()
    time_taken3 = round(end - start, 2)
    print('apply_greenspace_score_function complete')
    print('Time taken:', time_taken3)
    
    total_time_taken = time_taken1 + time_taken2 + time_taken3
    print('TOTAL time taken:', total_time_taken)
    
    return df
            
def upload_pickle_to_s3(bucket, model, key):
    """  Pickle model and upload to the designated S3 AWS bucket

    Args:
        bucket (string): name of bucket, e.g. asdi-hackathon
        model (object): in memory model to pickle
        key (string): path in bucket to save to including any subfolders and the filename and extension

    Returns:
        Confirmation of successful/unsuccessful upload
    """
    s3 = boto3.resource('s3')
    try:
        pickle_byte_obj = pickle.dumps([model])
        s3.Object(bucket,key).put(Body=pickle_byte_obj)
        print('Successful upload')
    except:
        print('Failed upload')
        
def upload_df_to_s3(bucket, df, key):
    """  Pickle model and upload to the designated S3 AWS bucket

    Args:
        bucket (string): name of bucket, e.g. asdi-hackathon
        df (Pandas dataframe): in memory dataframe to save as .csv
        key (string): path in bucket to save to including any subfolders and the filename and extension

    Returns:
        Confirmation of successful/unsuccessful upload
    """
    s3 = boto3.resource('s3')
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        s3.Object(bucket, key).put(Body=csv_buffer.getvalue())
        print('Successful upload')
    except:
        print('Failed upload')
        
def read_csv_from_s3(bucket, key):
    client = boto3.client('s3')
    obj = client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    return df

def create_function_for_aerosol_index():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/aerosol-index/ai_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df = df[df['qa_value'] >= 1]

    import math
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(df, random_state = 0, test_size = 0.01)

    X_train = df_train[['latitude', 'longitude']]
    y_train = df_train['aerosol_index_340_380'].ravel()
    X_test = df_test[['latitude', 'longitude']]
    y_test = df_test['aerosol_index_340_380'].ravel()

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
    y_train = df['aerosol_index_340_380'].ravel()

    k = grid_search.best_params_['n_neighbors']
    model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'brute', metric = 'haversine', n_jobs = -1)
    model.fit(X_train, y_train)

    upload_pickle_to_s3('asdi-hackathon', model, 'pickles/ai_model.pkl')
    
def create_function_for_co():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/carbon-monoxide/co_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['carbonmonoxide_total_column_corrected'] = abs(df['carbonmonoxide_total_column_corrected'])

    df = df[df['qa_value'] >= 0.7]

    import math
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(df, random_state = 0, test_size = 0.01)

    X_train = df_train[['latitude', 'longitude']]
    y_train = df_train['carbonmonoxide_total_column_corrected'].ravel()
    X_test = df_test[['latitude', 'longitude']]
    y_test = df_test['carbonmonoxide_total_column_corrected'].ravel()

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
    y_train = df['carbonmonoxide_total_column_corrected'].ravel()

    k = grid_search.best_params_['n_neighbors']
    model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'brute', metric = 'haversine', n_jobs = -1)
    model.fit(X_train, y_train)

    upload_pickle_to_s3('asdi-hackathon', model, 'pickles/co_model.pkl')
    
def create_function_for_no2():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/nitrogen-dioxide/no2_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['nitrogendioxide_tropospheric_column'] = abs(df['nitrogendioxide_tropospheric_column'])

    df = df[df['qa_value'] >= 1]   #0.9 for little more data or 1 for greatest quality measurements

    import math
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(df, random_state = 0, test_size = 0.01)

    X_train = df_train[['latitude', 'longitude']]
    y_train = df_train['nitrogendioxide_tropospheric_column'].ravel()
    X_test = df_test[['latitude', 'longitude']]
    y_test = df_test['nitrogendioxide_tropospheric_column'].ravel()

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
    y_train = df['nitrogendioxide_tropospheric_column'].ravel()

    k = grid_search.best_params_['n_neighbors']
    model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'brute', metric = 'haversine', n_jobs = -1)
    model.fit(X_train, y_train)

    upload_pickle_to_s3('asdi-hackathon', model, 'pickles/no2_model.pkl')
    
def create_function_for_o3():
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
    from sklearn.model_selection import train_test_split

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
    
def create_function_for_popdensity():
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

    import sys
    import pathlib
    ROOT = pathlib.Path().absolute().parent.as_posix()
    if ROOT not in sys.path:
        sys.path.append(ROOT)

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
    
def create_function_for_so2():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/sulphur-dioxide/so2_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['sulfurdioxide_total_vertical_column'] = abs(df['sulfurdioxide_total_vertical_column'])

    df = df[df['qa_value'] >= 1]

    import math
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(df, random_state = 0, test_size = 0.01)

    X_train = df_train[['latitude', 'longitude']]
    y_train = df_train['sulfurdioxide_total_vertical_column'].ravel()
    X_test = df_test[['latitude', 'longitude']]
    y_test = df_test['sulfurdioxide_total_vertical_column'].ravel()

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
    y_train = df['sulfurdioxide_total_vertical_column'].ravel()

    k = grid_search.best_params_['n_neighbors']
    model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'brute', metric = 'haversine', n_jobs = -1)
    model.fit(X_train, y_train)

    upload_pickle_to_s3('asdi-hackathon', model, 'pickles/so2_model.pkl')
    
def create_df_with_all_spatial_points():
    points_df = get_spaced_point_set_in_bbox(0.25, (51.239405, -0.625211), (51.737184, 0.328289))
    upload_df_to_s3('asdi-hackathon', points_df, 'points_df.csv')
    
def create_final_df():
    final_df = fill_points_df('asdi-hackathon', 'points_df.csv')
    upload_df_to_s3(bucket = 'asdi-hackathon', df = final_df, key = 'final-data/final_df.csv')

#ALGORITHM
#create_function_for_aerosol_index()
#create_function_for_co()
#create_function_for_no2()
#create_function_for_o3()
#create_function_for_popdensity()
#create_function_for_so2()
create_df_with_all_spatial_points()
#create_final_df()