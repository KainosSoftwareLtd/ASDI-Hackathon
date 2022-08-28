import pickle
import pandas as pd
import numpy as np
import math
import boto3
from io import StringIO # python3; python2: BytesIO 
import pickle
import math
from time import time
#from multiprocessing import Pool
#from multiprocessing import cpu_count
#for Jupyter Notebooks parallel processing
#multiprocess works better within Jupyter Notebooks than multiprocessing package
from multiprocess import Pool
from multiprocess import cpu_count
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import BallTree

def create_ai_pickle():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/aerosol-index/ai_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df = df[df['qa_value'] >= 1]

    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

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

def create_co_pickle():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/carbon-monoxide/co_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['carbonmonoxide_total_column_corrected'] = abs(df['carbonmonoxide_total_column_corrected'])

    df = df[df['qa_value'] >= 0.7]

    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

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

def create_no2_pickle():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/nitrogen-dioxide/no2_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['nitrogendioxide_tropospheric_column'] = abs(df['nitrogendioxide_tropospheric_column'])

    df = df[df['qa_value'] >= 1]   #0.9 for little more data or 1 for greatest quality measurements

    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

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

def create_o3_pickle():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/ozone/o3_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['ozone_total_vertical_column'] = abs(df['ozone_total_vertical_column'])

    df = df[df['qa_value'] >= 1]

    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

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

def create_popdensity_pickle():
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

    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

    #use all of data in train
    X_train = df[['latitude', 'longitude']]
    y_train = df['population'].ravel()

    k = 1
    model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', algorithm = 'ball_tree', metric = 'haversine', n_jobs = -1)
    model.fit(X_train, y_train)

    upload_pickle_to_s3('asdi-hackathon', model, 'pickles/popdensity_model.pkl')

def create_so2_pickle():
    client = boto3.client('s3')
    bucket='asdi-hackathon'
    file_key = 'aq-sentinel/sulphur-dioxide/so2_sentinel_01-16june.csv'
    obj = client.get_object(Bucket=bucket, Key=file_key)
    df = pd.read_csv(obj['Body'])

    df['sulfurdioxide_total_vertical_column'] = abs(df['sulfurdioxide_total_vertical_column'])

    df = df[df['qa_value'] >= 1]

    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)

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
        
def parallelise(df, func):
    #https://docs.python.org/3/library/multiprocessing.html
    #from multiprocessing import set_start_method
    #for Jupyter Notebook implementations:
    from multiprocess import set_start_method
    #set_start_method("spawn")
    #'fork' crashes process, a known issue with MacOS
    #gitignore of local csvs maybe causing problem with 'fork' start method
    set_start_method("fork")
    #set_start_method("forkserver")
    
    n_cores = cpu_count()
    df_splits = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    results = pool.map(func, df_splits)
    pool.close()
    pool.join()
    df = pd.concat(results)
    return df

def apply_aq_metric_functions(df):
    #molar mass constants
    co_molar_mass = 28.01
    no2_molar_mass = 46.0055
    o3_molar_mass = 48
    so2_molar_mass = 64.066

    #apply aq functions to each row (using latitude and longitude columns) and multiply by associated molar mass to give g/m2
    #axis = 1, apply function to each row
    
    df['Value_co'] = df.apply(lambda row : co_function(row['Latitude'], row['Longitude'], 'asdi-hackathon', 'pickles/co_model.pkl') * co_molar_mass, axis=1)
    print('co_function complete')
    df['Value_no2'] = df.apply(lambda row : no2_function(row['Latitude'], row['Longitude'], 'asdi-hackathon', 'pickles/no2_model.pkl') * no2_molar_mass, axis=1)
    print('no2_function complete')
    df['Value_o3'] = df.apply(lambda row : o3_function(row['Latitude'], row['Longitude'], 'asdi-hackathon', 'pickles/o3_model.pkl') * o3_molar_mass, axis=1)
    print('o3_function complete')
    df['Value_so2'] = df.apply(lambda row : so2_function(row['Latitude'], row['Longitude'], 'asdi-hackathon', 'pickles/so2_model.pkl') * so2_molar_mass, axis=1)
    print('so2_function complete')
    df['Value_ai'] = df.apply(lambda row : ai_function(row['Latitude'], row['Longitude'], 'asdi-hackathon', 'pickles/ai_model.pkl'), axis=1)
    print('ai_function complete')
    return df

def normalise_aq_metric_columns(df):
    norm_cols = ['Value_co', 'Value_no2', 'Value_o3', 'Value_so2', 'Value_ai']
    #normalise each aq metric value set between 1 and 0 where 0 = 0% and 1 = 20%
    for i in df[norm_cols]:   #normalise aq value columns
        df['norm_' + i]=(df[i]-df[i].min())/(df[i].max()-df[i].min())
    return df

def aq_score_function(aq1, aq2, aq3, aq4, aq5):
    #smaller value = better air quality
    aqs = (aq1 * (20/100)) + (aq2 * (20/100)) + (aq3 * (20/100)) + (aq4 * (20/100)) + (aq5 * (20/100))
    return aqs

def apply_aqs_function(df):
    #create normalised versions of aq metric columns in order to apply aq_score_function
    df = normalise_aq_metric_columns(df)
    
    #assumption: each metric is worth 20% of AQS, 100 / 5 metrics
    #5 air quality metrics. Values for each are normalised between 0 and 1 where each of the 5 metrics account for 20% (5 * 20% = 100%) of an air quality score. 
    #A value of 0 would mean the air quality value contributes nothing to the air quality score. 
    #A value of 1 means it contributes the full 20%. This is the MVP method for calculating an air quality score given blockers associated with the dataset, 
    #i.e. no access to near-surface air quality metrics, only a total vertical column. 
    # A lower air quality score reflects better air quality.
    #apply calculate_aqi function to each row of the 5 aq columns
    df['AQ_score'] = df.apply(lambda row : aq_score_function(row['norm_Value_co'], 
                                                            row['norm_Value_no2'], 
                                                            row['norm_Value_o3'], 
                                                            row['norm_Value_so2'], 
                                                            row['norm_Value_ai']), axis=1)
    
    #drop normalised columns
    df = df.drop(['norm_Value_co', 'norm_Value_no2', 'norm_Value_o3', 'norm_Value_so2', 'norm_Value_ai'], axis = 1)
    return df

def dist_nearest_greenspace_function(df):
    df_greenspace1 = df.loc[df.Green_Space == 1, :]
    df_greenspace1 = df_greenspace1[['Latitude', 'Longitude']]
    df_greenspace1 = df_greenspace1.apply(np.radians)

    df_greenspace0 = df.loc[df.Green_Space == 0, :]
    df_greenspace0 = df_greenspace0[['Latitude', 'Longitude']]
    df_greenspace0 = df_greenspace0.apply(np.radians)
    
    tree = BallTree(df_greenspace1, leaf_size=40, metric = 'haversine') 
    dist, ind = tree.query(df_greenspace0, k=1)
    
    distances = []
    for i in range(len(dist)):
        distances.append(dist[i][0])
        
    radius_earth = 6371
    distances_km = [item * radius_earth for item in distances]
    
    df_greenspace0 = df.loc[df.Green_Space == 0, :]
    df_greenspace0 = df_greenspace0.reset_index(drop = True)
    column_values = pd.Series(distances_km)
    df_greenspace0.insert(loc=8, column='Distance_Nearest_Greenspace', value=column_values)
    
    df_merged = pd.merge(df, df_greenspace0[['Latitude', 'Longitude', 'Distance_Nearest_Greenspace']], how="left", on=['Latitude', 'Longitude'])

    df_merged['Distance_Nearest_Greenspace'] = df_merged['Distance_Nearest_Greenspace'].replace(np.nan, 0.000000)
    
    return df_merged

def apply_popd_function(df):
    #same as above apply aq functions but with...
    #popdensity_function
    df['Pop_density'] = df.apply(lambda row : popdensity_function(row['Latitude'], row['Longitude'], 'asdi-hackathon', 'pickles/popdensity_model.pkl'), axis=1)
    return df

def calculate_popd_weight(df, resolution):
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6209905/
    #minimum of 9m2 per capita according to WHO 
    #50m2 per capita ideal according to WHO standards or 100m2 (our resolution) per 2 people
    standard_gs_per_pop_m2 = 50
    sum_df_popd = df['Pop_density'].sum()
    sum_df_greenspace_m2 = len(df[df['Green_Space'] == 1]) * resolution  #sum of greenspace multiplied by resolution (assumption greenspace covers entirety of 250m2 area which unlikely for all)
    gs_per_capita = sum_df_greenspace_m2 / sum_df_popd
    #if current greenspace per capita is BETTER than WHO standards, it is LESS likely greenspace is required so PENALISE with lower weighting
    #weight <1 will decrease contribution of pop density to greenspace score
    #if current greenspace per capita is WORSE than WHO standards, it is MORE likely greenspace is required so REWARD with higher weighting
    #weight >1 will increase contribution of pop density to greenspace score
    #weight =1 means weighting is essentially cancelled out
    popd_weight = standard_gs_per_pop_m2 / gs_per_capita   #a value >1 indicates failing to meet WHO standard, a value < 1 indicates beating the standard
    return popd_weight

def greenspace_score_function(aqs, pop_density, airport, water, building, green_space, railway_station, urban_area, dist_nearest_greenspace, popd_weight):
    #Population Density
    popd_pct = 25/100
    
    #Air Quality Score
    #aqs_pct derived from remainder of popd_weight * popd_pct so that AQ becomes focused more in greenspace score when population density less of a concern for greenspaces
    aqs_pct = (1 - (popd_weight * popd_pct))
    aqs_weight = 1
    
    #Distance from Nearest Greenspace (reward only based on magnitude distance in km)
    dist_nearest_greenspace += 1
    #all 0 values (i.e. currently a greenspace at coord) have 1 added to it so * 1 dist_nearest_greenspace has no effect
    #all values greater than 1 (i.e. currently NO greenspace at coord) have 1 added to it so * e.g. 1.5 (for 0.5 km distance) dist_nearest_greenspace acts as reward
    
    #Land Type
    ###############################
    if airport == 1:
        airport_weight = 0   #avg_penalty_reward = 0 means a reduction of the greenspace score to 0 (no greenspace permitted here)
    else:
        airport_weight = 1   #avg_penalty_reward = 1 means no reduction of the greenspace score (a greenspace is permitted here)
    ###############################
    if water == 1:
        water_weight = 0
    else:
        water_weight = 1
    ###############################
    if green_space == 1:
        green_space_weight = 0.5   #under assumption that while greenspace already exists in each 250m2 tile, that doesn't mean it is entirely greenspace, there could be an area of greenspace within the tile that could be expanded
        dist_nearest_greenspace = 1   #to avoid value from dist_nearest_greenspace contradicting green_space_weight, set dist_nearest_greenspace to 1, effectively cancelling it out from greenspace score calculation
    else:
        green_space_weight = 1.1   #small reward for no greenspace
    ###############################
    if railway_station == 1:
        railway_station_weight = 0
    else:
        railway_station_weight = 1
    ###############################
    if urban_area == 1:
        urban_area_weight = 1.25   #reward attributed to existence of urban area given assumption that urban areas probably already need greenspaces given pop density
    else:
        urban_area_weight = 1
    ###############################
    if building == 1:
        building_weight = 0.5   #due to inconvenience knocking down a building for a greenspace, a modest penalty
    else:
        building_weight = 1
    ###############################
    
    penalty_reward = airport_weight * water_weight * green_space_weight * railway_station_weight * urban_area_weight * building_weight * dist_nearest_greenspace
        
    Greenspace_score = ((aqs * (aqs_weight * aqs_pct)) + (pop_density * (popd_weight * popd_pct))) * penalty_reward
    
    return [Greenspace_score, penalty_reward]
    
def apply_greenspace_score_function(df, resolution):
    popd_weight = calculate_popd_weight(df, resolution)
    df[['Greenspace_score', 'penalty_reward']] = df.apply(lambda row : greenspace_score_function(row['AQ_score'], 
                                                                                row['Pop_density'],
                                                                                row['Airport'],
                                                                                row['Water'],
                                                                                row['Building'],
                                                                                row['Green_Space'],
                                                                                row['Railway_Station'],
                                                                                row['Urban_Area'],
                                                                                row['Distance_Nearest_Greenspace'], popd_weight), axis=1, result_type = 'expand')
    print('popd_weight = ', popd_weight)
    return df

def fill_points_land_type_df(bucket = '', key = ''):
    
    if bucket == '':
        #read locally
        df = pd.read_csv(ROOT_FOLDER_PATH + '/land_type_025.csv', index_col = 0)
    else:
        #read from s3 bucket
        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
    
    #preprocess land type columns, convert to binary from boolean
    for i in ['Airport', 'Water', 'Building', 'Green_Space', 'Railway_Station', 'Urban_Area']:
        df[i] = df[i].astype(int)
    
    #calculate distance to nearest greenspaces
    start = time()
    df = dist_nearest_greenspace_function(df)
    end = time()
    time_taken1 = round(end - start, 2)
    print('dist_nearest_greenspace_function complete')                      
    print('Time taken:', time_taken1)
    
    #predict air quality metric value at each coordinate using distance weighted knn models
    start = time()
    df = parallelise(df, apply_aq_metric_functions)
    end = time()
    time_taken2 = round(end - start, 2)
    print('apply_aq_metric_functions complete')                      
    print('Time taken:', time_taken2)
    
    #calculate an air quality score from aq metrics
    start = time()
    df = apply_aqs_function(df)
    print('apply_aqs_function complete')
    end = time()
    time_taken3 = round(end - start, 2)
    print('Time taken:', time_taken3)
    
    #predict population density metric value at each coordinate using distance weighted knn model
    start = time()
    df = parallelise(df, apply_popd_function)
    print('apply_popd_function complete')
    end = time()
    time_taken4 = round(end - start, 2)
    print('Time taken:', time_taken4)
    
    total_time_taken = time_taken1 + time_taken2 + time_taken3 + time_taken4
    print('TOTAL time taken:', total_time_taken)
    
    return df

def fill_penultimate_df(bucket = '', key = ''):
    
    if bucket == '':
        #read locally
        df = pd.read_csv(ROOT_FOLDER_PATH + '/penultimate_df.csv', index_col = 0)
    else:
        #read from s3 bucket
        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
    
    #calculate greenspace score from aq score, pop density and distance from nearest greenspace
    start = time()
    df = apply_greenspace_score_function(df, resolution = 250)
    end = time()
    time_taken4 = round(end - start, 2)
    print('apply_greenspace_score_function complete')
    print('Time taken:', time_taken4)
    
    return df

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

def create_final_df():
    penultimate_df = fill_points_land_type_df('asdi-hackathon', 'land_type_025.csv')
        
    upload_df_to_s3(bucket = 'asdi-hackathon', df = penultimate_df, key = 'final-data/penultimate_df.csv')

    final_df = fill_penultimate_df('asdi-hackathon', 'final-data/penultimate_df.csv')

    upload_df_to_s3(bucket = 'asdi-hackathon', df = final_df, key = 'final-data/final_df.csv')

    return final_df

if __name__ == "__main__":
    create_ai_pickle()
    print('create_ai_pickle complete')
    create_co_pickle()
    print('create_co_pickle complete')
    create_no2_pickle()
    print('create_no2_pickle complete')
    create_o3_pickle()
    print('create_o3_pickle complete')
    create_popdensity_pickle()
    print('create_popdensity_pickle complete')
    create_so2_pickle()
    print('create_so2_pickle complete')
    
    create_final_df()
    print('create_final_df complete')