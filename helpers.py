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
#from multiprocessing import Pool
#from multiprocessing import cpu_count
#for Jupyter Notebooks parallel processing
#multiprocess works better within Jupyter Notebooks than multiprocessing package
from multiprocess import Pool
from multiprocess import cpu_count

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

# def aqi_function_og(lat, lon):
#     co_value = co_function(lat, lon)
#     no2_value = no2_function(lat, lon)
#     o3_value = o3_function(lat, lon)
#     so2_value = so2_function(lat, lon)
    
#     #data arrives as unit mol/m^2 (moles per metre squared)
#     #moles to g by multiplying by molar mass of molecule
#     co_molar_mass = 28.01
#     no2_molar_mass = 46.0055
#     o3_molar_mass = 48
#     so2_molar_mass = 64.066
    
#     #need conversion to 3d concentration unit, satellite data is 2d scan of Earth so arrives as m-2
#     #xarray data details says its mol/m-2 but the documentation says it is in fact mol/cm-2; 
#     #going to go with xarray data details as documentation maybe outdated - assume m-2
#     #all are total vertical column, assuming uniform distribution, divide by height of atmosphere recorded (49500m) to give a natural concentration for entire atmosphere
    
#     #1 g/m^3 = 1 ppm
#     #1 ppm = 1000 ppb therefore * 1000 for ppm --> ppb conversion
#     #standard units for o3 = ppm, co = ppm, so2 = ppb and no2 = ppb (according to EPA reference tables)
    
#     #have to assume hourly as satellite takes snapshot lasting less than an hour
#     #but CO only has 8 hour EPA data available (artifact of the gas itself)
    
#     #data arriving as total vertical column, i.e. number of molecules viewable from perspective above Earth of column
#     #i.e. if column cylindrical, the number of molecules viewable from a planar view of top circle
#     #some molecules may be behind others so not accurate as is
#     #need to use atmospheric model of different layers of Earth's atmosphere (like air pressure) to possibly derive near-surface value
#     #as one moves up atmospheric levels, the density of the atmosphere reduces...
#     #can use data accompanying NO2 value and assume similar to e.g. CO but there maybe slight differences depending on e.g.
#     #penetratability of a CO molecule to higher levels of atmosphere if more/less dense than surrounding molecules 
    
#     #alternative is to use something like GAMs to model a combination of reference table BP ranges into one
    
#     #basic assumption
#     #no conversion m-2 to m-3, m-2 = m-3
#     co_converted = (co_value * co_molar_mass)
#     no2_converted = ((no2_value * no2_molar_mass)) * 1000
#     o3_converted = (o3_value * o3_molar_mass)
#     so2_converted = ((so2_value * so2_molar_mass)) * 1000
    
#     aq_metric_converted = [co_converted, no2_converted, o3_converted, so2_converted]

#     #need to calculate AQI of each pollutant separately
#     #the lowest AQI value of the pollutants is considered the real AQI value
    
#     #reference tables
#     #source: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
#     co_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
#                              'IHi': [50, 100, 150, 200, 300, 500], 
#                              #8 hour
#                              'BPLo': [0, 4.5, 9.5, 12.5, 15.5, 30.5], 
#                              'BPHi': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]})
    
#     no2_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
#                               'IHi': [50, 100, 150, 200, 300, 500], 
#                               #1 hour
#                               'BPLo': [0, 54, 101, 361, 650, 1250], 
#                               'BPHi': [53, 100, 360, 649, 1249, 2049]})
    
#     o3_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
#                              'IHi': [50, 100, 150, 200, 300, 500], 
#                              #1 hour, interpolated first 2 as no data for 1 hour (only 8 hour version)
#                              'BPLo': [0, 0.0626, 0.126, 0.165, 0.205, 0.405], 
#                              'BPHi': [0.0625, 0.125, 0.164, 0.204, 0.404, 0.604]})
    
#     so2_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
#                               'IHi': [50, 100, 150, 200, 300, 500], 
#                               #1 hour
#                               'BPLo': [0, 36, 76, 186, 305, 605], 
#                               'BPHi': [35, 75, 185, 304, 604, 1004]})
    
    
#     aqi_tables = [co_table, no2_table, o3_table, so2_table]
    
#     #manual calculation of AQI inc referencing EPA air quality standards tables
#     #Cp = truncated concentration of pollutant p
#     #BPHi = concentration breakpoint i.e. greater than or equal to Cp or upper bound on cp range
#     #BPLo = concentration breakpoint i.e. less than or equal to Cp or lower bound on cp range
#     #IHi = AQI value corresponding to BPHi, i.e. upper bound on aqi range
#     #ILo = AQI value corresponding to BPLo, i.e. lower bound on aqi range
    
#     aqi_list = []
#     labels = ['CO AQI -> ', 'NO2 AQI -> ', 'O3 AQI -> ', 'SO2 AQI -> ']
#     aqi_category_dict = {'Good': [0, 50], 
#                             'Moderate': [51, 100], 
#                             'Unhealthy for Sensitive Groups': [101, 150], 
#                             'Unhealthy': [151, 200], 
#                             'Very Unhealthy': [201, 300], 
#                             'Hazardous': [301, 500]}
#     for concentration, table, label in zip(aq_metric_converted, aqi_tables, labels):
#         Cp = concentration
#         BPHi = 'Out of Bounds'
#         BPLo = 'Out of Bounds'
#         for index, row in table.iterrows():
#             if concentration <= table['BPHi'].iloc[index] and concentration >= table['BPLo'].iloc[index]:
#                 BPHi = table['BPHi'].iloc[index]
#                 BPLo = table['BPLo'].iloc[index]
#             else:
#                 continue
#         IHi = 'Out of bounds'
#         ILo = 'Out of Bounds'
#         for index, row in table.iterrows():
#             if concentration <= table['IHi'].iloc[index] and concentration >= table['ILo'].iloc[index]:
#                 IHi = table['IHi'].iloc[index]
#                 ILo = table['ILo'].iloc[index]
#             else:
#                 continue
#         try:
#             aq_index = (IHi - ILo / BPHi - BPLo) * (Cp - BPLo) + ILo
#         except:
#             print(label, 'Values(s) out of bounds ->', 'IHi:', IHi, 'ILo:', ILo, 'BPHi:', BPHi, 'BPLo:', BPLo)
        
#         aqi_category = ''
#         for key in aqi_category_dict:
#                 if (aq_index < max(aqi_category_dict[key])) and (aq_index > min(aqi_category_dict[key])):
#                     aqi_category = key
#         if aqi_category == '':
#             aqi_category = 'Out of Bounds'
                
#         print(label + aqi_category + ' -> ' + str(aq_index))

#         aqi_list.append(aq_index)
    
#     print('\n')
#     for key in aqi_category_dict:
#         print(key, aqi_category_dict[key])
    
#     return max(aqi_list)

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

def apply_aq_metric_functions(df):
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

def apply_popd_function(df):
    #same as above apply aq functions but with...
    #popdensity_function
    df['Pop_density'] = df.apply(lambda row : popdensity_function(row['Latitude'], row['Longitude']), axis=1)
    return df

def calculate_popd_weight(df, resolution):
    #50m2 per capita according to WHO standards or 100m2 (our resolution) per 2 people
    standard_gs_per_pop_m2 = 50
    sum_df_popd = df['Pop_density'].sum()
    sum_df_greenspace_m2 = len(df[df['Green_Space'] == 1]) * resolution   #sum of greenspace multiplied by resolution
    gs_per_capita = sum_df_greenspace_m2 / sum_df_popd
    #if current greenspace per capita is BETTER than WHO standards, it is LESS likely greenspace is required so PENALISE lower weighting
    #weight <1 will decrease contribution of pop density to greenspace score
    #if current greenspace per capita is WORSE than WHO standards, it is MORE likely greenspace is required so REWARD higher weighting
    #weight >1 will increase contribution of pop density to greenspace score
    #weight =1 means weighting is essentially cancelled out
    popd_weight = gs_per_capita / standard_gs_per_pop_m2
    return popd_weight

def greenspace_score_function(aqs, pop_density, airport, water, building, green_space, railway_station, urban_area, popd_weight):
    #Population Density
    popd_pct = 50/100
    
    #Air Quality Score
    aqs_pct = (1 - (popd_weight * popd_pct))
    aqs_weight = 1
    
    #Land Type
    penalty_reward_row_sum = 0
    ###############################
    if airport == 1:
        penalty_reward_row_sum += 0   #avg_penalty_reward = 0 means a reduction of the greenspace score to 0 (no greenspace permitted here)
    else:
        penalty_reward_row_sum += 1   #avg_penalty_reward = 1 means no reduction of the greenspace score (a greenspace is permitted here)
    ###############################
    if water == 1:
        penalty_reward_row_sum += 0
    else:
        penalty_reward_row_sum += 1
    ###############################
    if green_space == 1:
        penalty_reward_row_sum += 0.5   #under assumption that while greenspace already exists in each 250m2 tile, that doesn't mean it is entirely greenspace, there could be an area of greenspace within the tile that could be expanded
    else:
        penalty_reward_row_sum += 1
    ###############################
    if railway_station == 1:
        penalty_reward_row_sum += 0
    else:
        penalty_reward_row_sum += 1
    ###############################
    if building == 1:
        penalty_reward_row_sum += 0.25   #assuming building refers to more key buildings rather than less key urban area buildings, the inconvenience of replacing these with greenspaces is so high, more penalty should be attributed
    else:
        penalty_reward_row_sum += 1
    ###############################
    if urban_area == 1:
        penalty_reward_row_sum += 1.25   #reward attributed to existence of urban area given assumption that urban areas probably already need greenspaces given pop density
    else:
        penalty_reward_row_sum += 1
    ###############################
    avg_penalty_reward = penalty_reward_row_sum / 6
        
    Greenspace_score = (aqs * (aqs_weight * aqs_pct)) + (pop_density * (popd_weight * popd_pct)) * avg_penalty_reward
    return [Greenspace_score, avg_penalty_reward]
    
def apply_greenspace_score_function(df, resolution):
    popd_weight = calculate_popd_weight(df, resolution)
    df[['Greenspace_score', 'avg_penalty_reward']] = df.apply(lambda row : greenspace_score_function(row['AQ_score'], 
                                                                                row['Pop_density'],
                                                                                row['Airport'],
                                                                                row['Water'],
                                                                                row['Building'],
                                                                                row['Green_Space'],
                                                                                row['Railway_Station'],
                                                                                row['Urban_Area'], popd_weight), axis=1, result_type = 'expand')
    print('popd_weight = ', popd_weight)
    return df

def fill_points_land_type_df(bucket = '', key = ''):
    
    if bucket == '':
        #read locally
        df = pd.read_csv(ROOT_FOLDER_PATH + '/Spikes/Dash/data/land_type_025.csv', index_col = 0)
    else:
        #read from s3 bucket
        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
    
    #preprocess land type columns, convert to binary from boolean
    for i in ['Airport', 'Water', 'Building', 'Green_Space', 'Railway_Station', 'Urban_Area']:
        df[i] = df[i].astype(int)
    
    start = time.time()
    df = parallelise(df, apply_aq_metric_functions)
    end = time.time()
    time_taken1 = round(end - start, 2)
    print('apply_aq_metric_functions complete')                      
    print('Time taken:', time_taken1)
    
    start = time.time()
    df = apply_aqs_function(df)
    print('apply_aqs_function complete')
    end = time.time()
    time_taken2 = round(end - start, 2)
    print('Time taken:', time_taken2)
    
    start = time.time()
    df = parallelise(df, apply_popd_function)
    print('apply_popd_function complete')
    end = time.time()
    time_taken3 = round(end - start, 2)
    print('Time taken:', time_taken3)
    
    total_time_taken = time_taken1 + time_taken2 + time_taken3
    print('TOTAL time taken:', total_time_taken)
    
    return df

def fill_penultimate_df(bucket = '', key = ''):
    
    if bucket == '':
        #read locally
        df = pd.read_csv(ROOT_FOLDER_PATH + '/Spikes/Dash/data/penultimate_df.csv', index_col = 0)
    else:
        #read from s3 bucket
        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        
    start = time.time()
    df = apply_greenspace_score_function(df, resolution = 250)
    end = time.time()
    time_taken4 = round(end - start, 2)
    print('apply_greenspace_score_function complete')
    print('Time taken:', time_taken4)
    
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
