import pickle
import pandas as pd
import math
import pathlib
from haversine import *
from tqdm import tqdm
from Enums.land_type import LAND_TYPE
import requests
from time import time


ROOT_FOLDER_PATH = pathlib.Path().absolute().parent.as_posix()
PICKLE_FOLDER_PATH = ROOT_FOLDER_PATH + '/Pickles/'

def co_function(lat, lon):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    #load model from pickle
    co_model = pickle.load(open(PICKLE_FOLDER_PATH + 'co_model.pkl', 'rb'))
    
    #predict with model
    preds = co_model.predict(input)
    return preds[0]

def no2_function(lat, lon):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    #load model from pickle
    no2_model = pickle.load(open(PICKLE_FOLDER_PATH + 'no2_model.pkl', 'rb'))
    
    #predict with model
    preds = no2_model.predict(input)
    return preds[0]

def o3_function(lat, lon):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    #load model from pickle
    o3_model = pickle.load(open(PICKLE_FOLDER_PATH + 'o3_model.pkl', 'rb'))
    
    #predict with model
    preds = o3_model.predict(input)
    return preds[0]

def so2_function(lat, lon):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    #load model from pickle
    so2_model = pickle.load(open(PICKLE_FOLDER_PATH + 'so2_model.pkl', 'rb'))
    
    #predict with model
    preds = so2_model.predict(input)
    return preds[0]

def popdensity_function(lat, lon):
    #preprocessing, convert lat/lon to radians
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    #load model from pickle
    popdensity_model = pickle.load(open(PICKLE_FOLDER_PATH + 'popdensity_model.pkl', 'rb'))
    
    #predict with model
    preds = popdensity_model.predict(input)
    return preds[0]
  
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
        d (float): the diameter of the circular space around a point in km
        bottom_left (float tuple): lat/long
        topright (float tuple): lat/long

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

def get_land_type(latitude, longitude, resolution_diameter, API_key):
    
    bbox = get_bbox_of_point(latitude, longitude, resolution_diameter)
    
    land_types = []
    
    if is_airport(bbox, API_key):
        land_types.append(LAND_TYPE.AIRPORT.value)
    if is_water(bbox, API_key):
        land_types.append(LAND_TYPE.WATER.value)
    if is_building(bbox, API_key):
        land_types.append(LAND_TYPE.BUILDING.value)
    if is_railway_station(bbox, API_key):
        land_types.append(LAND_TYPE.RAILWAYSTATION.value)
    if is_green_space(bbox, API_key):
        land_types.append(LAND_TYPE.GREENSPACE.value)
    if is_urban_area(bbox, API_key):
        land_types.append(LAND_TYPE.URBANAREA.value)

    return ', '.join(land_types)
    
    
def get_bbox_of_point(latitude, longitude, resolution_diameter):
    """ Given a point get the bbox around that point at the given resolution

    Args:
        latitude (flaot): _description_
        longitude (float): _description_
        resolution_diameter (float): This corrosponds to the value used to generate the 2D point array 
    
    Returns:
        bounding box (string): formatted like so bbox_bottom_left_lat,bbox_bottom_left_long,bbox_top_right_lat,bbox_top_right_long
    """
    
    radius = resolution_diameter / 2
    hypot = radius / math.cos(math.radians(45))
    
    bottom_left = inverse_haversine((latitude, longitude), hypot, Direction.SOUTHWEST)
    top_right = inverse_haversine((latitude, longitude), hypot, Direction.NORTHEAST)
    
    return f'{bottom_left[0]:.8f}' + ',' + f'{bottom_left[1]:.8f}' + ',' + f'{top_right[0]:.8f}' + ',' + f'{top_right[1]:.8f}'
    

def get_feature_type_in_bbox(bbox, feature_type, API_key):
        
    wfs_endpoint = ('https://api.os.uk/features/v1/wfs?')

    service = 'wfs'
    request = 'GetFeature'
    version = '2.0.0'
    typeNames = feature_type
    outputFormat = 'GEOJSON'
    OS_DATA_HUB_API_KEY = API_key

    params_wfs = {'service':service,
                  'key': OS_DATA_HUB_API_KEY,
                  'request':request,
                  'version':version,
                  'typeNames':typeNames,
                  'outputFormat':outputFormat,
                  'bbox': bbox,
                 }

    try:
        r = requests.get(wfs_endpoint, params=params_wfs)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(e)
    
    if r.status_code == 200:
        payload = r.json()
    elif r.status_code == 229:
        # Wait a minute and 5 seconds
        t0 = time()
        while(time() - t0 < 2):
            continue
        # try again
        try:
            r = requests.get(wfs_endpoint, params=params_wfs)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(e)
        payload = r.json()
        
        if r.status_code == 229:
            print('Not long enough')
            print(r.status_code)
            return 'Error - '+str(r.status_code)
        elif r.status_code != 200:
            print(r.status_code)
            return 'Error - '+str(r.status_code)
        # THIS NEEDS TO BE REFACTORED
    else:
        print(r.status_code)
        return 'Error - '+str(r.status_code)
    
    return payload
    
    
def is_airport(bbox, API_key):
    result = get_feature_type_in_bbox(bbox, 'Zoomstack_Airports', API_key)
    if (isinstance(result, str)):
        return False
    elif len(result['features']) > 0:
        return True
    else:
        return False

def is_water(bbox, API_key):
    result = get_feature_type_in_bbox(bbox, 'Zoomstack_Surfacewater', API_key)
    if (isinstance(result, str)):
        return False
    elif len(result['features']) > 0:
        return True
    else:
        return False

def is_building(bbox, API_key):
    result_local = get_feature_type_in_bbox(bbox, 'Zoomstack_LocalBuildings', API_key)
    result_district = get_feature_type_in_bbox(bbox, 'Zoomstack_DistrictBuildings', API_key)
    
    if (isinstance(result_local, str)):
        return False
    elif (isinstance(result_district, str)):
        return False
    elif len(result_local['features']) > 0:
        return True
    elif len(result_district['features']) > 0:
        return True
    else:
        return False
    
def is_green_space(bbox, API_key):
    result_Greenspace = get_feature_type_in_bbox(bbox, 'Zoomstack_Greenspace', API_key)
    result_NationalParks = get_feature_type_in_bbox(bbox, 'Zoomstack_NationalParks', API_key)
    result_Woodland = get_feature_type_in_bbox(bbox, 'Zoomstack_Woodland', API_key)
    
    if (isinstance(result_Greenspace, str)):
        return False
    elif (isinstance(result_NationalParks, str)):
        return False
    elif (isinstance(result_Woodland, str)):
        return False
    elif len(result_Greenspace['features']) > 0:
        return True
    elif len(result_NationalParks['features']) > 0:
        return True
    elif len(result_Woodland['features']) > 0:
        return True
    else:
        return False

def is_railway_station(bbox, API_key):
    result = get_feature_type_in_bbox(bbox, 'Zoomstack_RailwayStations', API_key)
    if (isinstance(result, str)):
        return False
    elif len(result['features']) > 0:
        return True
    else:
        return False

def is_urban_area(bbox, API_key):
    result = get_feature_type_in_bbox(bbox, 'Zoomstack_UrbanAreas', API_key)
    if (isinstance(result, str)):
        return False
    elif len(result['features']) > 0:
        return True
    else:
        return False

def get_land_types_for_points_in_csv(csv_path, save_path, start_point_index, end_point_index, diameter_resolution, API_key):
  # Open CSV
  points_df = pd.read_csv(csv_path)
  # Get correct set of points
  subset_points_df = points_df.loc[start_point_index:end_point_index]
  
  print('Number of points to be processed:', len(subset_points_df))
  print('Start/End index (inclusive):', start_point_index, end_point_index)
  print('Start point:', subset_points_df.iloc[0]['Latitude'], ',', points_df.iloc[0]['Longitude'])
  print('End point:', points_df.iloc[-1]['Latitude'], ',', points_df.iloc[-1]['Longitude'])
  
  
  land_type_list = []

  t0 = time()
  for i in tqdm(range(len(subset_points_df))):
    if i % 10 == 0 and i != 0:
      print('Saving and waiting...')
      # Create or append to a csv while waiting
      df = pd.DataFrame(dtype='object', index=subset_points_df.index[:len(land_type_list)])
      df['Land_Type'] = land_type_list
      df.to_csv(save_path)
      print('Last batch of points:' , land_type_list)
      while(time() - t0 < 2):
        continue
      t0 = time() # reset the timer
    
    land_type = get_land_type(subset_points_df.iloc[i]['Latitude'],
                              subset_points_df.iloc[i]['Longitude'],
                              diameter_resolution,
                              API_key)
    land_type_list.append(land_type)
  
  print('Saving...')
  df = pd.DataFrame(dtype='object', index=subset_points_df.index)
  df['Land_Type'] = land_type_list
  df.to_csv(save_path)
  print('Done.')
  