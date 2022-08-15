import pickle
import pandas as pd
import math
import pathlib
from haversine import *
from tqdm import tqdm

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