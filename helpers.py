import pickle
import pandas as pd
import math
import pathlib

#ROOT = pathlib.Path().absolute().parent
#import os
#ROOT = pathlib.Path(os.getcwd())
ROOT = '/Users/joshua.grefte/Projects/ASDI/Local_Repo/ASDI-Hackathon/'
PICKLE_FOLDER_PATH = ROOT + 'Pickles/'

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