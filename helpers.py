import pickle
import pandas as pd
import math
import pathlib

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

def aqi_function(lat, lon):
    co_value = co_function(lat, lon)
    no2_value = no2_function(lat, lon)
    o3_value = o3_function(lat, lon)
    so2_value = so2_function(lat, lon)
    
    #data arrives as unit mol/m^2 (moles per metre squared)
    #moles to g by multiplying by molar mass of molecule
    co_molar_mass = 28.01
    no2_molar_mass = 46.0055
    o3_molar_mass = 48
    so2_molar_mass = 64.066
    
    #need conversion to 3d concentration unit, multiply metre squared by a thickness to get metre cubed (volume)
    #as satellite data is 2d scan of Earth, no thickness derivable
    #will therefore assume m^2 = m^3
    #1 g/m^3 = 1 ppm
    #1 ppm = 1000 ppb therefore * 1000 for ppm --> ppb conversion
    #multiply by 1 * 10^6 to convert g to microgram (ug)
    #xarray data details says its mol/m-2 but the documentation says it is in fact mol/cm-2; 
    #...the values look off after conversion (out by an order of magnitude) so going to assume it is mol/cm-2 and make the relevant additional conversions
    #...therefore / 10000 for cm-2 to m-2
    thickness = 1
    co_converted = ((co_value * co_molar_mass) * thickness) * 1*(10**6) / 10000
    no2_converted = ((no2_value * no2_molar_mass) * thickness) * 1*(10**6) / 10000
    o3_converted = ((o3_value * o3_molar_mass) * thickness) * 1*(10**6) / 10000
    so2_converted = ((so2_value * so2_molar_mass) * thickness) * 1*(10**6) / 10000
    
    aq_metric_converted = [co_converted, no2_converted, o3_converted, so2_converted]
    
    print(aq_metric_converted)

    #need to calculate AQI of each pollutant separately
    #the lowest AQI value of the pollutants is considered the real AQI value
    
    #manual calculation of AQI
    #Cp = truncated concentration of pollutant p
    #BPHi = concentration breakpoint i.e. greater than or equal to Cp
    #BPLo = concentration breakpoint i.e. less than or equal to Cp
    #IHi = AQI value corresponding to BPHi
    #ILo = AQI value corresponding to BPLo
    #aqi_list = []
    #for i in aq_metric_converted:
        #Cp = 
        #BPHi = 
        #BPLo = 
        #IHi = 
        #ILo = 
        
        #aq_index = (IHi - ILo / BPHi - BPLo) * (Cp - BPLo) + ILo
        #aqi_list.append(aq_index)
        
    #return min(aqi_list)
    
    #external Python package to calculate (I)AQI
    #https://github.com/hrbonz/python-aqi
    #aqi.algos.epa: pm10 (µg/m³), o3_8h (ppm), co_8h (ppm), no2_1h (ppb), o3_1h (ppm), so2_1h (ppb), pm25 (µg/m³)
    #aqi.algos.mep: no2_24h (µg/m³), so2_24h (µg/m³), no2_1h (µg/m³), pm10 (µg/m³), o3_1h (µg/m³), o3_8h (µg/m³), so2_1h (µg/m³), co_1h (mg/m³), pm25 (µg/m³), co_24h (mg/m³)
    # import aqi
    # aqi_list = []
    # pollutant_constant_list = [aqi.POLLUTANT_CO_8H, aqi.POLLUTANT_NO2_1H, aqi.POLLUTANT_O3_1H, aqi.POLLUTANT_SO2_1H]
    # for i, c in zip(aq_metric_converted, pollutant_constant_list):
    #     aq_index = aqi.to_aqi(c, str(i))   #default algo is EPA
    #     aqi_list.append(aq_index)
        
    # return min(aqi_list)

#https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info&pollutant=ozone#pollutant
co_table = pd.DataFrame({'US AQI Lower': [0, 51, 101, 151, 201, 301], 'US AQI Higher': [50, 100, 150, 200, 300, 500], 'US EPA Cp Range Lower': [0, 4.5, 9.5, 12.5, 15.5, 30.5], 'US EPA Cp Range Higher': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]})
no2_table = pd.DataFrame({'US AQI Lower': [0, 51, 101, 151, 201, 301], 'US AQI Higher': [50, 100, 150, 200, 300, 500], 'US EPA Cp Range Lower': [0, 4.5, 9.5, 12.5, 15.5, 30.5], 'US EPA Cp Range Higher': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]})
so2_table = pd.DataFrame({'US AQI Lower': [0, 51, 101, 151, 201, 301], 'US AQI Higher': [50, 100, 150, 200, 300, 500], 'US EPA Cp Range Lower': [0, 4.5, 9.5, 12.5, 15.5, 30.5], 'US EPA Cp Range Higher': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]})
o3_table = pd.DataFrame({'US AQI Lower': [0, 51, 101, 151, 201, 301], 'US AQI Higher': [50, 100, 150, 200, 300, 500], 'US EPA Cp Range Lower': [0, 4.5, 9.5, 12.5, 15.5, 30.5], 'US EPA Cp Range Higher': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]})