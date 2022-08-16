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
    
    #need conversion to 3d concentration unit, satellite data is 2d scan of Earth so arrives as m-2
    #xarray data details says its mol/m-2 but the documentation says it is in fact mol/cm-2; 
    #going to go with xarray data details as documentation maybe outdated - assume m-2
    #all are total vertical column, assuming uniform distribution, divide by height of atmosphere recorded (49500m) to give a natural concentration for entire atmosphere
    
    #1 g/m^3 = 1 ppm
    #1 ppm = 1000 ppb therefore * 1000 for ppm --> ppb conversion
    #standard units for o3 = ppm, co = ppm, so2 = ppb and no2 = ppb (according to EPA reference tables)
    
    #have to assume hourly as satellite takes snapshot lasting less than an hour
    #but CO only has 8 hour EPA data available (artifact of the gas itself)
    
    # co_converted = (co_value * co_molar_mass) * 1*(10**-6) * 49500
    # no2_converted = ((no2_value * no2_molar_mass) * 1*(10**-6) * 49500) * 1000
    # o3_converted = (o3_value * o3_molar_mass) * 1*(10**-6) * 49500
    # so2_converted = ((so2_value * so2_molar_mass) * 1*(10**-6) * 49500) * 1000
    
    #no conversion m-2 to m-3, m-2 = m-3
    co_converted = (co_value * co_molar_mass)
    no2_converted = ((no2_value * no2_molar_mass)) * 1000
    o3_converted = (o3_value * o3_molar_mass)
    so2_converted = ((so2_value * so2_molar_mass)) * 1000
    
    aq_metric_converted = [co_converted, no2_converted, o3_converted, so2_converted]

    #need to calculate AQI of each pollutant separately
    #the lowest AQI value of the pollutants is considered the real AQI value
    
    #reference tables
    #source: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    co_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
                             'IHi': [50, 100, 150, 200, 300, 500], 
                             #8 hour
                             'BPLo': [0, 4.5, 9.5, 12.5, 15.5, 30.5], 
                             'BPHi': [4.4, 9.4, 12.4, 15.4, 30.4, 50.4]})
    
    no2_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
                              'IHi': [50, 100, 150, 200, 300, 500], 
                              #1 hour
                              'BPLo': [0, 54, 101, 361, 650, 1250], 
                              'BPHi': [53, 100, 360, 649, 1249, 2049]})
    
    o3_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
                             'IHi': [50, 100, 150, 200, 300, 500], 
                             #1 hour, interpolated first 2 as no data for 1 hour (only 8 hour version)
                             'BPLo': [0, 0.0626, 0.126, 0.165, 0.205, 0.405], 
                             'BPHi': [0.0625, 0.125, 0.164, 0.204, 0.404, 0.604]})
    
    so2_table = pd.DataFrame({'ILo': [0, 51, 101, 151, 201, 301], 
                              'IHi': [50, 100, 150, 200, 300, 500], 
                              #1 hour
                              'BPLo': [0, 36, 76, 186, 305, 605], 
                              'BPHi': [35, 75, 185, 304, 604, 1004]})
    
    
    aqi_tables = [co_table, no2_table, o3_table, so2_table]
    
    #manual calculation of AQI inc referencing EPA air quality standards tables
    #Cp = truncated concentration of pollutant p
    #BPHi = concentration breakpoint i.e. greater than or equal to Cp or upper bound on cp range
    #BPLo = concentration breakpoint i.e. less than or equal to Cp or lower bound on cp range
    #IHi = AQI value corresponding to BPHi, i.e. upper bound on aqi range
    #ILo = AQI value corresponding to BPLo, i.e. lower bound on aqi range
    
    aqi_list = []
    labels = ['CO AQI -> ', 'NO2 AQI -> ', 'O3 AQI -> ', 'SO2 AQI -> ']
    aqi_category_dict = {'Good': [0, 50], 
                            'Moderate': [51, 100], 
                            'Unhealthy for Sensitive Groups': [101, 150], 
                            'Unhealthy': [151, 200], 
                            'Very Unhealthy': [201, 300], 
                            'Hazardous': [301, 500]}
    for concentration, table, label in zip(aq_metric_converted, aqi_tables, labels):
        Cp = concentration
        for index, row in table.iterrows():
            if (concentration < table['BPHi'][index]) and (concentration > table['BPLo'][index]):
                BPHi = table['BPHi'][index]
                BPLo = table['BPLo'][index]
            else:
                continue
        for index, row in table.iterrows():
            if (concentration < table['IHi'][index]) and (concentration > table['ILo'][index]):
                IHi = table['IHi'][index]
                ILo = table['ILo'][index]
            else:
                continue
        
        try:
            aq_index = (IHi - ILo / BPHi - BPLo) * (Cp - BPLo) + ILo
        except:
            print('Values out of bounds')
            print(IHi, ILo, BPHi, BPLo)
        
        aqi_category = ''
        for key in aqi_category_dict:
                if (aq_index < max(aqi_category_dict[key])) and (aq_index > min(aqi_category_dict[key])):
                    aqi_category = key
        if aqi_category == '':
            aqi_category = 'Out of Bounds'
                
        print(label + aqi_category + ' -> ' + str(aq_index))

        aqi_list.append(aq_index)
    
    print('\n')
    for key in aqi_category_dict:
        print(key, aqi_category_dict[key])
    
    return max(aqi_list)