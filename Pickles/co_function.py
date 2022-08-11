def co_function(lat, lon):
    #preprocessing, convert lat/lon to radians
    import pandas as pd
    df = pd.DataFrame({'latitude': lat, 'longitude': lon}, index=[0])
    import math
    df['latitude'] = df['latitude'].apply(math.radians)
    df['longitude'] = df['longitude'].apply(math.radians)
    input = df[['latitude', 'longitude']]
    
    #load model from pickle
    import pickle
    co_model = pickle.load(open('co_model.pkl', 'rb'))
    
    #predict with model
    preds = co_model.predict(input)
    return preds