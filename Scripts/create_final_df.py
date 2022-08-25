from helpers import *

penultimate_df = fill_points_land_type_df('asdi-hackathon', 'land_type_025.csv')
    
upload_df_to_s3(bucket = 'asdi-hackathon', df = penultimate_df, key = 'final-data/penultimate_df.csv')

final_df = fill_penultimate_df('asdi-hackathon', 'final-data/penultimate_df.csv')

upload_df_to_s3(bucket = 'asdi-hackathon', df = final_df, key = 'final-data/final_df.csv')