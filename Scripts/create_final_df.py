import sys
import pathlib
import pandas as pd

ROOT = pathlib.Path().absolute().parent.as_posix()
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from helpers import *

final_df = fill_points_df()

upload_df_to_s3(bucket = 'asdi-hackathon', df = final_df, key = 'final-data/final_df.csv')