import sys
import pathlib

ROOT = pathlib.Path().absolute().parent.as_posix()
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from helpers import *

points_df = get_spaced_point_set_in_bbox(0.25, (51.239405, -0.625211), (51.737184, 0.328289))

upload_df_to_s3('asdi-hackathon', points_df, 'points_df.csv')