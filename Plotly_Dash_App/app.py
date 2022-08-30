# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import boto3
import numpy as np

import sys
import pathlib
import pandas as pd

ROOT = pathlib.Path().absolute().parent.as_posix()
if ROOT not in sys.path:
    sys.path.append(ROOT)


app = Dash(__name__)

#import data CSV locally
df = pd.read_csv(ROOT + '/final_df.csv', engine = 'c')

#import data CSV from s3 bucket
#client = boto3.client('s3')
#obj = client.get_object(Bucket='asdi-hackathon', Key='final-data/final_df.csv')
#df = pd.read_csv(obj['Body'])

df = df.rename(columns={"Greenspace_score": "Green Space Score", 
                   "AQ_score": "Air Quality (AQ) Metric", 
                   "Pop_density": "Population Density", 
                   "Distance_Nearest_Greenspace": "Distance from Nearest Green Space (km)", 
                   "Value_co": "AQ Carbon Monoxide", 
                   "Value_no2": "AQ Nitrogen Dioxide", 
                   "Value_o3": "AQ Ozone", 
                   "Value_so2": "AQ Sulphur Dioxide", 
                   "Value_ai": "AQ Aerosol Index", 
                   "Green_Space": "Current Green Spaces", 
                   "Building": "Buildings", 
                   "Airport": "Airports", 
                   "Water": "Water Bodies", 
                   "Railway_Station": "Railway Stations", 
                   "penalty_reward": "Land Type Penalty (<1) or Reward (>1)", 
                   "Urban_Area": "Urban Areas"})

app.layout = html.Div(children=[
    
    html.H1(children='Green Space Suggestion Dashboard', style = {'font-family': 'Arial', 
                                                                  'textAlign': 'center', 
                                                                  #'background-color': 'limegreen', 
                                                                  'color': 'black', 
                                                                  'font-size': '40px',
                                                                  #'border-bottom-color': 'limegreen',
                                                                  #'border-bottom-style': 'solid',
                                                                  'border-bottom': '4px solid limegreen',
                                                                  'padding-bottom': '0.5em'}),

    html.Div([
        
        html.Div(id='output_data'),
    
        html.Label('Select Overlay', style = {'font-family': 'Arial', 'font-size': '14px', 'padding': '2px'}),
        
        dcc.Dropdown(id='my_dropdown',
            options=[
                        {'label': 'Green Space Score', 'value': 'Green Space Score'},
                        {'label': 'Air Quality (AQ) Metric (lower values = higher air quality)', 'value': 'Air Quality (AQ) Metric'},
                        {'label': 'Population Density', 'value': 'Population Density'},
                        {'label': 'Average Distance from Nearest 3 Green Spaces (km)', 'value': 'Distance from Nearest Green Space (km)'},
                        {'label': 'AQ Carbon Monoxide', 'value': 'AQ Carbon Monoxide'},
                        {'label': 'AQ Nitrogen Dioxide', 'value': 'AQ Nitrogen Dioxide'},
                        {'label': 'AQ Ozone', 'value': 'AQ Ozone'},
                        {'label': 'AQ Sulphur Dioxide', 'value': 'AQ Sulphur Dioxide'},
                        {'label': 'AQ Aerosol Index (larger particulates, e.g. soot and dust)', 'value': 'AQ Aerosol Index'},
                        {'label': 'Current Green Spaces (of all sizes)', 'value': 'Current Green Spaces'},
                        {'label': 'Urban Areas', 'value': 'Urban Areas'},
                        {'label': 'Buildings', 'value': 'Buildings'},
                        {'label': 'Airports', 'value': 'Airports'},
                        {'label': 'Water Bodies', 'value': 'Water Bodies'},
                        {'label': 'Railway Stations', 'value': 'Railway Stations'},
                        {'label': 'Land Type Penalty (<1) or Reward (>1)', 'value': 'Land Type Penalty (<1) or Reward (>1)'}
            ],
            optionHeight=20,                    #height/space between dropdown options
            value='Water Bodies',         #dropdown value selected automatically when page loads
            disabled=False,                     #disable dropdown value selection
            multi=False,                        #allow multiple dropdown values to be selected
            searchable=True,                    #allow user-searching of dropdown values
            search_value='',                    #remembers the value searched in dropdown
            placeholder='Select data to overlay onto the map',     #gray, default text shown when no option is selected
            clearable=True,                     #allow user to removes the selected value
            style={'width':"55%",'font-family': 'Arial', 'font-size': '16px', 'textAlign': 'center'},             #use dictionary to define CSS styles of your dropdown
            # className='select_box',           #activate separate CSS document in assets folder
            # persistence=True,                 #remembers dropdown value. Used with persistence_type
            # persistence_type='memory'         #remembers dropdown value selected until...
            ),                                  #'memory': browser tab is refreshed
                                                #'session': browser tab is closed
                                                #'local': browser cookies are deleted
        ],className='fifteen columns'),
    
    html.Div([
        dcc.Graph(id='our_graph')
    ],className='fifteen columns'),
    
])
    
@app.callback(
    Output(component_id='our_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def build_graph(column_chosen):
    if column_chosen in ['Population Density']:
        # density
        #https://plotly.com/python/builtin-colorscales/
        fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=column_chosen, radius=15, opacity=0.4,
                            center=dict(lat=51.50009, lon=0.1268072), zoom=9.25,
                            #'open-street-map', 'carto-positron", 'carto-darkmatter', 'stamen-terrain', 'stamen-toner', 'stamen-watercolor' 
                            mapbox_style="carto-positron",
                            color_continuous_scale = 'Turbo')   #or Turbo
    elif column_chosen in ['Current Green Spaces', 'Urban Areas', 'Buildings', 'Water Bodies', 'Airports', 'Railway Stations']:
        # scatter
        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', 
                                opacity = 0.25, color = column_chosen,
                                zoom=9.25, mapbox_style="carto-positron", color_continuous_scale = ['white', 'green'])
    elif column_chosen in ['Green Space Score', 'Distance from Nearest Green Space']:
        # scatter
        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', size = column_chosen,
                                opacity = 0.3, zoom=9.5, mapbox_style="carto-positron", 
                                color = column_chosen, color_continuous_scale = 'Turbo')
    else:
        # hexbin
        # >100 horizontal hexagons has performance issues, long to load, also get empyt hexagons as no value to fill (would also therefore need to up resolution)
        fig = ff.create_hexbin_mapbox(df, lat="Latitude", lon="Longitude", color=column_chosen, nx_hexagon=150, 
                                    opacity=0.25, center=dict(lat=51.50009, lon=0.1268072),
                                    mapbox_style="carto-positron", zoom=9.25, color_continuous_scale = 'Turbo',
                                    labels={"color": column_chosen}, agg_func = np.mean)
                                    #, min_count=1)
        fig.update_traces(marker_line_width=0)
    
    fig.update_layout(
        autosize=True,
        #width=2000,
        height=825,
        margin=dict(l=20, r=20, t=20, b=20),
        #paper_bgcolor="chartreuse"
        )
    
    fig.layout.coloraxis.colorbar.title = ''
    
    #add border to graphing area
    # import plotly.graph_objects as go
    # fig.update_layout(shapes=[go.layout.Shape(
    #                                             type='rect',
    #                                             xref='paper',
    #                                             yref='paper',
    #                                             x0=0,
    #                                             y0=-0.1,
    #                                             x1=1.01,
    #                                             y1=1.02,
    #                                             line={'width': 1, 'color': 'black'}
    #                                             )])
    
    fig.update_mapboxes(pitch=40)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
  
