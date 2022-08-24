# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import boto3

app = Dash(__name__)

#import data CSV locally
#df = pd.read_csv('./data/final_df.csv')
df = pd.read_csv('./data/land_type_025.csv')

#import data CSV from s3 bucket
# client = boto3.client('s3')
# obj = client.get_object(Bucket='asdi-hackathon', Key='final-data/final_df.csv')
# df = pd.read_csv(obj['Body'])


app.layout = html.Div(children=[
    html.H2(children='Green Space Suggestion Dashboard'),

    # html.H4(children='''
    #     Here we show our heatmap of London to suggest the most suitbale places for green spaces.
    # '''),

    html.Div([
        
        html.Div(id='output_data'),
    
        html.Label('Select Overlay'),
        dcc.Dropdown(id='my_dropdown',
            options=[
                        {'label': 'Greenspace Viability Score', 'value': 'Greenspace_score', 'disabled': True},
                        {'label': 'Population Density', 'value': 'Pop_density', 'disabled': True},
                        {'label': 'Air Quality (AQ) Metric', 'value': 'AQ_score', 'disabled': True},
                        {'label': 'AQ Carbon Monoxide', 'value': 'Value_co', 'disabled': True},
                        {'label': 'AQ Nitrogen Dioxide', 'value': 'Value_no2', 'disabled': True},
                        {'label': 'AQ Ozone', 'value': 'Value_o3', 'disabled': True},
                        {'label': 'AQ Sulphur Dioxide', 'value': 'Value_so2', 'disabled': True},
                        {'label': 'AQ Aerosol Index', 'value': 'Value_ai', 'disabled': True},
                        {'label': 'Current Greenspaces', 'value': 'Green_Space'}
            ],
            optionHeight=25,                    #height/space between dropdown options
            value='AQ_score',         #dropdown value selected automatically when page loads
            disabled=False,                     #disable dropdown value selection
            multi=False,                        #allow multiple dropdown values to be selected
            searchable=True,                    #allow user-searching of dropdown values
            search_value='',                    #remembers the value searched in dropdown
            placeholder='Select data to overlay onto the map',     #gray, default text shown when no option is selected
            clearable=True,                     #allow user to removes the selected value
            style={'width':"100%"},             #use dictionary to define CSS styles of your dropdown
            # className='select_box',           #activate separate CSS document in assets folder
            # persistence=True,                 #remembers dropdown value. Used with persistence_type
            # persistence_type='memory'         #remembers dropdown value selected until...
            ),                                  #'memory': browser tab is refreshed
                                                #'session': browser tab is closed
                                                #'local': browser cookies are deleted
        ],className='eight columns'),
    
    html.Div([
        dcc.Graph(id='our_graph')
    ],className='eight columns'),
    
])
    
@app.callback(
    Output(component_id='our_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def build_graph(column_chosen):
    # density
    #https://plotly.com/python/builtin-colorscales/
    # fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=column_chosen, radius=15, opacity=0.5,
    #                     center=dict(lat=51.50009, lon=0.1268072), zoom=10.1,
    #                     #'open-street-map', 'carto-positron", 'carto-darkmatter', 'stamen-terrain', 'stamen-toner', 'stamen-watercolor' 
    #                     mapbox_style="carto-positron",
    #                     color_continuous_scale = 'Turbo')   #Thermal
    
    # scatter
    # fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', opacity = 0.5,
    #                         color = column_chosen, zoom=9.5, mapbox_style="carto-positron")
    
    # >100 horizontal hexagons has performance issues, long to load, also get empyt hexagons as no value to fill (would also therefore need to up resolution)
    fig = ff.create_hexbin_mapbox(df, lat="Latitude", lon="Longitude", color=column_chosen, nx_hexagon=200, 
                                  opacity=0.3, center=dict(lat=51.50009, lon=0.1268072),
                                  mapbox_style="carto-positron", zoom=10.1, color_continuous_scale = 'Turbo',
                                  labels={"color": column_chosen})
                                  #, min_count=1)
    fig.update_traces(marker_line_width=0)
    
    fig.update_layout(
        autosize=True,
        #width=2000,
        height=850,
        margin=dict(l=20, r=20, t=20, b=20)
        #paper_bgcolor="PaleGreen"
        )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
  
