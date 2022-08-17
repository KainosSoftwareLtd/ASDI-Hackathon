# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# df = pd.DataFrame({'Longitude': [-0.621715, 0.369802], 'Latitude': [51.251246, 51.715616], 'Magnitude': [100, 100]})
# df = pd.read_csv('./data/points_df.csv', index_col=0)
# df = pd.read_csv('./data/points_df_aq_filled.csv', index_col=0)
df = pd.read_csv('./data/points_df_aqindex_filled.csv', index_col=0)

# heatmap
#z = 'Value_co'
#z = 'Value_no2'
#z = 'Value_o3'
#z = 'Value_so2'
#z = 'AQI'
#fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=z, radius=5,
                        #center=dict(lat=51.5072, lon=0.1276), zoom=8,
                        #mapbox_style="carto-darkmatter") #"stamen-terrain"


# scatter
# fig = px.scatter_mapbox(points_df,
#                         lat='Latitude',
#                         lon='Longitude',
#                         hover_name="population",
#                         zoom=9,
#                         mapbox_style="carto-darkmatter")


#fig.update_layout(
    # autosize=True,
    # width=1200,
    #height=800,)




app.layout = html.Div(children=[
    html.H1(children='Green space suggestion dashboard'),

    html.H3(children='''
        Here we show our heatmap of london to suggest the most suitbale places for green spaces.
    '''),

    html.Div([
        
        html.Br(),
            html.Div(id='output_data'),
            html.Br(),
    
        html.Label('Select Overlay'),
        dcc.Dropdown(id='my_dropdown',
            options=[
                        {'label': 'Greenspaceability Metric', 'value': 'GM', 'disabled': True},
                        {'label': 'Air Quality Metric', 'value': 'AQI'},
                        {'label': 'Carbon Monoxide', 'value': 'Value_co'},
                        {'label': 'Nitrogen Dioxide', 'value': 'Value_no2'},
                        {'label': 'Ozone', 'value': 'Value_o3'},
                        {'label': 'Sulphur Dioxide', 'value': 'Value_so2'}
            ],
            optionHeight=25,                    #height/space between dropdown options
            value='AQI',         #dropdown value selected automatically when page loads
            disabled=False,                     #disable dropdown value selection
            multi=False,                        #allow multiple dropdown values to be selected
            searchable=True,                    #allow user-searching of dropdown values
            search_value='',                    #remembers the value searched in dropdown
            placeholder='Select a data overlay',     #gray, default text shown when no option is selected
            clearable=True,                     #allow user to removes the selected value
            style={'width':"100%"},             #use dictionary to define CSS styles of your dropdown
            # className='select_box',           #activate separate CSS document in assets folder
            # persistence=True,                 #remembers dropdown value. Used with persistence_type
            # persistence_type='memory'         #remembers dropdown value selected until...
            ),                                  #'memory': browser tab is refreshed
                                                #'session': browser tab is closed
                                                #'local': browser cookies are deleted
        ],className='three columns'),
    
    html.Div([
        dcc.Graph(id='our_graph')
    ],className='nine columns'),
    
])
    
@app.callback(
    Output(component_id='our_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def build_graph(column_chosen):
    fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z=column_chosen, radius=5,
                        center=dict(lat=51.5072, lon=0.1276), zoom=8.7,
                        mapbox_style="carto-darkmatter")
    fig.update_layout(
        # autosize=True,
        # width=1200,
        height=800,)
    return fig

#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ])

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
  
