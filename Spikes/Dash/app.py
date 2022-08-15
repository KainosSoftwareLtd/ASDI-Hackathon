# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


points_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

# points_df = pd.DataFrame({'Longitude': [-0.621715, 0.369802], 'Latitude': [51.251246, 51.715616], 'Magnitude': [100, 100]})
points_df = pd.read_csv('./data/point_array_1.csv', index_col=0)

# heatmap
fig = px.density_mapbox(points_df, lat='Latitude', lon='Longitude', z='Magnitude', radius=5,
                        center=dict(lat=51.5072, lon=0.1276), zoom=8,
                        mapbox_style="carto-darkmatter") #"stamen-terrain"



# scatter
# fig = px.scatter_mapbox(points_df,
#                         lat='Latitude',
#                         lon='Longitude',
#                         hover_name="population",
#                         zoom=9,
#                         mapbox_style="carto-darkmatter")


fig.update_layout(
    # autosize=True,
    # width=1200,
    height=800,)




app.layout = html.Div(children=[
    html.H1(children='Green space suggestion dashboard'),

    html.H3(children='''
        Here we show our heatmap of london to suggest the most suitbale places for green spaces.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
  
