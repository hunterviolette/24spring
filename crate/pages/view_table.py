import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import os

from dash import dcc, html, Input, Output, callback, State, register_page

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class Table(DashUtil, Preprocessing):

  cfgPath = './vol/html_tables'

  def __init__(self) -> None:
    pass

  def layout(self):
    return html.Div([

      dcc.Interval(id='table_interval', interval=1*1000, n_intervals=0),  # Interval component for initial trigger

      dbc.Row([
        dbc.Col([
            html.H4("Table Select", style={'text-align': 'center'}),
            dcc.Dropdown(id='table_name', multi=True, 
                        style=Table.Formatting('textStyle'),
                      ),
        ]),
      ], align='center'),

      html.Div(id='table_div'),

    ], className='mb-4', style=Table.Formatting('mdiv'))
  
  def callbacks(self):

    @callback(
      Output("table_name", "options"),
      Input('table_interval', 'n_intervals')
    )
    def initPred(n_intervals):
      return [
          x for x in os.listdir(Table.cfgPath) 
          if os.path.isfile(f"{Table.cfgPath}/{x}") 
          and x.endswith('.html')
        ]
  
    @callback(
      [Output("table_name", "value"),
      Output("table_div", "children")
      ],
      Input("table_name", "value"),
    )
    def maincb(tables):      

      print(tables)

      div = []
      if tables is not None:
        
        [div.append(
          html.Iframe(src=f'{Table.cfgPath}/{x}', 
                      style={'width': '100%', 'height': '500px'})
        ) for x in tables]
              
      return (tables, div)
      

x = Table()
layout = x.layout()
x.callbacks()

