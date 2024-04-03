import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd

from dash import dcc, html, Input, Output, callback, State, register_page

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil

register_page(__name__, suppress_callback_exceptions=True)

class Flows(DashUtil, Preprocessing):

  def __init__(self) -> None:
    pass

  def layout(self):
    Flows.UnitFlowsV2(self) 

    return html.Div([

      Flows.DarkDashTable(self.flows.round(3))

    ], className='mb-4', style=Flows.Formatting('mdiv'))
  
  def callbacks(self):
    pass

x = Flows()
layout = x.layout()
x.callbacks()
