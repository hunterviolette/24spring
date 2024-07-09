import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import os

from dash import dcc, html, Input, Output, callback, State, register_page

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
else:
    from pages.util.util import Preprocessing, DashUtil
    from src.steady_state import SteadyState

register_page(__name__, suppress_callback_exceptions=True)

class Sim(DashUtil, Preprocessing, SteadyState):

  cfgPath = './vol/configs'

  def __init__(self) -> None:
    pass

  def layout(self):
    return html.Div([

      dbc.Row([
        dbc.Col([
            html.H4("Config File", style={'text-align': 'center'}),
            dcc.Dropdown(id='sim_cfgs', multi=False, 
                        style=Sim.Formatting('textStyle'),
                      ),
        ]),
        dbc.Col([
            html.H4("Max Steady-State Iterations", style={'text-align': 'center'}),
            dcc.Input(id='sim_iters', type='number', value=15, min=1, step=1,
                      className=Sim.Formatting('input'),
                      style=Sim.Formatting('textStyle')
                    ),
        ]),
        dbc.Col([
            html.H4("Run Simulator"),
            html.Button("click here", n_clicks=0, id="sim_run", 
                        className=Sim.Formatting('button', 'info')

                      ),
        ], className='text-center'),
      ], align='center'),

      html.Div(id='sim_div'),

    ], className='mb-4', style=Sim.Formatting('mdiv'))
  
  def callbacks(self):

    @callback(
      Output("sim_cfgs", "options"),
      Input("sim_run", "n_clicks"),
    )
    def initPred(clicks):
      return [
          x for x in os.listdir(Sim.cfgPath) 
          if os.path.isfile(f"{Sim.cfgPath}/{x}") 
          and x.endswith('.json')
        ]
  
    @callback(
      [Output("sim_cfgs", "value"),
      Output("sim_iters", "value"),
      Output("sim_div", "children")
      ],
      Input("sim_run", "n_clicks"),
      [State("sim_cfgs", "value"),
      State("sim_iters", "value"),
      ],
    )
    def maincb(clicks, cfg, iter):      

      print(clicks, cfg, type(iter))

      div = []
      if clicks > 0 and \
          cfg != None and \
          isinstance(iter, int):
        
        SteadyState.__init__(self,
              cfgPath=f"{Sim.cfgPath}/{cfg}",
              maxIterations=iter
            )
        
        Sim.Steady_State_Setpoint(self)
        Sim.UnitFlowsV2(self)
        Sim.CloseBalance(self, debug=False)
        Sim.HtmlTables(self)
              
      return (cfg, iter, div)
      

x = Sim()
layout = x.layout()
x.callbacks()
