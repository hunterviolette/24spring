import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
    from util.core import Schmoo
else:
    from pages.util.util import Preprocessing, DashUtil
    from pages.util.core import Schmoo


register_page(__name__, suppress_callback_exceptions=True)

class Generator(DashUtil, Preprocessing):

  modelPath = './vol/models'
  tmodelsPath = f"{modelPath}/test_models"
  dataPath = './vol/image_data'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([
            dbc.Row([
              html.H2("Training Parameters", style={'text-align': 'center'}),

              dbc.Col([
                html.H4("Image directories", style={'text-align': 'center'}),
                dcc.Dropdown(id='gen_trainDirs', multi=True,
                            style=Generator.Formatting('textStyle'),
                          ),
              ]),
              dbc.Col([
                  html.H4("Number of steps", style={'text-align': 'center'}),
                  dcc.Input(id='gen_steps', type='number', value=1,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("Start epoch", style={'text-align': 'center'}),
                  dcc.Input(id='gen_spoch', type='number', value=50,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("End epoch", style={'text-align': 'center'}),
                  dcc.Input(id='gen_epoch', type='number', value=500,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("Start weight decay", style={'text-align': 'center'}),
                  dcc.Input(id='gen_sdecay', type='number', value=.0001,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("End weight decay", style={'text-align': 'center'}),
                  dcc.Input(id='gen_edecay', type='number', value=.005,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("Start learning rate", style={'text-align': 'center'}),
                  dcc.Input(id='gen_slr', type='number', value=.1,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("End learning rate", style={'text-align': 'center'}),
                  dcc.Input(id='gen_elr', type='number', value=.5,
                            className=Generator.Formatting('input'),
                            style=Generator.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                  html.H4("Train models", style={'text-align': 'center'}),
                  dcc.Dropdown(id='gen_train', multi=False, value=False,
                      style=Generator.Formatting('textStyle'),
                      options=[
                          {'label': 'True', 'value': True},
                          {'label': 'False', 'value': False}
                        ]
                      ),
              ]),
            ], align='center'),

            html.Br(),

            dbc.Row([
              html.H2("Testing Parameters", style={'text-align': 'center'}),

              dbc.Col([
                html.H4("Image directories", style={'text-align': 'center'}),
                dcc.Dropdown(id='gen_testDirs', multi=True,
                            style=Generator.Formatting('textStyle'),
                          ),
              ]),
              dbc.Col([
                html.Button("click to run", n_clicks=0, id="gen_button", 
                            className=Generator.Formatting('button', 'info')
                          ),
              ], className='text-center'),

            ], align='center'),
            
          html.Div(id='gen_mdiv')

        ], className='mb-4', style=Generator.Formatting('mdiv'))  
  
  def callbacks(self):
    @callback(
      [Output("gen_trainDirs", "options"),
      Output("gen_testDirs", "options"),
      ],
      Input("gen_button", "n_clicks"),
    )
    def initMP(clicks):
      data = [x for x in os.listdir(Generator.dataPath) 
              if os.path.isdir(f"{Generator.dataPath}/{x}") 
              and "_nomask" not in x
            ]
      return (data, data) 
      
    @callback(
      [Output("gen_trainDirs", "value"),
      Output("gen_steps", 'value'),
      Output("gen_spoch", "value"),
      Output("gen_epoch", 'value'),
      Output("gen_sdecay", "value"),
      Output("gen_edecay", 'value'),
      Output("gen_slr", "value"),
      Output("gen_elr", 'value'),
      Output("gen_train", 'value'),
      Output("gen_testDirs", 'value'),
      Output("gen_mdiv", "children")
      ],
      Input("gen_button", "n_clicks"),
      [State("gen_trainDirs", "value"),
      State("gen_steps", 'value'),
      State("gen_spoch", "value"),
      State("gen_epoch", 'value'),
      State("gen_sdecay", "value"),
      State("gen_edecay", 'value'),
      State("gen_slr", "value"),
      State("gen_elr", 'value'),
      State("gen_train", 'value'),
      State("gen_testDirs", 'value'),
      ]
    )

    def maincb(clicks, trDirs, steps, 
              spoch, epoch, 
              sdecay, edecay, 
              slr, elr, 
              train, teDirs, 
            ):
      
      mdiv = []
      mdiv.append(html.H2(Generator.TorchGPU(), 
                  className=Generator.Formatting(color='warning')))
      
      baseModels = ['cyto3'] #['cyto', 'cyto2', 'cyto3']

      if clicks > 0 and \
          steps != None and \
          spoch != None and \
          epoch != None and \
          sdecay != None and \
          edecay != None and \
          slr != None and \
          elr != None and \
          trDirs != None and \
          teDirs != None:
                  

        trainDirs = [f"{Generator.dataPath}/{x}" for x in trDirs]
        testDirs = [f"{Generator.dataPath}/{x}" for x in teDirs]

        x = Schmoo()
        
        df, modelsPath = x.BatchTrain(
                savePath=f"{Generator.modelPath}/test_models",
                steps=steps,
                sEpoch=spoch,
                eEpoch=epoch,
                sLearningRate=slr,
                eLearningRate=elr,
                sWeightDecay=sdecay,
                eWeightDecay=edecay,
                diamMeans=[30],
                baseModels=baseModels,
                dataPaths=trainDirs,
                train=train
              )
        
        print(df)
        mdiv.append(Generator.DarkDashTable(df))
        
        df = x.BatchEval(
                diamMeans=[30,80,120],
                imageDir=testDirs,
                testModels=modelsPath
              )
        mdiv.extend([
                Generator.DarkDashTable(
                  Generator.EvalAggTransforms(df)),
                
                Generator.DarkDashTable(df),
              ])
                  
      return (trDirs, steps, 
              spoch, epoch, 
              sdecay, edecay, 
              slr, elr, 
              train, teDirs, 
              mdiv
            )
    
  
x = Generator()
layout = x.layout()
x.callbacks()
