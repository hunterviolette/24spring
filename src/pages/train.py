import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os

from dash import dcc, html, Input, Output, callback, State, dash_table, register_page

if __name__ == '__main__':
    from util.util import Preprocessing, DashUtil
    from util.core import Schmoo
else:
    from pages.util.util import Preprocessing, DashUtil
    from pages.util.core import Schmoo

register_page(__name__, suppress_callback_exceptions=True)

class Train(DashUtil, Preprocessing):

  dataPath = './vol/image_data'
  modelDir = './vol/models'

  def __init__(self) -> None:
    pass

  def layout(self):
    return html.Div([
            dbc.Row([
              dbc.Col([
                  html.H4("Training Data", style={'text-align': 'center'}),
                  dcc.Dropdown(id='trainDirs', multi=True,
                              style=Train.Formatting('textStyle'),
                          ),
              ]),
              dbc.Col([
                  html.H4("Base Model", style={'text-align': 'center'}),
                  dcc.Dropdown(id='trainBmodels', multi=True,
                              style=Train.Formatting('textStyle'),
                          ),
              ]),
              dbc.Col([
                  html.H4("Learning Rate", style={'text-align': 'center'}),
                  dcc.Input(id='trainLrate', type='number', value=.2, 
                      className=Train.Formatting('input'),
                      style=Train.Formatting('textStyle')
                    ),
              ]),
              dbc.Col([
                  html.H4("Weight Decay", style={'text-align': 'center'}),
                  dcc.Input(id='trainDecay', type='number', value=.00001, 
                      className=Train.Formatting('input'),
                      style=Train.Formatting('textStyle')
                    ),
              ]),
              dbc.Col([
                  html.H4("Epochs", style={'text-align': 'center'}),
                  dcc.Input(id='trainEpoch', type='number', step=1, value=500, 
                      className=Train.Formatting('input'),
                      style=Train.Formatting('textStyle')
                    ),
              ]),
              dbc.Col([
                  html.H4("Diameter Mean", style={'text-align': 'center'}),
                  dcc.Input(id='trainDmean', type='number', step=1, value=30, 
                      className=Train.Formatting('input'),
                      style=Train.Formatting('textStyle')
                    ),
              ]),
              dbc.Col([
                  html.H4("Click to train model"),
                  html.Button("click here", n_clicks=0, id="trainButton", 
                              className=Train.Formatting('button', 'info')
                            ),
              ], className='text-center'),

            ], align='center'),
            html.Div(id='train_mdiv'),

          ], className='mb-4', style=Train.Formatting('mdiv'))  
  
  def callbacks(self):
    @callback(
      [Output("trainDirs", "options"),
      Output("trainBmodels", "options"),
      ],
      Input("trainButton", "n_clicks"),
    )
    def initMP(clicks):
      return (
        [x for x in os.listdir(Train.dataPath) 
          if "_nomask" not in x],
        [x for x in os.listdir(Train.modelDir) 
          if os.path.isfile(f'{Train.modelDir}/{x}')]
      )

    @callback(
      [Output('train_mdiv', 'children'),
      Output("trainDirs", 'value'),
      Output("trainBmodels", 'value'),
      Output("trainLrate", 'value'),
      Output("trainDecay", 'value'),
      Output("trainEpoch", "value"),
      Output("trainDmean", 'value'),
      ],
      Input('trainButton', 'n_clicks'),
      [State("trainDirs", 'value'),
      State("trainBmodels", "value"),
      State("trainLrate", "value"),
      State("trainDecay", 'value'),
      State("trainEpoch", "value"),
      State("trainDmean", 'value'),
      ],
    )
    def maincb(
            clicks, 
            dirs,
            baseModels,
            learningRate,
            weightDecay,
            n_epochs,
            diamMean
          ):
    
      mdiv = []
      if clicks > 0 and \
        baseModels != None and \
        learningRate != None  and \
        weightDecay != None and \
        n_epochs != None and \
        diamMean != None and \
        dirs != None:
        
        sPath = f"{Train.modelDir}/test_models"
        Train.initDir(sPath)

        dirPaths = [f"{Train.dataPath}/{x}" for x in dirs]

        for model in baseModels:
          print(f"{Train.modelDir}/{model}")

          name = Schmoo().Train(
                      model_type=model,
                      learning_rate=learningRate,
                      weight_decay=weightDecay,
                      n_epochs=n_epochs,
                      bigGen=dirPaths,
                      savePath=sPath
                    )
          
          mdiv.append(html.H3(f"Saved model: {name} to vol/models/test_models"))

      return (
            mdiv, 
            dirs,
            baseModels, 
            learningRate, 
            weightDecay, 
            n_epochs, 
            diamMean
          )

  
x = Train()
layout = x.layout()
x.callbacks()
