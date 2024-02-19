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

class ModelPerformance(DashUtil, Preprocessing):

  modelPath = './vol/models'
  tmodelsPath = f"{modelPath}/test_models"
  dataPath = './vol/image_data'

  def __init__(self) -> None:
    pass
  
  @staticmethod
  def layout():
    return html.Div([
            dbc.Row([
              dbc.Col([
                html.H4("Input directory", style={'text-align': 'center'}),
                dcc.Dropdown(id='perf_dataDir', multi=True,
                            style=ModelPerformance.Formatting('textStyle'),
                          ),
              ]),
              dbc.Col([
                html.H4("Add test models", style={'text-align': 'center'}),
                dcc.Dropdown(id='perf_testModels', multi=True, clearable=True,
                            style=ModelPerformance.Formatting('textStyle'),
                          ),
              ]),
              dbc.Col([
                  html.H4("Max predictions", style={'text-align': 'center'}),
                  dcc.Input(id='perf_num', type='number', value=None,
                            className=ModelPerformance.Formatting('input'),
                            style=ModelPerformance.Formatting('textStyle')
                          ),
              ]),
              dbc.Col([
                html.H4("Click for model predictions"),
                html.Button("click here", n_clicks=0, id="perf_button", 
                            className=ModelPerformance.Formatting('button', 'info')
                          ),
              ], className='text-center'),
          
          ], align='center'),
            
          html.Div(id='perf_mdiv')

        ], className='mb-4', style=ModelPerformance.Formatting('mdiv'))  
  
  def callbacks(self):
    @callback(
      [Output("perf_dataDir", "options"),
      Output("perf_testModels", 'options')],
      Input("perf_button", "n_clicks"),
    )
    def initMP(clicks):
      return (
        os.listdir(ModelPerformance.dataPath),
        [x for x in os.listdir(ModelPerformance.tmodelsPath) 
          if os.path.isdir(f'{ModelPerformance.tmodelsPath}/{x}')]
      ) 
      
    @callback(
      [Output("perf_dataDir", "value"),
      Output("perf_testModels", 'value'),
      Output("perf_num", "value"),
      Output("perf_mdiv", "children")],
      Input("perf_button", "n_clicks"),
      [State("perf_dataDir", "value"),
      State("perf_testModels", "value"),
      State("perf_num", "value")]
    )
    def maincb(clicks, images, testModels, numPred):
      mdiv = []
      mdiv.append(html.H2(ModelPerformance.TorchGPU(), 
                  className=ModelPerformance.Formatting(color='warning')))
      
      if clicks > 0 and images != None:
        data = [f"{ModelPerformance.dataPath}/{x}" for x in images]
        df = Schmoo().BatchEval(
                          modelDir=ModelPerformance.modelPath,
                          numPredictions=numPred,
                          diamMeans=[30, 80],
                          saveCsv=False,
                          testModels=testModels,
                          imageDir=data
                        )
        
        df = df.astype({
                "average precision": 'float64',
                "true positives": 'float64',
                "false negatives": 'float64',
                "false positives": 'float64',
              })
        
        aggs = {"jaccard index": ["count", 'mean', 'std'],
                "euclidean normalized rmse": ['mean'],
                "structural similarity": ['mean'],
                "average precision": ["mean"],
                "true positives": ["mean"],
                "false positives": ["mean"],
                "false negatives": ["mean"]
              }
        
        mdf = df.groupby(by=["model", "diam mean"], as_index=True
                        ).agg(aggs).reset_index().round(5)
                
        for x in [
              ["Group by model and diameter mean", mdf],
              ["Base data", df]
            ]:
            
          d = x[1]
          if x[0] == "Base data":
            d = d.sort_values("jaccard index", ascending=False)
          
          else:
            d.columns = d.columns.map(' '.join)
            d = d.sort_values("jaccard index mean", ascending=False
                ).rename(columns={"jaccard index count": 'sample size'})
          
          print(d.dtypes)

          mdiv.extend([
              
              html.H3(x[0], 
                className=ModelPerformance.Formatting(color='primary')),
              
              ModelPerformance.DarkDashTable(d.round(4)),
            ])
        
      return (images, testModels, numPred, mdiv)      
  
x = ModelPerformance()
layout = x.layout()
x.callbacks()
