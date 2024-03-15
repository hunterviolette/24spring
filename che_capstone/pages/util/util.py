import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os 
import datetime
import re
import json
import io
import base64

from tifffile import imread, imwrite
from skimage.io import imread as pngRead
from dash import dcc, dash_table, html
from PIL import Image

class DashUtil:

  @staticmethod
  def EvalAggTransforms(df: pd.DataFrame):
    df = df.groupby(by=["model", "diam mean"], as_index=True
                        ).agg({"jaccard index": ["count", 'mean', 'std'],
                                "euclidean normalized rmse": ['mean'],
                                "structural similarity": ['mean'],
                                "average precision": ["mean"],
                                "true positives": ["mean"],
                                "false positives": ["mean"],
                                "false negatives": ["mean"]
                        }).reset_index(
                        ).round(5)  
    df.columns = df.columns.map(' '.join)
    return df.sort_values("euclidean normalized rmse mean",
          ).rename(columns={"jaccard index count": 'sample size'})


  @staticmethod
  def DarkDashTable(df, rows: int = 30):
    return dash_table.DataTable(
            data = df.to_dict('records'),
            columns = [{'name': i, 'id': i} for i in df.columns],
            export_format="csv",
            sort_action='native', 
            page_size=rows,
            filter_action='native',
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)','color': 'white'},
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '70px', 'width': '70px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
              }
          ) 

  @staticmethod
  def PI2(img: np.ndarray, 
          wh: int = 450, # width height max pixel size
          flip: bool = True, 
          zoom: bool = False
        ):
    
    buffer = io.BytesIO()
    
    if flip: Image.fromarray(img).transpose(Image.FLIP_TOP_BOTTOM).save(buffer, format='PNG')
    else: Image.fromarray(img).save(buffer, format='PNG')

    encImage = base64.b64encode(buffer.getvalue()).decode('utf-8')
    if zoom:
      fig = go.Figure()
      fig.add_trace(go.Image(source=f"data:image/png;base64,{encImage}"))
      fig.update_layout(
          margin=dict(l=0.1, r=0.1, t=0.1, b=0.1), 
          height=wh, width=wh
        )

      return dcc.Graph(figure=fig)
    else: return html.Div(
                  html.Img(
                    src=f"data:image/png;base64,{encImage}",
                    style={'max-height': f'{wh}px', 'max-width': f'{wh}px', 
                          'width': 'auto', 'height': 'auto', 
                          'display': 'block', 'margin': 'auto'}
                    ))

  @staticmethod
  def Formatting( 
                className: str = 'heading', 
                color: str = 'info',
                textAlign: str = 'center'
              ):

    if className == 'heading':
      return f"bg-opacity-50 p-1 m-1 bg-{color} text-dark fw-bold rounded text-{textAlign}"
    
    elif className == 'mdiv': return {"padding": "10px"} # style not className

    elif className == 'button': return f"btn btn-{color}"

    elif className == 'input': return 'form-control'

    elif className == 'textStyle': return {'text-align': 'center', 'color': 'black'} # style
    
    else: raise Exception("className not found")

class Preprocessing:

  def __init__(self) -> None:
    pass

  @staticmethod
  def StrTime(asInt: bool = False):
    x = int(datetime.datetime.utcnow().timestamp())
    if asInt: return x
    else: return str(x)
        
  @staticmethod
  def ReadImage(name, directory):
    if ".tif" in name: return imread(f"{directory}/{name}") 
    if ".png" in name: return pngRead(f"{directory}/{name}") 

  @staticmethod
  def initDir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    else: print(f"Directory: {dir_path} exists")


  def JSON_to_Pandas(self, readDir: str = "./states"):

    mdf = pd.DataFrame()
    for file in sorted(os.listdir(readDir), 
                      key=lambda x: int(re.search(r'\d+', x).group())):
      
      with open(f"{readDir}/{file}", "r") as f: js = json.load(f)

      df = pd.DataFrame(js["Units"]).T
      df.index.name = "Unit"
      df["iteration"] = file.split("_")[-1].split(".")[0]
      mdf = pd.concat([mdf, df.rename(columns={"Unamed: 0": "Unit"})], axis=0)

    self.unitdf = mdf.reset_index()
    print(self.unitdf, self.unitdf.dtypes, sep='\n')

  def UnitFlows(self):
    Preprocessing.JSON_to_Pandas(self)
    df = self.unitdf[["Unit", "flow", "iteration"]]

    mdf = pd.DataFrame()
    for unit in df["Unit"].unique():
      
      for i in df["iteration"]:
        print(unit, i)

        flows = (
            df.loc[
              (df["Unit"] == unit) &
              (df["iteration"] == i)
            ]["flow"].values[0])
        
        for key in flows.keys():
          for chem in flows[key].keys():
            for basis in flows[key][chem].keys():
              mdf = pd.concat([mdf,
                        pd.DataFrame({
                          "Unit": [unit],
                          "Iteration": [i],
                          "Stream Type": [key],
                          "Chemical": [chem],
                          basis: [abs(flows[key][chem][basis])]
                        })
              ])

    df = mdf.groupby(
              ['Unit', 'Iteration', 'Stream Type', 'Chemical'], 
              as_index=False).first()
    
    self.flows= df.astype({"Iteration": int})


if __name__ == "__main__":
  gpuEnabled = True

  x = Preprocessing()
