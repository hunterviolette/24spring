import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os 
import datetime
import re
import json
import io
import base64
from time import time

from tifffile import imread, imwrite
from dash import dcc, dash_table, html
from PIL import Image
from typing import List

class DashUtil:

  @staticmethod
  def FigAnnotations(
      df: pd.DataFrame,
      xcol: str,
      ycol: str,
      plot_every_n_points: int = 2
    ): 
    return [
        dict(
          x=x,
          y=y + (df[ycol].max() - df[ycol].min()) * 0.0000000005, 
          text=str(y),
          showarrow=False,
          font=dict(size=8),
          yanchor='bottom', 
          yshift=10  
        ) 
        for x, y in zip(df[xcol], df[ycol].round(3))
        if x % plot_every_n_points == 0
      ]

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

  def HtmlTables(self):
    Preprocessing.Df_To_HTML(
        self.mb.drop(columns=["Mass Flow (kg/h)",
                              "Molecular Weight (grams/mol)", 
                              "Component Flow (kmol/h)"]),
        header="",
        name="./vol/html_tables/non_ss_balance"
      )
    
    df = self.flows.assign(**{
              "Flow (mtpd)": lambda x: x["Flow (kg/batch)"
                ].apply(lambda y: self.q(y, 'kg/batch').to('mtpd').magnitude),
              
              "Stream Type": lambda x: x["Stream Type"
                ].replace({"products": "outlet", "reagents": "inlet"})
              })
          
    Preprocessing.Df_To_HTML(
        self.ssbal,
        header="",
        name="./vol/html_tables/ss_balance"
      )
    
    Preprocessing.Df_To_HTML(
        df,
        header="",
        name="./vol/html_tables/iteration_table"
      )
    
    Preprocessing.Df_To_HTML(
        df.loc[
            (df["Iteration"] == df["Iteration"].max()) 
          ].pivot_table(
            index=['Unit', 'Iteration', "Stream Type"], 
            columns='Chemical', 
            values='Flow (kg/batch)', 
            fill_value=""
        ), 
        header="Inlet/Outlet flows for unit operations (kg/batch) at steady state",
        name="./vol/html_tables/ss_pass"
      )

    Preprocessing.Df_To_HTML(
        df.loc[
            (df["Iteration"] == df["Iteration"].min()) 
          ].pivot_table(
            index=['Unit', 'Iteration', "Stream Type"], 
            columns='Chemical', 
            values='Flow (kg/batch)', 
            fill_value=""
        ), 
        header="Inlet/Outlet flows for unit operations (kg/batch) at first pass",
        name="./vol/html_tables/s0_pass"
      )
    
    Preprocessing.Df_To_HTML(
        df.loc[
            (df["Iteration"] == df["Iteration"].max()) &
            (df["Stream Type"] != "inlet")
          ].pivot_table(
              index=['Unit', 'Stream Type'], 
              columns='Chemical', 
              values='Flow (mtpd)', 
              fill_value="", 
        ), 
        header="Outlet flows for unit operations (mtpd) at steady state",
        name="./vol/html_tables/outlet_ss"
      )

    Preprocessing.Df_To_HTML(
        df.loc[
            (df["Iteration"] == df["Iteration"].max()) &
            (df["Stream Type"] == "inlet")
          ].pivot_table(
              index=['Unit', 'Stream Type'], 
              columns='Chemical', 
              values='Flow (mtpd)', 
              fill_value="", 
        ), 
        header="Inlet flows for unit operations (mtpd) at steady state",
        name="./vol/html_tables/inlet_ss"
      )


  @staticmethod
  def Df_To_HTML(df: pd.DataFrame,
                header: str = "", 
                name: str = 'mdf'
              ):
    
    style = [
      dict(selector="th", props=[("background-color", "#154c79"),
                                  ("color", "white"),
                                  ("border", "1px solid white"),
                                  ("text-align", "center"),
                                  ("font-size", "16px"),
                                  ("padding", "4px"),
                                  ("font-weight", "bold")]),
      dict(selector="td", props=[("background-color", "lightgray"),
                                  ("color", "black"),
                                  ("border", "1px solid black"),
                                  ("text-align", "center"),
                                  ("font-size", "14px"),
                                  ("padding", "4px")]),
    ]

    with open(f'{name}.html', 'w', encoding='utf-8') as file:
      file.write(f"<h2>{header}</h2>")
      file.write(df.style.set_table_styles(style
                                  ).format(precision=4
                                  ).to_html())

  @staticmethod
  def StrTime(asInt: bool = False):
    x = int(datetime.datetime.utcnow().timestamp())
    if asInt: return x
    else: return str(x)
        
  @staticmethod
  def ReadImage(name, directory):
    if ".tif" in name: return imread(f"{directory}/{name}") 

  @staticmethod
  def initDir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    else: print(f"Directory: {dir_path} exists")

  def JSON_to_Pandas(self, readDir: str = "./vol/states"):

    mdf = pd.DataFrame()
    for file in sorted(os.listdir(readDir), 
                      key=lambda x: int(re.search(r'\d+', x).group())):
      
      with open(f"{readDir}/{file}", "r") as f: js = json.load(f)

      df = pd.DataFrame(js["Units"]).T
      df.index.name = "Unit"
      df["iteration"] = file.split("_")[-1].split(".")[0]
      mdf = pd.concat([mdf, df.rename(columns={"Unamed: 0": "Unit"})], axis=0)

    self.unitdf = mdf.reset_index()

  def UnitFlows(self):
    Preprocessing.JSON_to_Pandas(self)
    df = self.unitdf[["Unit", "flow", "iteration"]]

    mdf = pd.DataFrame()
    for unit in df["Unit"].unique():
      
      for i in df["iteration"]:
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

  def UnitFlowsV2(self):
    Preprocessing.JSON_to_Pandas(self)
    df = self.unitdf[["Unit", "flow", "iteration", 'execution stage']]

    mdf = pd.DataFrame()
    for unit, unit_data in df.groupby("Unit"):
      for iteration, iteration_data in unit_data.groupby("iteration"):
        flows = iteration_data["flow"].iloc[0]
        for stream_type, stream_data in flows.items():
          for chem, chem_data in stream_data.items():
            for basis, value in chem_data.items():
              mdf = pd.concat([
                        mdf, 
                        pd.DataFrame({
                            "Unit": [unit],
                            "Stage": [unit_data["execution stage"].iloc[0]],
                            "Iteration": [iteration],
                            "Stream Type": [stream_type],
                            "Chemical": [chem],
                            basis: [abs(value)]
                          })
                        ], ignore_index=True)

    self.flows = mdf.groupby(
              ['Unit', 'Stage', 'Iteration', 'Stream Type', 'Chemical'], 
              as_index=False
            ).first(
            ).astype({"Iteration": int})

  def HeatFlows(self):
    if not hasattr(self, 'unitdf'): 
      Preprocessing.JSON_to_Pandas(self)
    
    df = self.unitdf[["Unit", "reaction", "iteration"]]
    df = df.loc[df["Unit"].apply(lambda x: x.split("-")[0] in ["R", "EL"])]

    mdf = pd.DataFrame()
    for unit, unit_data in df.groupby("Unit"):
      for iteration, iteration_data in unit_data.groupby("iteration"):
        dat = iteration_data["reaction"].iloc[0]
        
        mdf = pd.concat([
          mdf, 
          pd.DataFrame({
            "Unit": [unit],
            "Iteration": [int(iteration)],
            "Overall Molar Heat (kJ/mol)": [dat.get('Overall Molar Heat (kJ/mol)', 0)],
            "Limiting Reagent Flow (kmol/batch)": [dat.get("Limiting Reagent Flow (kmol/batch)", 0)],
            "Utility (kW)": [dat.get("Utility (kW)", 0)]
            })
        ])

    self.heat_flows = mdf

if __name__ == "__main__":
  gpuEnabled = True

  x = Preprocessing()