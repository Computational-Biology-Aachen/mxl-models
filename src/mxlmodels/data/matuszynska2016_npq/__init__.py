

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


@dataclass
class Data:
    """Container for model data."""

    fig4_data: pd.DataFrame
    fig6_data: pd.DataFrame


_default = Path(__file__).parent

def default(data_dir: Path = _default) -> Data:
    """Load default Matuszynska 2016 data."""
    db_path = data_dir / "matuszynska2016_npq.db"
    
    cnx = sqlite3.connect(db_path)
    cur = cnx.cursor()
    
    # reading all table names
    table_list = list(cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
    
    df = pd.read_sql_query("select * from LIGHTMEMORY", cnx)
    
    columns_to_keep = ["ExpId", "Fm", "Specie", "LightIntensity", "DarkDuration", "Time", "Replicate", "Ft", "Yield"]
    df = df.drop(columns=[i for i in df.columns if i not in columns_to_keep])


    return Data(
        fig4_data=prepare_experimental_data(df, "Arabidopsis", [100, 300, 900], [15, 30, 60]),
        fig6_data=prepare_experimental_data(df, "Pothos", [100], [15]),
    )

def prepare_experimental_data(
    data: pd.DataFrame,
    species: Literal["Arabidopsis", "Pothos"],
    pfds: list,
    darkdurations: list
) -> pd.DataFrame:
    df = data[data.Specie == species].copy()
    
    res = {}
    for lightintensity, darkduration in zip(pfds, darkdurations):
        df_li = df[df.LightIntensity == lightintensity].copy()
        df_li_dd = df_li[df_li.DarkDuration == darkduration].copy()
        df_li_dd['Time'] = pd.to_timedelta(df_li_dd['Time'])
        
        replicates = []
        
        for replic in [1, 2, 3]:
            df_li_dd_replic = df_li_dd[df_li_dd['Replicate'] == replic].copy()
            
            df_li_dd_replic['Timedelta'] = (df_li_dd_replic['Time'] - df_li_dd_replic['Time'].iloc[0]).apply(lambda x: x.total_seconds())
            df_li_dd_replic['Fm_rel'] = (df_li_dd_replic['Fm'] / df_li_dd_replic['Fm'].iloc[0])
            df_li_dd_replic['Ft_rel'] = (df_li_dd_replic['Ft'] / df_li_dd_replic['Fm'].iloc[0])
            df_li_dd_replic['Yield'] = df_li_dd_replic['Yield']
            
            replicates.append(df_li_dd_replic)

        df_li_dd = pd.concat(replicates)

        df_li_dd_mean = df_li_dd[['Timedelta', 'Fm_rel', 'Ft_rel', 'Yield', 'ExpId']].groupby('ExpId').agg({'Timedelta': 'mean', 'Fm_rel': ['mean', 'std'], 'Ft_rel': ['mean', 'std'], 'Yield': 'mean'})
        df_li_dd_mean.columns = ['Timedelta_mean','Fm_rel_mean', 'Fm_rel_std','Ft_rel_mean', 'Ft_rel_std', 'Yield_mean']
        
        res[lightintensity] = df_li_dd_mean

    return res