import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

import pandas as pd
from io import StringIO
import seaborn as sns
import os.path

import numpy as np
import scipy as sp

import argparse

def create_tavg_df_omp_pth_t10(csv_data: str):
    """
    Returns a dataframe for given CSV data in OMP,PTHREAD,t0..t9 format.

    Parameters:
        csv_data (str): String of CSV data in OMP,PTHREAD,SLICES,t0..t9 format

    Returns:
        pandas.DataFrame: DataFrame of OMP vs PThread average time data. 
    """
    
    sio_dat = StringIO(csv_data)
    df = pd.read_csv(sio_dat, sep=",")
    df_t = pd.DataFrame(df[["t0","t1","t2","t3","t4","t5","t6","t7","t8","t9"]])
    df_tavg = pd.DataFrame([df["OMP"], df["PTHREAD"], df_t.mean(axis=1)]).transpose().rename(columns = {"Unnamed 0" : "t_avg"})

    df_tavg["OMP"] = df_tavg["OMP"].astype('int')
    df_tavg["PTHREAD"] = df_tavg["PTHREAD"].astype('int')

    return pd.pivot_table(df_tavg, values='t_avg', index=['OMP'], columns='PTHREAD')

def create_tavg_df_omp_pth_sl_t10(csv_data: str):
    """
    Returns a dictionary of sliced dataframes for given CSV data in OMP,PTHREAD,SLICES,t0..t9 format.

    Parameters:
        csv_data (str): String of CSV data in OMP,PTHREAD,SLICES,t0..t9 format

    Returns:
        dict( {int: pandas.DataFrame}): Dictionary mapping the number of slices to the DataFrame of OMP vs PThread average time data. 
    """
    
    sliced_data = {}
    s_dat = StringIO(csv_data)
    df = pd.read_csv(s_dat, sep=",")
    slices = df.SLICES.unique()
    
    for i in slices:
        df_slice = df[df["SLICES"]==i]
        df_slice_t = pd.DataFrame(df_slice[["t0","t1","t2","t3","t4","t5","t6","t7","t8","t9"]])
        df_slice_tavg = pd.DataFrame([df_slice["OMP"], df_slice["PTHREAD"], df_slice_t.mean(axis=1)]).transpose().rename(columns = {"Unnamed 0" : "t_avg"})

        df_slice_tavg["OMP"] = df_slice_tavg["OMP"].astype('int')
        df_slice_tavg["PTHREAD"] = df_slice_tavg["PTHREAD"].astype('int')

        sliced_data[i] = pd.pivot_table(df_slice_tavg, values='t_avg', index=['OMP'], columns='PTHREAD')
    return sliced_data

def plot_single(df_omp_pthread: pd.DataFrame, title: str = ""):
    """
    Creates a plot for DataFrame of OMP vs PThread average time data.

    Parameters:
        df_omp_pthread (pandas.DataFrame): DataFrame of OMP vs PThread average time data. 
    """
    f = plt.figure()
    sns.heatmap(df_omp_pthread, 
                cmap="mako_r", 
                annot=True, 
                linewidths=.5, 
                fmt='1.2f', 
                cbar_kws={'label': r"$t$ (s)"})
    plt.xlabel(r"Taskflow pthreads")
    plt.ylabel(r"OpenMP threads")
    plt.title(title, fontsize=12)
    return f

def load_dat_to_str(filename):
    str_dat = None
    if os.path.isfile(filename):
        with open(filename) as f:
            str_dat = f.read()
    return str_dat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Output name for figures")
    parser.add_argument("csv", help="Data file to load")
    parser.add_argument("sliced", help="Indicate whether data is sliced", choices=["y", "n"])
    args = parser.parse_args()

    data = load_dat_to_str(args.csv)
    if(args.sliced == "y"):
        df_dict = create_tavg_df_omp_pth_sl_t10(data)
        for sl,dat in df_dict.items():
            f = plot_single(dat, f"{sl} sliced indices")
            f.savefig(f"{args.name}_slice{sl}.pdf")
    else:
        df = create_tavg_df_omp_pth_t10(data)
        f = plot_single(df, f"No sliced indices")
        f.savefig(f"{args.name}.pdf")