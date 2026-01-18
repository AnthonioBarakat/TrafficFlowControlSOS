import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime




def load_kaggle_traffic_csvs(data_dir):
    """Load all CSV files in data_dir; expects each CSV to have columns 'time' and 'count'.
    Returns a DataFrame with columns ['sensor','time','count'].
    """
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")


    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'time' not in df.columns or 'count' not in df.columns:
            # try common alternatives
            cols = df.columns.tolist()
            if len(cols) >= 2:
                df = df.iloc[:, :2]
                df.columns = ['time', 'count']
            else:
                raise ValueError(f"CSV {f} does not have expected columns")
        df = df[['time', 'count']].copy()
        # normalize time
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception:
            # try epoch integer
            df['time'] = pd.to_datetime(df['time'], unit='s')
        sensor_name = os.path.splitext(os.path.basename(f))[0]
        df['sensor'] = sensor_name
        dfs.append(df)


    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values(['sensor', 'time']).reset_index(drop=True)
    return all_df




def build_hourly_baselines(df):
    """Return baseline average counts per sensor per hour (0-23).
    Useful for seeding environment density or as target patterns.
    Output: DataFrame sensor,hour,avg_count
    """
    df['hour'] = df['time'].dt.hour
    baseline = df.groupby(['sensor', 'hour'])['count'].mean().reset_index()
    baseline.rename(columns={'count': 'avg_count'}, inplace=True)
    return baseline

# x = load_kaggle_traffic_csvs('./Data')

# x = build_hourly_baselines(x)

# print(x.iloc[0:24])
