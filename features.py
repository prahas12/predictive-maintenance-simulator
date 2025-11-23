# src/features.py
"""
Feature engineering helpers for predictive maintenance simulator.

Functions:
- load_data(path) -> DataFrame
- make_window_features(df, window_sec=60, step_sec=60) -> X (pd.DataFrame), y (pd.Series)
- small helper features used by train.py
"""

import numpy as np
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure timestamp is datetime
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def _agg_stats(g):
    # g is a dataframe window for a single device
    out = {}
    cols = ['vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'pressure', 'rpm', 'current']
    for c in cols:
        out[f'{c}_mean'] = g[c].mean()
        out[f'{c}_std'] = g[c].std()
        out[f'{c}_min'] = g[c].min()
        out[f'{c}_max'] = g[c].max()
        out[f'{c}_median'] = g[c].median()
        # simple trend: slope using np.polyfit on index
        try:
            x = np.arange(len(g))
            if len(x) > 1:
                slope = np.polyfit(x, g[c].values, 1)[0]
            else:
                slope = 0.0
        except Exception:
            slope = 0.0
        out[f'{c}_slope'] = float(slope)
    # cross-axis simple features
    out['vib_xy_corr'] = g['vibration_x'].corr(g['vibration_y'])
    out['vib_x_rms'] = np.sqrt(np.mean(g['vibration_x']**2))
    out['vib_y_rms'] = np.sqrt(np.mean(g['vibration_y']**2))
    out['vib_z_rms'] = np.sqrt(np.mean(g['vibration_z']**2))
    return out

def make_window_features(df: pd.DataFrame, window_sec: int = 60, step_sec: int = 60):
    """
    Convert raw time-series into windowed features.

    - df: must contain device_id, timestamp (datetime), and sensor columns.
    - window_sec: window length in seconds
    - step_sec: window step in seconds

    Returns:
    - X: DataFrame of features (one row per device-window)
    - y: Series of integer labels (max operational_state in the window: 0/1/2)
    """
    assert 'device_id' in df.columns
    assert 'timestamp' in df.columns

    df = df.sort_values(['device_id', 'timestamp']).reset_index(drop=True)
    devices = df['device_id'].unique()

    rows = []
    labels = []
    meta = []

    for dev in devices:
        ddf = df[df['device_id'] == dev].copy().reset_index(drop=True)
        if ddf.empty:
            continue
        t0 = ddf['timestamp'].min()
        t_end = ddf['timestamp'].max()
        # convert to integer seconds from t0 for indexing
        ddf['_secs'] = (ddf['timestamp'] - t0).dt.total_seconds().astype(int)

        start = 0
        last_start = int((t_end - t0).total_seconds()) - 1
        while start <= last_start:
            end = start + window_sec
            win = ddf[(ddf['_secs'] >= start) & (ddf['_secs'] < end)]
            if len(win) >= max(1, int(0.1 * window_sec)):  # require at least some data
                feats = _agg_stats(win)
                rows.append(feats)
                # label: maximum operational_state observed in the window (0 normal, 1 degraded, 2 failed)
                labels.append(int(win['operational_state'].max()) if 'operational_state' in win.columns else 0)
                meta.append({'device_id': dev, 'window_start': t0 + pd.to_timedelta(start, unit='s')})
            start += step_sec

    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(labels, name='label')
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df
