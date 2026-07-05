#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:58:14 2024

@author: Kim Miikki and Paula Översti
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
has_start = True
has_end = False



def compute_curvature(x, y):
    """
    Compute curvature kappa = |y''(x)| / (1 + (y'(x))^2)^(3/2).
    """
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    denom = (1 + dydx**2)**1.5
    denom[denom == 0] = 1e-12
    kappa = np.abs(d2ydx2) / denom
    return kappa

def find_local_maxima(signal, threshold_ratio=0.1):
    """
    Find local maxima above threshold_ratio * max(signal).
    """
    maxima_indices = []
    if len(signal) < 3:
        return maxima_indices
    
    max_val = np.nanmax(signal)  # use np.nanmax in case of any remaining NaNs
    threshold = threshold_ratio * max_val
    
    for i in range(1, len(signal) - 1):
        # Make sure we don't break if there's a NaN
        if np.isnan(signal[i-1]) or np.isnan(signal[i]) or np.isnan(signal[i+1]):
            continue
        
        if (signal[i] > signal[i - 1]) and (signal[i] >= signal[i + 1]) and (signal[i] >= threshold):
            maxima_indices.append(i)
    
    return maxima_indices

def find_largest_gaps(arr, top_n=1):
    """
    Finds the top_n largest gaps in a NumPy array.

    Parameters:
    - arr (np.ndarray): Input array.
    - top_n (int): Number of top gaps to return.

    Returns:
    - gaps (list of dict): Each dict contains 'start', 'end', and 'gap_length'.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    if arr.size < 2:
        raise ValueError("Array must contain at least two elements to have a gap.")
    
    # Sort the array
    sorted_arr = np.sort(arr)
    
    # Compute differences between consecutive elements
    diffs = np.diff(sorted_arr)
    
    # Find indices of the top_n largest gaps
    if top_n > len(diffs):
        top_n = len(diffs)
    
    # Argsort in descending order
    largest_indices = np.argsort(diffs)[-top_n:][::-1]
    
    # Retrieve gap details
    gaps = []
    for idx in largest_indices:
        gap_info = {
            'start': sorted_arr[idx],
            'end': sorted_arr[idx + 1],
            'gap_length': diffs[idx]
        }
        gaps.append(gap_info)
    
    return gaps

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-quiet', action="store_true", required=False,
                        help="dont show graph so script is non-blocking")
    args = parser.parse_args()
    # -- FILE NAMES AND COLUMN NAMES --
    infile = 'rgb-tr.csv'
    outfile_csv = 'rgb-tr-sg.csv'
    outfile_fig = 'rgb-tr-sg.png'
    
    temp_col = 'Tr (°C)'  # Temperature
    bw_col   = 'BW'       # BW data

    # --------------------------------------------------------------------
    # 1. Read CSV and handle NaNs
    # --------------------------------------------------------------------
    df = pd.read_csv(infile, encoding='utf-8')
    
    # Option A: drop rows that have NaN in temperature or BW
    # df.dropna(subset=[temp_col, bw_col], inplace=True)
    
    # Option B: fill with interpolation (example: linear)
    df[bw_col] = df[bw_col].interpolate(method='linear')
    
    # If there are still NaNs at the extremes (first/last), you could also:
    # df[bw_col].fillna(method='ffill', inplace=True)
    # df[bw_col].fillna(method='bfill', inplace=True)

    # --------------------------------------------------------------------
    # 2. Savitzky-Golay Smoothing
    # --------------------------------------------------------------------
    window_length = 101
    polyorder = 3
    
    bw_smooth = savgol_filter(df[bw_col], window_length, polyorder)
    
    # Add smoothed data to the DataFrame
    bw_index = df.columns.get_loc(bw_col)
    df.insert(bw_index + 1, 'BW_smooth', bw_smooth)
    
    # --------------------------------------------------------------------
    # 3. Compute Curvature & Find Local Maxima
    # --------------------------------------------------------------------
    x_vals = df[temp_col].values
    y_vals = df['BW_smooth'].values
    
    kappa = compute_curvature(x_vals, y_vals)
    curvature_peaks = find_local_maxima(kappa, threshold_ratio=0.00001)
    filtered_peaks = []
    if len(curvature_peaks) > 100:
        regions = find_largest_gaps(curvature_peaks, top_n=2)
        for gap in regions:
            if has_start:
                filtered_peaks.append(int(gap['start']))
            if has_end:
                filtered_peaks.append(int(gap['end']))
        filtered_peaks.sort()
        curvature_peaks = filtered_peaks

    # --------------------------------------------------------------------
    # 4. Save CSV
    # --------------------------------------------------------------------
    df.to_csv(outfile_csv, index=False, encoding='utf-8')
    print(f"Smoothed data saved to {outfile_csv}")

    # --------------------------------------------------------------------
    # 5. Plot
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,5), dpi=300)
    ax.plot(x_vals, y_vals, label='BW (smoothed)', color='blue')

    # Mark peaks
    for idx in curvature_peaks:
        tx = x_vals[idx]
        ty = y_vals[idx]
        ax.axvline(x=tx, color='red', linestyle='--', alpha=0.8)
        ax.text(tx, ty, f"{tx:.1f}°C", color='red', rotation=90, 
                va='bottom', ha='right', fontsize=8)

    ax.set_xlabel('Temperature (°C) - cooling')
    ax.set_ylabel('BW (smoothed)')
    ax.set_title('BW vs. Temperature (Smoothed) - Bend Points')
    ax.grid(True)
    ax.legend()
    
    # Reverse the X-axis (cooling)
    plt.gca().invert_xaxis()

    plt.savefig(outfile_fig, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {outfile_fig}")
    if not args.quiet:
        plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-quiet', action="store_true", required=False,
                        help="dont show graph so script is non-blocking")
    args = parser.parse_args()
    main()
