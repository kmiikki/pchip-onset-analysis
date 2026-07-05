#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:21:24 2024

@author: kim
"""

import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Align and process sensor data.')
    parser.add_argument('-ts', type=str, help="The .ts file to read")
    parser.add_argument('-csv', type=str, help="The .csv file to read")
    parser.add_argument('-min', action = 'store_true', help='Convert s to min', required=False)
    parser.add_argument('-o', '--offset', type=int, default=0, help='The time offset for alignment')
    parser.add_argument('-a', '--align', type=int, choices=[1, 2], default=1, help='Alignment method: 1 for Timestamp, 2 for Time (s)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    ts_file = args.ts if args.ts else 'tlpicam.ts'
    csv_file = args.csv
    offset = args.offset
    align_method = args.align
    
    if args.min:
        is_min = True
    else:
        is_min = False

    print(f"ts_file: {ts_file}")
    print(f"csv_file: {csv_file}")
    print(f"offset: {offset}")
    print(f"align_method: {align_method}")

    # Check if the rgb file exists
    rgb_file = 'rgb.csv'
    rgb_exists = os.path.isfile(rgb_file)
    print(f"rgb_exists: {rgb_exists}")

    # Read the ts file and csv file
    ts_df = pd.read_csv(ts_file)
    ts_df.columns = [col.strip() for col in ts_df.columns]
    print("ts_df columns:", ts_df.columns)

    csv_df = pd.read_csv(csv_file)
    csv_df.columns = [col.strip() for col in csv_df.columns]
    csv_df['Time (s)'] = csv_df['Time (s)'].astype(float)
    print("csv_df columns:", csv_df.columns)

    # Read the rgb file and extract picture number from picture_name
    if rgb_exists:
        rgb_df = pd.read_csv(rgb_file)
        rgb_df.columns = [col.strip() for col in rgb_df.columns]
        rgb_df['Image'] = rgb_df['picture_name'].str.extract(r'(\d+)').astype(int)
        print("rgb_df columns:", rgb_df.columns)

        # Align based on Image number
        merged_df = pd.merge(ts_df, rgb_df, on='Image', how='inner')
        print("Merged ts_df and rgb_df.")
        
        output_columns = ['Image', 'Datetime', 'Timestamp', 'Time (s)', 'BW', 'R', 'G', 'B']
        merged_df.rename(columns={'bw': 'BW', 'red': 'R', 'green': 'G', 'blue': 'B'}, inplace=True)
        
        # Merge temperature data from the provided CSV file based on Timestamp
        if 'Timestamp' in csv_df.columns:
            # Ensure the timestamps are aligned and properly formatted
            csv_df['Timestamp'] = csv_df['Timestamp'].astype(int)  # Convert to int for consistent comparison
            merged_df = pd.merge_asof(
                merged_df.sort_values('Timestamp'),
                csv_df[['Timestamp', 'Tr (°C)']].sort_values('Timestamp'),
                on='Timestamp',
                direction='nearest'
            )

        if 'Tr (°C)' in merged_df.columns:
            output_columns.insert(4, 'Tr (°C)')
    else:
        merged_df = ts_df.copy()
        if is_min:
            output_columns = ['Image', 'Datetime', 'Timestamp', 'Time (s)', 'Tr (°C)']

    print("merged_df columns:", merged_df.columns)

    # Filter the columns to keep, ensure they exist in the DataFrame
    output_columns = [col for col in output_columns if col in merged_df.columns]
    result_df = merged_df[output_columns]
    if is_min:
        unit = "Time (min)"
        result_df.rename(columns={'Time (s)': 'Time (min)'}, inplace=True)
        result_df['Time (min)'] = result_df['Time (min)'] / 60
    else:
        unit = "Time (s)"
    
    # Create analysis directory if it does not exist
    if not os.path.exists('analysis'):
        os.makedirs('analysis')

    # Write the results to CSV
    output_file = 'rgb-tr.csv' if rgb_exists else 'tl-tr.csv'
    result_path = os.path.join('analysis', output_file)
    result_df.to_csv(result_path, index=False)
    print(f"Result written to: {result_path}")

    # Plotting
    if rgb_exists and not result_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        if 'R' in result_df.columns and 'G' in result_df.columns and 'B' in result_df.columns:
            ax1.plot(result_df[unit], result_df['R'], 'r-', label='R')
            ax1.plot(result_df[unit], result_df['G'], 'g-', label='G')
            ax1.plot(result_df[unit], result_df['B'], 'b-', label='B')

        if 'Tr (°C)' in result_df.columns:
            ax2.plot(result_df[unit], result_df['Tr (°C)'], 'k-', label='Temperature (°C)')

        ax1.set_xlabel(unit)
        ax1.set_ylabel('RGB Values')
        ax2.set_ylabel('Temperature (°C)')

        fig.legend(loc='right', bbox_to_anchor=(0.9, 0.5))
        plt.title('RGB and Temperature Over Time')
        ax1.grid(axis='x')
        ax2.grid(axis='y')

        plt.savefig(os.path.join('analysis', 'rgb-tr.png'), dpi=300)
        plt.close()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(result_df[unit], result_df['BW'], 'k-', label='BW')
        if 'Tr (°C)' in result_df.columns:
            ax2.plot(result_df[unit], result_df['Tr (°C)'], color='0.5', linestyle='-', label='Temperature (°C)')

        ax1.set_xlabel(unit)
        ax1.set_ylabel('BW Values')
        ax2.set_ylabel('Temperature (°C)')

        fig.legend(loc='right', bbox_to_anchor=(0.9, 0.5))
        plt.title('BW and Temperature Over Time')
        ax1.grid(axis='x')
        ax2.grid(axis='y')

        plt.savefig(os.path.join('analysis', 'bw-tr.png'), dpi=300)
        plt.close()
    else:
        print("No data to plot or required columns are missing.")

if __name__ == "__main__":
    main()
