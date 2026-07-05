#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import re

# Define the main function
def main(xlsx_file):
    # Read the 'Measured values' sheet starting from the 3rd row
    df = pd.read_excel(xlsx_file, sheet_name='Measured values', skiprows=[1])

    # Identify the 'Abs. Time' column and extract UTC offset
    abs_time_col = [col for col in df.columns if 'Abs. Time' in col][0]
    utc_offset = int(re.search(r'UTC\+(\d+)', abs_time_col).group(1))

    # Convert 'Abs. Time' to ISO format and then to integer timestamp
    df['Datetime (local)'] = pd.to_datetime(df[abs_time_col], format='%d/%m/%Y %H:%M:%S')
    df['Timestamp'] = (df['Datetime (local)'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') - utc_offset * 3600

    # Create a new DataFrame with the necessary columns
    new_df = df[['Datetime (local)', 'Timestamp', 'Rel. Time (in s)', 'Tr']].rename(
        columns={'Rel. Time (in s)': 'Time (s)', 'Tr': 'Tr (°C)'}
    )

    # Create the 'analysis' sub-directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'analysis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the original stem of the filename
    filename_stem = os.path.splitext(os.path.basename(xlsx_file))[0]

    # Define the output CSV file path
    output_csv = os.path.join('', f'ts-{filename_stem}.csv')

    # Write the DataFrame to CSV
    new_df.to_csv(output_csv, index=False)

    print(f"CSV file has been saved to {output_csv}")
    
    # Create a temperature graph
    plt.figure()
    plt.grid()
    plt.xlim(new_df['Time (s)'].iloc[0], new_df['Time (s)'].iloc[-1])
    plt.plot(new_df['Time (s)'], new_df['Tr (°C)'], color='k', label='Tr (°C)')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    output_png = os.path.join(output_dir, f'ts-{filename_stem}.png')
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"PNG file has been saved to {output_png}")

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an xlsx file and save it as a CSV file.')
    parser.add_argument('xlsx_file', type=str, help='The path to the xlsx file to be processed')

    args = parser.parse_args()
    main(args.xlsx_file)
