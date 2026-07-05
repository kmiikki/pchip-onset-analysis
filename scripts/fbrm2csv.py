#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import argparse
import re
import math
# Function to convert datetime to timestamp
def convert_to_timestamp(date_time_str, time_offset):
    dt = datetime.strptime(date_time_str, '%d/%m/%Y %H:%M:%S')
    timestamp = dt.timestamp()
    return int(timestamp - time_offset)

# Extract time offset from the Abs. Time column name
def extract_time_offset(column_name):
    match = re.search(r'UTC\+(\d+)', column_name)
    if match:
        offset_hours = int(match.group(1))
        return offset_hours * 3600  # Convert to seconds
    else:
        return 0

def main(input_file):
    # Read the data from Excel file
    df = pd.read_excel(input_file, header=[0, 1])

    # Print all available columns for debugging
    print("Detected columns:")
    for x in df.columns:
        print(x)

    # Normalize column names: lower case and strip spaces.
    normalized_columns = {(col[0].strip().lower(), col[1].strip().lower()): col for col in df.columns}

    # Extract relative time and total counts columns
    required_columns_regexs = {
        'relative time': 'rel. time \(in s\)',
        'small counts': '.*, no wt, <10 \(primary\)',
        'medium counts': '.*, no wt, 10-100 \(primary\)',
        'large counts': '.*, no wt, 100-1000 \(primary\)',
        'Abs. time': 'abs\. time \(utc\+[0-9]+ : 00\)',
        'fouling index': '.*fouling index \(%\) \(primary\)',
        }
    
    required_matches_columns = dict.fromkeys(required_columns_regexs, None)
    # readable key for column and value is regex for the actual found column name

    for (key0, key1), col in normalized_columns.items():
        for key, regex in required_columns_regexs.items():
            
            if re.match(regex, key0):
                required_matches_columns[key] = col

    missing_columns = [key for key,value in required_matches_columns.items() if value is None]
    if 'fouling index' in missing_columns:
        missing_columns.remove('fouling index')
    if missing_columns:
        raise ValueError("Following required columns not found in the Excel file:\n{}\nExact regex patterns found in fbrm2csv.py".format("\n".join(missing_columns)))

    # Get time offset from the Abs. Time column header
    time_offset = extract_time_offset(required_matches_columns['Abs. time'][0])

    # Extract initial time
    initial_time = df.loc[2, required_matches_columns['Abs. time']]
    initial_dt = datetime.strptime(initial_time, '%d/%m/%Y %H:%M:%S')
    initial_timestamp = convert_to_timestamp(initial_time, time_offset)

    # Create a new DataFrame for results
    results = {
        'Abs. Time': [],
        'Timestamp': [],
        'Total Counts': [],
        'fouling index': []
    }

    # Populate the new DataFrame
    for index, row in df.iterrows():
        if index < 2:  # Skip the header rows
            continue

        abs_time = row[required_matches_columns['Abs. time']]
        total_counts = row[required_matches_columns['small counts']] + \
        row[required_matches_columns['medium counts']] + row[required_matches_columns['large counts']]
        if  math.isnan(total_counts):
            continue
        if pd.notnull(row[required_matches_columns['relative time']]):
            relative_seconds = row[required_matches_columns['relative time']]

            actual_time = initial_dt + timedelta(seconds=relative_seconds)
            timestamp = int(actual_time.timestamp())
            results['Abs. Time'].append(actual_time.strftime('%d/%m/%Y %H:%M:%S'))
            results['Timestamp'].append(timestamp)
            results['Total Counts'].append(total_counts)
            if required_matches_columns['fouling index'] in row:
                results['fouling index'].append(row[required_matches_columns['fouling index']])
            else:
                results['fouling index'].append('NA')

    # Create DataFrame from results dictionary
    results_df = pd.DataFrame(results)

    # Export to CSV
    # output_file = input_file.replace('.xlsx', '.csv')
    output_file = "ts-fbrm.csv"
    results_df.to_csv(output_file, index=False)

    print(f"CSV file created successfully as {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an Excel file to generate a CSV file with specific columns.")
    parser.add_argument('input_file', type=str, help='The path to the input Excel file')
    
    args = parser.parse_args()
    main(args.input_file)
