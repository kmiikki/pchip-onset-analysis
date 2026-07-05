#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:15:35 2024

@author: Kim Miikki & Paula Översti
"""

import sys
import csv

def read_fbrm_data(fbrm_file):
    fbrm_data = []
    with open(fbrm_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Expected header: Abs. Time,Timestamp,Total Counts
        for row in reader:
            # Convert timestamp to int if needed
            if row['fouling index'] == 'NA':
                fbrm_data.append({
                    'Abs. Time': row['Abs. Time'],
                    'Timestamp': int(row['Timestamp']),
                    'Total Counts': float(row['Total Counts']),
                    'fouling index': row['fouling index']
                })
            else:
                fbrm_data.append({
                    'Abs. Time': row['Abs. Time'],
                    'Timestamp': int(row['Timestamp']),
                    'Total Counts': float(row['Total Counts']),
                    'fouling index': float(row['fouling index'])
                })
    return fbrm_data

def read_tr_data(tr_file):
    tr_data = []
    with open(tr_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Expected header: Datetime (local),Timestamp,Time (s),Tr (°C)
        for row in reader:
            if row['Tr (°C)'] == '':
                temp = None
            else:    
                temp = float(row['Tr (°C)'])
            tr_data.append({
                'Datetime (local)': row['Datetime (local)'],
                'Timestamp': int(row['Timestamp']),
                'Tr (°C)': temp
            })
    return tr_data

def find_bracketing_points(tr_data, target_timestamp):
    """
    Find two bracketing points in tr_data such that:
    T1 <= target_timestamp <= T2
    where T1 and T2 are from the EA timestamp column.
    
    If target_timestamp is before the first T in tr_data, return first two points for extrapolation.
    If target_timestamp is after the last T in tr_data, return last two points for extrapolation.
    """
    # If target timestamp is before the first entry
    if target_timestamp < tr_data[0]['Timestamp']:
        return tr_data[0], tr_data[1]
    
    # If target timestamp is after the last entry
    if target_timestamp > tr_data[-1]['Timestamp']:
        return tr_data[-2], tr_data[-1]
    
    # Otherwise, find where it fits
    for i in range(len(tr_data)-1):
        T1 = tr_data[i]['Timestamp']
        T2 = tr_data[i+1]['Timestamp']
        if T1 <= target_timestamp <= T2:
            return tr_data[i], tr_data[i+1]
    
    # If for some reason it's not found (which should not happen), default to last two
    return tr_data[-2], tr_data[-1]

def interpolate_tr(T1, Tr1, T2, Tr2, T_target):
    """Linear interpolation (or extrapolation if T_target is outside [T1,T2])"""
    if T2 == T1:
        return Tr1  # Degenerate case, should not happen if data is well-formed
    slope = (Tr2 - Tr1) / (T2 - T1)
    return Tr1 + slope * (T_target - T1)

def main(tr_filename):
    fbrm_file = 'ts-fbrm.csv'  # always the same
    tr_file = tr_filename      # passed as argument

    # Read data
    fbrm_data = read_fbrm_data(fbrm_file)
    tr_data = read_tr_data(tr_file)

    # Compute Tr values for each row in fbrm_data
    output_data = []
    for row in fbrm_data:
        T_target = row['Timestamp']
        
        # If exact timestamp match in tr_data
        exact_match = next((d for d in tr_data if d['Timestamp'] == T_target), None)
        if exact_match is not None:
            Tr_value = exact_match['Tr (°C)']
        else:
            lower_point, upper_point = find_bracketing_points(tr_data, T_target)
            T1, Tr1 = lower_point['Timestamp'], lower_point['Tr (°C)']
            T2, Tr2 = upper_point['Timestamp'], upper_point['Tr (°C)']
            Tr_value = interpolate_tr(T1, Tr1, T2, Tr2, T_target)

        new_row = {
            'Abs. Time': row['Abs. Time'],
            'Timestamp': row['Timestamp'],
            'Total Counts': row['Total Counts'],
            'Tr (°C)': Tr_value,
            'fouling index': row['fouling index']
        }
        output_data.append(new_row)

    # Write output
    with open('ts-fbrm-tr.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Abs. Time', 'Timestamp', 'Total Counts', 'Tr (°C)', 'fouling index'])
        for row in output_data:
            writer.writerow([row['Abs. Time'], row['Timestamp'], row['Total Counts'], row['Tr (°C)'], row['fouling index']])

if __name__ == "__main__":
    # Usage: python script.py "ts-EA-15 (2%H20_EA)_NEW_2s.csv"
    if len(sys.argv) != 2:
        print("Usage: python script.py <tr_data_filename>")
        sys.exit(1)
    
    tr_filename = sys.argv[1]
    main(tr_filename)
