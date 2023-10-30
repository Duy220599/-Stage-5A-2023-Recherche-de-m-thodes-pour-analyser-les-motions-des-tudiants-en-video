# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:12:41 2023

@author: thanh
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
folder_path = r'E:\Projet 2023\Data\Output\Micheal'
file_list = os.listdir(folder_path)
merged_data = {}  # Dictionary to hold merged data
all_headers = []
# Step 2-4: Read file contents, extract headers, and sort them
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_excel(file_path,sheet_name='Sheet1')
    headers = df.columns  # Skip the first column (timestamps)
    all_headers.extend(headers)

    
# Find non-existing headers and add them to a separate header group
# Remove duplicates and sort the headers
all_headers = sorted(set(all_headers))

max_length = 0
#Create DataFrame with header is complete all_header
merged_df = pd.DataFrame(index = range(20), columns=all_headers,dtype = float) #have to specify the datatype

total_length = 0
merged_data = {header: [] for header in all_headers}

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_excel(file_path,sheet_name = 'Sheet1')
    file_length = sum(len(df[column]) for column in df.columns)
    total_length += file_length
    for column in df.columns:
        timestamp = column
        # Check if the column header is a timestamp
        if " - " in column:
            column_data = pd.to_numeric(df[column].reset_index(drop=True), errors='coerce')
            # Check if the column header is already present in the merged DataFrame
            # if timestamp in merged_df.columns:
            if timestamp in all_headers:
                # Concatenate the column data with the existing values in the merged DataFrame
                # column_data = pd.Series(column_data)
                # concatenated_data = pd.concat([merged_df[timestamp], column_data], ignore_index=True,axis = 0)
                # merged_df[timestamp] = concatenated_data
                # print(concatenated_data)
                merged_data[timestamp].extend(column_data)
                max_length = max(max_length, len(merged_data[timestamp]))
                if merged_df[timestamp].dtype != column_data.dtype:
                    print(f"Data types of columns '{timestamp}' do not match.")    
                    
# Fill shorter arrays with NaN to ensure equal length            
for header, values in merged_data.items():
    if len(values) < max_length:
        merged_data[header].extend([np.nan] * (max_length - len(values)))

merged_df = pd.DataFrame(merged_data)
merged_lengths = [merged_df[header].count() for header in merged_df.columns]

# print(total_length)
# print(sum(merged_lengths))
if total_length == sum(merged_lengths): 
    print("Files merged successfully")
    
# output_file_path  = os.path.join(folder_path,'Micheal_merged.xlsx')
# # Export the DataFrame to a new Excel file
# merged_df.to_excel(output_file_path, index=False)