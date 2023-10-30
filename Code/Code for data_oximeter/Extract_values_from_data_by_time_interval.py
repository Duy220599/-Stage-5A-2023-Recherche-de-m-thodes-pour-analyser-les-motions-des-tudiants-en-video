# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:18:40 2023

@author: thanh
"""

import pandas as pd 
import numpy as np
import csv
from datetime import datetime,timedelta
import scipy.interpolate as interp
import os
# parent_dir = r'E:\Projet 2023\Data\Data Oximeter\08-03-2023\Yeliz'

# "Test parent path"
parent_dir = r'E:\FileHistory'
# def Extract_by_time_interval_and_interpolate(directory):
def Extract_values_by_time_interval(parent_dir):
    for file in os.listdir(parent_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(parent_dir,file)
    
            df = pd.read_csv(file_path,header = None, names = ['time','value'])
            #Drop NaN
            df.dropna(inplace=True)
            "This is an attempt to fix bug "
            # Create a copy of the 'time' column
            df['time_copy'] = df['time']
            print(df['time_copy'])
            # Remove leading/trailing whitespaces from the 'time' column
            df['time_copy'] = df['time_copy'].str.strip()
           
            # Extract the time part before the semicolon
            df['time_copy'] = df['time_copy'].str.split(';').str[0]
            
            # Convert the 'value' column to string data type
            df['value'] = df['value'].astype(str)
            
            # Extract integer and decimal parts of 'value' column
            df['value_integer'] = df['time'].str.split(';').str[1]
            df['value_decimal'] = df['value']
            print(df['value_integer'])
            
            # Convert the 'value_integer' column to integer
            df['value_integer'] = df['value_integer'].astype(float)
            df['value_decimal'] = df['value_decimal'].astype(float)
            
            # Replace semicolon with dot in the 'value' column
            df['value'] = df['value'].str.replace(',', '.')
            
            # Combine 'value_integer' and 'value_decimal' columns to get the final 'value' column
            df['value'] = df['value_integer'] + df['value_decimal'] / 100
            
            
            # Drop the intermediate columns
            df.drop(['value_integer', 'value_decimal'], axis=1, inplace=True)
            
            # Drop the original 'time' column
            df.drop(['time'], axis=1, inplace=True)
            
            # Rename the 'time_copy' column to 'time'
            df.rename(columns={'time_copy': 'time'}, inplace=True)
            """"""""""""""""""""""""
            # Convert 'time' column to datetime object
            df['time'] = pd.to_datetime(df['time'], format='%H:%M')
            
            #Sort the value follow increasing time-order
            df = df.sort_values(by = 'time')
            #Swap the 2 columns
            df = df.reindex(columns=['time', 'value'])
            #Transform back to string object
            df['time'] = df['time'].dt.strftime('%H:%M')
            
            #Use the first value of 'time' column as start_time
            df['time'] = pd.to_datetime(df['time'], format='%H:%M')
            start_time = df['time'].iloc[0]
            
            "Code for 1min interval"
            #Create an array to store the value between interval
            time_interval = 1 # time interval 1 min
            df_interval = pd.DataFrame()
            values = []
            df['time'] = pd.to_datetime(df['time'], format='%H:%M')
            
            "Code for 1 min interval"
            #Get the maximum size of each interval can have by group df by 1 min interval ,then find the max length of the whole df
            max_length = df.groupby(pd.Grouper(key='time', freq='1Min')).size().max()
            
            while start_time <= df['time'].max():
                #Calculate the start and end time of the interval
    
                end_time = start_time + timedelta(minutes=time_interval)
               
                interval_values = df[(df['time'] >= start_time) & (df['time'] < end_time)]
                
                # Reset index of filtered values
                interval_values = interval_values.reset_index(drop=True)
                
                # Get the maximum length of the data in the interval
                max_length_interval = interval_values.shape[0]

                # Create column header as interval start time - end time
                column_header = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
                print(column_header)
                #If this is less than max_length, the code appends NaN values to interval_values,if not have this line the df will be truncated 
                interval_values = pd.concat([interval_values, pd.DataFrame(index=range(max_length_interval, max_length), columns=interval_values.columns)], ignore_index=True)
                print(interval_values )
                # Append values to df_interval DataFrame with column header
                df_interval.loc[:, column_header] = interval_values['value']
                # df_interval[column_header] = interval_values['value'].dropna()
                print(df_interval)
                # Update start time for next interval as end time + 1 minute
                start_time = end_time 
     
            # "Code for 5 min interval"
            # Start the loop thorough the time column
            # while start_time <= df['time'].max():
            #     #Calculate the start and end time of the interval
            #     end_time = start_time + timedelta(minutes=time_interval)
            #     interval_values = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
                 
            #     # Reset index of filtered values
            #     interval_values = interval_values.reset_index(drop=True)
                
            #     # Create column header as interval start time - end time
            #     column_header = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
                
            #     # Append values to df_interval DataFrame with column header
            #     df_interval[column_header] = interval_values['value']
                
            #     # Update start time for next interval as end time + 1 minute
            #     start_time = end_time + timedelta(minutes=1)
            
            # #Step of interpolation of data for missing data
            # Find the longest length 
            max_length = df_interval.count().max()
            
            longest_columns = []
            # Find the columns) with the specified length
            for col in df_interval.columns:
                if (df_interval[col]).count() == max_length: #count method exclude NaN value
                    longest_columns.append(col)
                    
            # print(f'The longest length in the table is: {max_length}')
            # print(f'The first column with the length of {max_length} is: {longest_columns}')
            #"Choose the first column as reference
            ref_column = df_interval[longest_columns[0]]
            sorted_indices = np.argsort(ref_column)
            # #Perfom interpolation on other column
            for col in df_interval.columns:
                column = df_interval[col]
                if column.count() < max_length:
                    column = column.interpolate(method = 'linear',axis = 0,inplace = False,limit_direction = 'both')
                    #update value 
                    df_interval[col] = column
            print(df_interval)
            # if os.path.exists('output-1-min.xlsx'):
            #     print('Output file exist')
            #     output_path_1 = os.path.join(parent_dir,'output-5-min.xlsx')
            #     df_interval.to_excel(output_path_1,index = False)
            #     break
            # print(df_interval)
            output_path = os.path.join(parent_dir,'output.xlsx')
            df_interval.to_excel(output_path,index = False)
 



# Extract_values_from_data(parent_dir)
Extract_values_by_time_interval(parent_dir)