# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:46:54 2023

@author: thanh
"""

import Extract_values_from_data_by_time_interval
from Extract_values_from_data_by_time_interval import  Extract_values_by_time_interval
from Calculate_similarities_between_columns import Calculate_similarities_Euclidean,Calculate_similarities_Pearsonr,Write_similarities_to_sheet
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import glob
root_folder = 'E:\Projet 2023\Data\Data Oximeter'
"Test parent path"
test_dir = r'E:\Projet 2023\Data\Data Oximeter\16-03-2023\Second session\Micheal'
names_list = []
save_graph_total_path = r'E:\Projet 2023\Data\Graph_Oximeter\Graph total'
output_dir = r'E:\Projet 2023\Data\Output'
# print(os.listdir(os.path.dirname(test_dir)))
#Get name of people from outer folder of current dir
while len(names_list) <= 4:
    for filename in os.listdir(os.path.dirname(test_dir)):
            if not filename.endswith('.png'):
                names_list.append(filename)
print(names_list)
#     #Create figure
# if len(names_list) == 5:
#     for name in names_list:
#     # create the figure with two subplots
#         fig_1, ax1 = plt.subplot(nrows=1, ncols=1, figsize=(200, 20))
#         fig_2, ax2 = plt.figure(nrows=1, ncols=1, figsize=(200, 20))
#         # set the title for the first subplot
#         ax1.set_title(f'Euclidean Similarities of {name}',)
    
#         # set the title for the second subplot
#         ax2.set_title(f'Pearsonr Similarities of {name}')
        
#         # add the figures to the dictionaries
#         figures_euclidean[name] = fig_1
#         # print(grand_figures_euclidean[name])
#         figures_pearsonr[name] = fig_2
#         # print(grand_figures_pearsonr[name])
#Initialize key name for dict
# similarities_euclidean_dict = {name: [] for name in names_list}
# similarities_pearsonr_dict = {name: [] for name in names_list}
# dates= []
# x_values = []
for dirpath, dirnames, filenames in os.walk(root_folder):
    # dirpath is the path of the current directory
    # dirnames is a list of the names of subdirectories in the current directory
    # filenames is a list of the names of files in the current directory
    # print("Current directory:", dirpath)


    for dirname in dirnames:
          subdirectory_path = os.path.join(dirpath, dirname)
        # for file in os.listdir(subdirectory_path):
    # #         print(file)
    # #         if file.endswith('.xlsx'):
    # #             file_path = os.path.join(subdirectory_path,file)
    # #             os.remove(file_path)
          Extract_values_by_time_interval(subdirectory_path)
       
          if dirname in names_list:
                 subdirectory_path = os.path.join(dirpath, dirname)
                 print(subdirectory_path)
             # Loop through the files in the current directory
                 x  = Calculate_similarities_Euclidean(subdirectory_path)
                 y  = Calculate_similarities_Pearsonr(subdirectory_path)
                 Write_similarities_to_sheet(subdirectory_path,x,y)

                 session_name =  os.path.basename(os.path.dirname(subdirectory_path))
                 current_date = os.path.basename(os.path.dirname(os.path.dirname(subdirectory_path)))
                 file_pattern = os.path.join(subdirectory_path, 'output.xlsx')
                 # print(file_pattern)
                
                 for input_file_path in glob.glob(file_pattern):
                     # Construct the output file path with a unique name
                    
                     output_file_name = os.path.basename(input_file_path)
                     output_file_name = f"{current_date}_{session_name}_{output_file_name}"
                     output_dir_path = os.path.join(output_dir, dirname)
        
                     # for file in os.listdir(output_dir_path):
                     #     if file.endswith("xlsx"):
                     #         file_path = os.path.join(output_dir_path,file)
                     #         os.remove(file_path)
                     
                     output_file_path = os.path.join(output_dir_path, output_file_name)
                     # Copy the input file to the output directory
                     shutil.copy2(input_file_path, output_file_path)

#     for dirname in dirnames:
#         # if dirname.lowercase() == 'micheal':
            
#         if dirname in names_list:
# #             # print(dirname)
#             subdirectory_path = os.path.join(dirpath, dirname)
# #            
#             Extract_values_from_data(subdirectory_path)
#             x  = Calculate_similarities_Euclidean(subdirectory_path)
#             y  = Calculate_similarities_Pearsonr(subdirectory_path)
#             Write_similarities_to_sheet(subdirectory_path,x,y)
#             # if x is not None:
#             #     # similarities_euclidean_dict[dirname].append(x)
#             #     similarities_pearsonr_dict[dirname].append(y)
# #             # similarities_pearsonr_dict[dirname] = y

#             current_date = os.path.basename(os.path.dirname(os.path.dirname(subdirectory_path)))
            #             if current_date not in dates:
            #                 dates.append(current_date)
            
# for name in names_list:
#     # similarities_euclidean_arrays = similarities_euclidean_dict[name]
#     similarities_pearsonr_arrays = similarities_pearsonr_dict[name]
#     # similarities_euclidean_arrays = [arr for arr in similarities_euclidean_arrays if np.sum(arr) > 0]
#     similarities_pearsonr_arrays = [arr for arr in similarities_pearsonr_arrays if np.sum(arr) != 0]
    
#     print(f'Array of {name}')
#     print(similarities_euclidean_arrays)
#     Find the longest array 
#     max_length = max(len(arr) for arr in similarities_euclidean_arrays)
    
#     print(f'max_length for {name}:', max_length)
#     longest_array = max(similarities_euclidean_arrays, key=len)
#     print(longest_array)
#     x_values = np.arange(1, max_length+ 1) #Use the longest array's length to set the x-axis values
#     print(len(x_values))
#     # Create a single plot for this person
#     fig, ax = plt.subplots(figsize=(10, 4))

#     # Plot each array on the same plot
#     for i, arr in enumerate(similarities_euclidean_arrays):
#             diff = max_length - len(arr)
#             arr_padded = np.pad(arr, (0, diff), constant_values=np.nan, mode='constant')
#             print(f'length arr_padded for {name} - {i+1}:', len(arr_padded))
#             ax.plot(x_values, arr_padded, label=f'{name} - {i+1}')
#     ax.set_ylim(0, 80)
#     ax.set_xticks(x_values[:len(arr_padded)-1])
#     ax.set_xticklabels([f'{j} - {j+1}' for j in range(1, max_length)], fontsize=5)
#     ax.set_title(f"Euclidean Similarities of {name}", fontsize=10, fontweight='bold')
#     ax.set_xlabel('Time (minutes)', fontsize=8, fontweight='bold')
#     ax.set_ylabel('Similarity', fontsize=8, fontweight='bold')
#     ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.6), frameon=False)
#     save_path  = os.path.join(save_graph_total_path,f'{name}_euclidean.jpg')
#     plt.savefig(save_path, dpi = 300)
#     plt.tight_layout(pad=3)
#     plt.show()
    
    
#     max_length = max(len(arr) for arr in similarities_pearsonr_arrays)
    
#     print(f'max_length for {name}:', max_length)
#     longest_array = max(similarities_pearsonr_arrays, key=len)
#     print(longest_array)
#     x_values = np.arange(1, max_length+ 1) #Use the longest array's length to set the x-axis values
#     print(len(x_values))
#     # Create a single plot for this person
#     fig, ax = plt.subplots(figsize=(10, 4))

#     # Plot each array on the same plot
#     for i, arr in enumerate(similarities_pearsonr_arrays):
#             diff = max_length - len(arr)
#             arr_padded = np.pad(arr, (0, diff), constant_values=np.nan, mode='constant')
#             print(f'length arr_padded for {name} - {i+1}:', len(arr_padded))
#             ax.plot(x_values, arr_padded, label=f'{name} - {i+1}')
#     ax.set_ylim(-1.5,1.5)
#     ax.set_xticks(x_values[:len(arr_padded)-1])
#     ax.set_xticklabels([f'{j} - {j+1}' for j in range(1, max_length)], fontsize=5)
#     ax.set_title(f"Pearsonr Similarities of {name}", fontsize=10, fontweight='bold')
#     ax.set_xlabel('Time (minutes)', fontsize=8, fontweight='bold')
#     ax.set_ylabel('Similarity', fontsize=8, fontweight='bold')
#     ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.6), frameon=False)
#     save_path  = os.path.join(save_graph_total_path,f'{name}_pearsonr.jpg')
#     plt.savefig(save_path, dpi = 300)
#     plt.tight_layout(pad=3)
#     plt.show()
    
    # for filename in filenames:
    #       file_path = os.path.join(subdirectory_path,filename)
    #       Extract_values_from_data(filename)
