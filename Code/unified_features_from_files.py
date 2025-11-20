import os
import glob
import pandas as pd
import sys
import numpy as np
import traceback
import csv
import globals
from globals import os_name



# Function to process a single file
def process_file(file_path, threshold):
    # Read the file into a DataFrame (assuming it's a CSV file)
    df = pd.read_excel(file_path)

    # Filter rows based on a numerical threshold (assuming the threshold column is named 'threshold_column')
    filtered_df = df[abs(df[g.feature_name_name]) > threshold]

    # Extract names from adjacent columns (assuming 'name_column' and 'adjacent_name_column')
    names = filtered_df[[filter_col_name, g.feature_name_name]]   #.values.flatten()

    return names

def extract_all_files(token, savefig_path):
    try:
        # List all files in the folder with the specified prefix
        file_list = glob.glob(os.path.join(folder_path, f"{file_prefix}*{token}.xlsx"))

        # Initialize an empty list to store all names
        all_names = pd.DataFrame()

        # Loop through the files and process each one
        for file_path in file_list:
            names_from_file = process_file(file_path, numerical_threshold)
            #all_names.extend(names_from_file)

            single_file_df = pd.DataFrame({
                'origin': file_path,
                filter_col_name: names_from_file[filter_col_name],
                g.feature_name_name: names_from_file[g.feature_name_name],
            })
            all_names = pd.concat([all_names, single_file_df], axis=0)

            # Save the comparison DataFrame to a CSV file
        all_names.to_excel(savefig_path + '\selected_top_features_{0}.xlsx'.format(token),
                                 index=True)

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad save_model_results.predcition_vs_gt')


if __name__ == "__main__":
    #globals.savefig_path + '\data_analysis_tool_output\CoxPH_single_{0}_{1}.xlsx'.format(name, token))
    extract_all_files(token = token, savefig_path = globals.savefig_path)



