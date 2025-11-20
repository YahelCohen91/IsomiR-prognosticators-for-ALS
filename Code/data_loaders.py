import os
import platform
import glob, random
import pandas as pd
import sys
import numpy as np
import traceback
import matplotlib.pyplot as plt
import inspect
import copy

import globals
from globals import os_name , summary_tab_name, summary_tab_name_ci
#from unified_features_from_files import token
from summerize_multi_runs_main import multiple_runs_stats_file_name   # , token

from sksurv.datasets import load_whas500

non_isomirs_col_train = ['sex','onset','treatment','Batch','ALSFRS','ALSFRS slope','Age of onset',	'Age of enrolment','Age of death','Survival from onset','Survival from enrolment','Disease duration at enrolment','D50','rD50 at enrolment','outcome','NfL SIMOA']
non_isomirs_col_test_new = ['outcome',	'Survival from onset','Survival from enrolment','sex','Age of enrolment','ALSFRS slope','ALSFRS','Age of onset']
test_2_train_col_name_change = {'Survival.from.onset':'Survival from onset', 'Survival.from.enrolment':'Survival from enrolment', 'Sex_c':'sex', 'CollAge':'Age of enrolment', 'deltaFRS':'ALSFRS slope', 'age_onset':'Age of onset'}

non_isomirs_col_train_float = ['outcome', 'Survival from onset','Survival from enrolment']

from globals import isomiRsII_data_train, isomiRsII_data_test, isomiRsII_data_train_sheet_name
from save_model_results import save_results_file_name

   
test_2_train_col_name_change = {'Survival.from.onset':'Survival from onset', 'Survival.from.enrolment':'Survival from enrolment', 'Sex_c':'sex', 'CollAge':'Age of enrolment', 'deltaFRS':'ALSFRS slope', 'age_onset':'Age of onset'}

perc_selected = f'% selected'
score = f'ImBS_train_median'


death_name = 'outcome'
survival_name = 'Survival from onset'
survival_name2 = 'Survival from enrolment'

def toy_data_load():
    try:
        x, y = load_whas500()

        return x, y

    except Exception as ex:
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        print('bad toy_data_load')

def data_load(break_xy = True):
    try:
        b_float_data = True
        if b_float_data:
            train_input = isomiRsII_data_train
            train_input_sheet = isomiRsII_data_train_sheet_name
            test_input = isomiRsII_data_test
            print('float data type')
            df = pd.read_excel(train_input, sheet_name = train_input_sheet)

            df['new_index'] = df.index
            df.drop(columns=['new_index'], inplace=True)
            df.rename(columns=test_2_train_col_name_change, inplace=True) 
            df.set_index('patient',inplace = True)
            df.index = df.index.astype(object)
        else:
            train_input = isomiRsII_bin_data_train
            test_input = isomiRsII_bin_data_test
            print('binary data type')
            df = pd.read_csv(train_input)

        y = pd.DataFrame(index=df.index)
        y[death_name] = df[death_name]
        y[survival_name2] = df[survival_name2]

        if b_float_data:
            Xt = df.drop(columns=non_isomirs_col_train_float)
        else:
            Xt = df.drop(columns=non_isomirs_col_train)
            Xt = Xt.drop(Xt.columns[[0,1]], axis = 1)
####################################################
        if b_float_data:
            if os_name == 'Linux':
                df2 = pd.read_csv(test_input, sep='\t', lineterminator='\r')  #windows lineterminator='\r'
            else:
                df2 = pd.read_csv(test_input, sep='\t', lineterminator='\r') #windows lineterminator='\r' - otherwise leave \r which will jeprotize renaming
            df2.drop(index=df2.index[-1], axis=0, inplace=True)
            df2['new_index'] = df2.index
            for index, cell_value in df2['new_index'].items():
                df2.at[index, 'new_index'] = eval(cell_value)
            df2.set_index('new_index', inplace=True)

            df2.rename(columns=test_2_train_col_name_change, inplace=True)
        else:
            df2 = pd.read_csv(test_input)


        y2 = pd.DataFrame(index=df2.index)
        y2[death_name] = df2[death_name]
        y2[survival_name2] = df2[survival_name2]

        if b_float_data:
            Xt2 = df2.drop(columns=non_isomirs_col_train_float)
        else:
            Xt2 = df2.drop(columns=non_isomirs_col_test_new)
            Xt2 = Xt2.drop(Xt2.columns[[0, 1]], axis=1)

#######################################################
### drop non mutual isomirs/features
        print('Input: ',Xt.shape)
        print('Input:', Xt2.shape)
        mutual_col_names = Xt.columns.intersection(Xt2.columns)
        Xt = Xt[mutual_col_names]
        Xt2 = Xt2[mutual_col_names]
        print('output - mutual f only:')
        print(Xt.shape)
        print(Xt2.shape)
##########  set types & order
        y['outcome2'] = y['outcome'] == 1.0
        y = y.drop(columns=['outcome'])
        y = y.rename(columns={'outcome2': 'outcome'})
        y = y[['outcome', survival_name2]]  # the bool must come first
        yy = y.to_records(index=False)
        yy['outcome'] = yy['outcome'].astype(bool)

        y2['outcome2'] = y2['outcome'] == 1.0
        y2 = y2.drop(columns=['outcome'])
        y2 = y2.rename(columns={'outcome2': 'outcome'})
        y2 = y2[['outcome', survival_name2]]  # the bool must come first
        yy2 = y2.to_records(index=False)
        yy2['outcome'] = yy2['outcome'].astype(bool)


        X_train = Xt
        X_test = Xt2
        y_train = yy  
        y_test =  yy2 

        if break_xy:
            return X_train, X_test, y_train, y_test
        else:
            col_to_retain = list(mutual_col_names)+[death_name, survival_name, survival_name2]
            df = df[col_to_retain]
            df2 = df2[col_to_retain]
            return df, df2
    except Exception as ex:
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        print('bad data_load')


def select_least_correlated_features(feature_train, feature_test, criteria, name,  token ):
    try:

        correlation_matrix_x_train = np.corrcoef(feature_train, rowvar=False)
        correlation_matrix_x_test = np.corrcoef(feature_test, rowvar=False)

        # Calculate absolute correlations
        abs_correlations = np.abs(correlation_matrix_x_train)

        # Calculate average correlation for each feature
        average_correlations_per_f = abs_correlations.mean(axis=1)

        # Select the least correlated features
        least_correlated_indices = np.argsort(average_correlations_per_f)
        least_correlated_selected = least_correlated_indices[:criteria]

        feature_train_selected = feature_train.iloc[:,least_correlated_selected]
        feature_test_selected = feature_test.iloc[:, least_correlated_selected]

        assert (np.all(feature_train_selected.columns == feature_test_selected.columns))

        results = pd.DataFrame({
            'features': feature_train.columns,
            'average_abs_correlations_per_train_f': average_correlations_per_f,
        })

        results = results.sort_values(by='average_abs_correlations_per_train_f', ascending=True)

        # Create an Excel workbook and add a worksheet
        name_N_path = os.path.join(globals.savefig_path,'least_correnated_train_features_sorted_{0}_{1}.xlsx'.format(name, token))

        # Save the comparison DataFrame to a CSV file
        results.to_excel(name_N_path, index=True)


        # Calculate bin edges for the histograms

        bin_edges = np.linspace(-1, 1, num=2*50+1)  # Adjust the range and number of bins as needed

        # Create histograms for correlation matrices with the same bin edges
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(correlation_matrix_x_train.flatten(), bins=bin_edges, color='blue', alpha=0.7)
        plt.title("Correlation - Train")
        plt.xlabel("Correlation percentile")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(correlation_matrix_x_test.flatten(), bins=bin_edges, color='orange', alpha=0.7)
        plt.title("Correlation - Test")
        plt.xlabel("Correlation Value")
        plt.ylabel("Frequency")

        plt.tight_layout()

        name2 = os.path.join(globals.savefig_path,"least_correnated_features_{0}_{1}.png".format(name, token))
        plt.savefig(name2, bbox_inches='tight')

        plt.close()
        return feature_train_selected, feature_test_selected

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad select_least_correlated_features')



class Unified_features(object):
    filter_col_name = 'features'
    folder_path = 'c:\AlgoCode\Hornstein\isoMIRsII\outputs\pipeline_2nd_floor' 

    def __init__(self, token, savefig_path = globals.savefig_path, feature_name_name = globals.feature_name_name, file_prefix = globals.second_round_file_prefix):
        self.token = token
        self.savefig_path = savefig_path
        self.feature_name_name = feature_name_name
        self.file_prefix = file_prefix

    def process_file(self, file_path, threshold):
        try:
            df = pd.read_excel(file_path)
            r,f = df.shape
            # Filter rows based on a numerical threshold (assuming the threshold column is named 'threshold_column')
            filtered_df = df[abs(df[self.feature_name_name]) >= threshold]
            r2,f2 = filtered_df.shape
            # Extract names from adjacent columns (assuming 'name_column' and 'adjacent_name_column')
            names = filtered_df[[self.filter_col_name, self.feature_name_name]]  
            print('file:'+file_path+'.f before:{0}.after:{1}'.format(r,r2))
            return names

        except Exception as ex:
            print(ex.__context__)
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('bad Unified_features.process_file')


    def extract_all_files(self, save_path, numerical_threshold):
        try:
            # List all files in the folder with the specified prefix
            file_list = glob.glob(os.path.join(self.folder_path, f"{self.file_prefix}*{self.token}.xlsx"))

            # Initialize an empty list to store all names
            all_names = pd.DataFrame()

            # Loop through the files and process each one
            for file_path in file_list:
                names_from_file = self.process_file(file_path, numerical_threshold)

                single_file_df = pd.DataFrame({
                    'origin': file_path,
                    'feature': names_from_file[self.filter_col_name],
                    'feature importance': names_from_file[self.feature_name_name],
                })
                all_names = pd.concat([all_names, single_file_df], axis=0)

                # Save the comparison DataFrame to a CSV file
            all_names.to_excel(self.savefig_path + save_path, index=True)

            return list(all_names['feature'])

        except Exception as ex:
            print(ex.__context__)
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('bad process_file.extract_all_files')






def smooth_data_load(train_asource, test_source):
    try:

        print('smooth data loaded')
        df_train = pd.read_csv(train_asource)
        df_test = pd.read_csv(test_source)

        return df_train, df_test



    except Exception as ex:

        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print('bad {0}'.format(current_function_name))





def re_break_to_xy(train_df, test_df):
    try:
        y_train = pd.DataFrame(index=train_df.index)
        y_train[death_name] = train_df[death_name]
        y_train[survival_name2] = train_df[survival_name2]
        x_train = train_df.drop(y_train.columns.to_list(), axis = 1)

        y_test = pd.DataFrame(index=test_df.index)
        y_test[death_name] = test_df[death_name]
        y_test[survival_name2] = test_df[survival_name2]
        x_test = test_df.drop(y_test.columns.to_list(), axis = 1)

        return x_train, y_train, x_test, y_test



    except Exception as ex:

        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print('bad {0}'.format(current_function_name))



def extract_features_stats_file(current_token, extraction_token, numerical_threshold = None, filter_by = 'precentage'):
    try:
        # Read the data from the input file
        name_N_path_f_input = os.path.join(globals.savefig_path,multiple_runs_stats_file_name.format(extraction_token))
        output_selected_top_features = f"selected_top_features_asof_token_{extraction_token}_created_{current_token}.xlsx"
        name_N_path_f_output = os.path.join(globals.savefig_path, output_selected_top_features)
        df = pd.read_excel(name_N_path_f_input)   #
        # Filter features based on the % selected column
        if filter_by == 'precentage':
            filtered_features = df.loc[df[perc_selected]/100 >= numerical_threshold, 'features']
        elif filter_by == 'score':
            try:
                # determine score threshold by second derivative
                comp_score = 1 - np.sqrt(df[score])
                scores_sorted = np.sort(comp_score)
                
                # Compute the first and second derivatives
                first_derivative = np.diff(scores_sorted)
                second_derivative = np.diff(first_derivative)
                
                # Find the index of the maximum second derivative (the "elbow")
                elbow_index = np.argmax(second_derivative) + 1  # +1 because diff reduces the array size by 1
                numerical_threshold = scores_sorted[elbow_index]
                filtered_features = df.loc[comp_score >= numerical_threshold, 'features']
            
            except Exception as ex:
                print(ex.__context__)
                print('numerical_threshold was already input and should be None')

        ## Save the filtered features to a new file
        #filtered_features.to_csv(name_N_path_f_output, index=False, header=True)

        # Return the filtered features
        print(filtered_features)
        return filtered_features
    
    except Exception as ex:

        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print('bad {0}'.format(current_function_name))


def second_round_selection(second_round_features_selection, DT_token, first_round_DT_token, numerical_threshold = None, sort_by_score = 'score_complement_ImBS_train', filter_by = 'precentage'):
    """

    Args:
        second_round_features_selection (Class object ENUM): from globals to distinguish between filtration by precenatge is the 2nd iteration and by score in the other ones.
        DT_token (time token): time when analysis run for identification
        first_round_DT_token (list of strings/ string): if second_round_features_selection is  from_f_stats_over_multiple_runs_file must be list
                                                        since I want the isomiRs from the first iteration of each model
                                                        second_round_features_selection is from_top_runs_file needs to be string
        numerical_threshold (None/ float):  Defaults to None, value of cutoff by ratio or if None than elbow method for score filtration from the snd floor
        sort_by_score (str, optional): _description_. Defaults to 'score_complement_ImBS_train', by which column of "best_model_per_loop" file to sort data (this can be changed in globals)
        filter_by (str, optional): _description_. Defaults to 'precentage' if to run 2nd floor filtration by precentage cutoff of score cutoff by elbow method.

    Returns:
        _type_: _description_
    """
    try:

        first_round_DT_token_name =   '-'.join(first_round_DT_token)
        output_selected_top_features = "selected_top_features_asof_token_{0}_created_{1}.xlsx".format(first_round_DT_token_name, DT_token)
        name_N_path_f_output = os.path.join(globals.savefig_path, output_selected_top_features)
        
        if (second_round_features_selection.value == globals.second_round_features_selection.from_f_importance_files.value):
            chosen_features = Unified_features(token = first_round_DT_token, savefig_path = globals.savefig_path )
            high_prominant_features_selected = chosen_features.extract_all_files(save_path  = output_selected_top_features, numerical_threshold = numerical_threshold)

            

        if (second_round_features_selection.value == globals.second_round_features_selection.from_f_stats_over_multiple_runs_file.value):
            high_prominant_features_selected = []
            for token in first_round_DT_token:
                if filter_by == 'precentage':
                    high_prominant_features_selected_part = extract_features_stats_file(current_token = DT_token, extraction_token = token, numerical_threshold = numerical_threshold, filter_by = 'precentage')
                elif filter_by == 'score':
                    high_prominant_features_selected_part = extract_features_stats_file(current_token = DT_token, extraction_token = token, numerical_threshold = numerical_threshold, filter_by = 'score')
                print(f'token:{first_round_DT_token}. features:')

                high_prominant_features_selected.append(high_prominant_features_selected_part.to_list())

            single_list = [item for sublist in high_prominant_features_selected for item in sublist]
            high_prominant_features_selected = list(set(single_list))


        if (second_round_features_selection.value == globals.second_round_features_selection.from_list_of_f_names_file.value):
            high_prominant_features_selected = []
            name_N_path_f_input = os.path.join(globals.savefig_path, first_round_DT_token)

            output_selected_top_features = f'selected_top_features_asof_token_{first_round_DT_token}_created_{DT_token}.xlsx'
            name_N_path_f_output = os.path.join(globals.savefig_path, output_selected_top_features)
            high_prominant_features_selected = pd.read_excel(name_N_path_f_input)
            print(f'token:{first_round_DT_token}. features:')

            high_prominant_features_selected = high_prominant_features_selected['selected_features'].to_list()

        if (second_round_features_selection.value == globals.second_round_features_selection.from_top_runs_file.value):
            name_N_path_f_input = os.path.join(globals.savefig_path, save_results_file_name.format(first_round_DT_token))

            stats_summary_df = pd.read_excel(name_N_path_f_input, sheet_name=summary_tab_name_ci)

            df_sorted = stats_summary_df.sort_values(by=sort_by_score, ascending=False)
            
            # get all isomiRs before filtration
            unnamed_columns_ori = [col for col in df_sorted.columns if col.startswith('Unnamed:')]
            All_rel_col_ori = ['features'] + unnamed_columns_ori
            isomiRs_col_df_ori = df_sorted[All_rel_col_ori]
            all_isomirs_values_ori = pd.concat([isomiRs_col_df_ori[col] for col in isomiRs_col_df_ori.columns]).dropna()
            all_isomiRs_ori = set(all_isomirs_values_ori)
            
            # Filter rows where 'score_complement_ImBS_test' > T and extract features
            # determine score threshold by second derivative
            if numerical_threshold is None:
                # Find the index of the peak (the maximum frequency)
                peak_index = np.argmax(np.histogram(df_sorted[sort_by_score], bins='auto')[0])
                max_thresh = np.histogram(df_sorted[sort_by_score], bins = 'auto')[1][peak_index+1]
                
                # Compute the first and second derivatives only to the right of the peak
                scores_right = df_sorted[sort_by_score][df_sorted[sort_by_score] >= max_thresh]
                scores_right.reset_index(drop=True, inplace=True)
                first_derivative = np.diff(scores_right)
                second_derivative = np.diff(first_derivative)
                
                # Find the index of the maximum second derivative (the "elbow")
                elbow_index = np.argmax(second_derivative) + 1  # +1 because diff reduces the array size by 1
                numerical_threshold = scores_right[elbow_index]
                
                # initiate tracking parameters for loop
                best_threshold = numerical_threshold # set to check best minimum threshold if reducution of isomiRs was less than 10
                largest_reduction = 0  # Track the largest reduction found
                
                unique_values = all_isomiRs_ori # for the while loop
                # intiate a while loop to iterate on all score_right to test reduction of at least 10 isomiRs
                while elbow_index >= 0:
                    # get names of features in relevant repeats above threshold
                    filtered_df = df_sorted[df_sorted[sort_by_score] >= numerical_threshold]
                    unnamed_columns = [col for col in filtered_df.columns if col.startswith('Unnamed:')]
                    All_rel_col = ['features'] + unnamed_columns
                    filtered_df = filtered_df[All_rel_col]
                    all_values_series = pd.concat([filtered_df[col] for col in filtered_df.columns]).dropna()
                    # Convert the Series to a set to remove duplicates
                    unique_values = set(all_values_series)
                    
                    # condition to stop loop if the threshold reduced isomiRs number by at least 10
                    reduction = len(all_isomiRs_ori) - len(unique_values)
                    if reduction >= 10:
                        largest_reduction = reduction
                        break
                    
                    # Update best reduction and threshold if it's the largest reduction so far
                    if reduction > largest_reduction:
                        largest_reduction = reduction
                        best_threshold = numerical_threshold
                    
                    # increase the index in case of no reduction in the # of isomiRs passing the threshold
                    elbow_index -= 1
                    if elbow_index >= 0:
                        numerical_threshold = scores_right[elbow_index]
                    else:
                        # If out of bounds, keep the last valid threshold and break the loop
                        break
                
                # if redcution of isomiRs was below 10 return the first threshold that achived this reduction
                # return all runs with this threshold
                if largest_reduction <= 10:
                    numerical_threshold = best_threshold
                    filtered_df = df_sorted[df_sorted[sort_by_score] >= numerical_threshold]
                    unnamed_columns = [col for col in filtered_df.columns if col.startswith('Unnamed:')]
                    All_rel_col = ['features'] + unnamed_columns
                    filtered_df = filtered_df[All_rel_col]
                    all_values_series = pd.concat([filtered_df[col] for col in filtered_df.columns]).dropna()
                    # Convert the Series to a set to remove duplicates
                    unique_values = set(all_values_series)  
            else:
                filtered_df = df_sorted[df_sorted[sort_by_score] >= numerical_threshold]
                unnamed_columns = [col for col in filtered_df.columns if col.startswith('Unnamed:')]
                All_rel_col = ['features'] + unnamed_columns
                filtered_df = filtered_df[All_rel_col]
                all_values_series = pd.concat([filtered_df[col] for col in filtered_df.columns]).dropna()
                unique_values = set(all_values_series)
                largest_reduction = len(all_isomiRs_ori) - len(unique_values)
            
            unique_features_count = all_values_series.value_counts()
            # Create a new DataFrame with feature counts
            high_prominant_features_selected = list(unique_values)
            df = pd.DataFrame(unique_values)
            df = df.rename(columns={0: '2nd floor selected features'})
            df.to_excel(name_N_path_f_output, index=True)
            
            df2 = pd.DataFrame(unique_features_count)
            new_tab_name = 'freq'
            # Open the existing Excel file
            with pd.ExcelWriter(name_N_path_f_output, engine='openpyxl', mode='a') as writer:
                df2.to_excel(writer, sheet_name=new_tab_name, index=True)
            note = f'# of groups:{len(filtered_df)} with T >= {numerical_threshold}.extracted #f:{len(all_values_series)}. unique values #:{len(unique_features_count)}. from col: {sort_by_score}'
            
            df3 = pd.DataFrame(list([note]))            
            with pd.ExcelWriter(name_N_path_f_output, engine='openpyxl', mode='a') as writer:
                df3.to_excel(writer, sheet_name='note', index=True)
            
            if largest_reduction != 0:
                df4 = pd.DataFrame({'score threshold' : numerical_threshold,
                                    'threshold index' : elbow_index,
                                    'isomiRs dropped' : largest_reduction},index=[0])
            else:
                df4 = pd.DataFrame(list(['No reduction of isomiRs found after increasing thershold to maximum']))
            with pd.ExcelWriter(name_N_path_f_output, engine='openpyxl', mode='a') as writer:
                df4.to_excel(writer, sheet_name='threshold', index=True)

            return high_prominant_features_selected

        print('# of features selected for 2nd round:{0}'.format(len(high_prominant_features_selected)))
        #save the file as debug
        df = pd.DataFrame({'2nd floor selected features': high_prominant_features_selected})
        df.to_excel(name_N_path_f_output, index=True)
        
        return high_prominant_features_selected
            
    except Exception as ex:

        print(ex.__context__)
        print(''.join(traceback.format_exception(type=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print('bad {0}'.format(current_function_name))    
    

def extract_dict_items_generator(dictionary, indices):
    for index in indices:
        key = list(dictionary.keys())[index]
        yield key, list(dictionary[key])
        
        
def extract_features_indexes_from_names(participating_features_names):
    try:
        participating_features = []
        X_train, X_test, y_train, y_test = data_load()
        assert list(X_train.columns) == list(X_test.columns), "feautes inputs are not the same."
        participating_features = [X_train.columns.get_loc(col) for col in participating_features_names]

        return participating_features

    except Exception as ex:
            print(ex.__context__)
            print(''.join(traceback.format_exception(type=type(ex), value=ex, tb=ex.__traceback__)))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            current_function_name = inspect.currentframe().f_back.f_code.co_name
            print('bad {0}'.format(current_function_name))

