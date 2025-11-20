import os
import glob
import pandas as pd
import sys
import numpy as np
import traceback
import matplotlib.pyplot as plt
import inspect
import globals as g

from globals import os_name
from globals import root as root
from globals import mature_train, mature_test
from data_loaders import death_name,survival_name, survival_name2


mature_f_to_retain_test = ['Sex','Age','deltaFRS','ALSFRS','Onset','Riluzole']

mature_f_to_retain_train = ['sex','Age_onset','onset','treatment','ALSFRS','slope']
# test align as train names ( and order also - toward mature_data_load end
mature_test_2_train_col_name_change = {'Sex': 'sex','Age': 'Age_onset','deltaFRS':'slope', 'Onset':'onset','Riluzole':'treatment', 'miR_181':'miR-181a*b'}
mature_mirs_prefix_col = ['miR_181']
mature_prefix_2_add = 'mature_'


def mature_data_load(output_labels = False, output_censorship = False, mature_mirs= None,
                     mature_f_to_retain_test = mature_f_to_retain_test, mature_f_to_retain_train = mature_f_to_retain_train):
    try:

        train_input = os.path.join(root,mature_train)
        test_input = os.path.join(root, mature_test)
        
        if os_name == 'Linux':
            df2 = pd.read_csv(test_input, sep='\t', lineterminator='\r')  # windows lineterminator='\r'
            df = pd.read_csv(train_input, sep='\t', lineterminator='\r')
        else:
            df2 = pd.read_csv(test_input, sep='\t',  lineterminator='\r')    # windows lineterminator='\r' - otherwise leave \r which will jeprotize renaming
            df = pd.read_csv(train_input, sep='\t', lineterminator='\r')
        
        df2.drop(index=df2.index[-1], axis=0, inplace=True)
        df2['new_index'] = df2['sample'].str.strip().str.strip('"')
        
        df.drop(index=df.index[-1], axis=0, inplace=True)
        df['new_index'] = df['index.case'].str.strip().str.strip('"')
            

        df2.set_index( 'new_index' ,inplace = True)
        df.set_index('new_index', inplace=True)
    
        df2.drop( 'sample' ,inplace = True, axis = 1)
        df.drop('index.case', inplace=True, axis = 1)
  
        
        if output_labels:

            df2.rename(columns = {'Survival.from.enorlment': 'Survival from enrolment'}, inplace=True)   # and not Survival.from.enrolment spelling error
            df.rename(columns={'Survival_from_enrolment':  survival_name2}, inplace=True)


            mature_f_to_retain_test.append(survival_name2)
            mature_f_to_retain_train.append(survival_name2)
        if output_censorship:
            mature_f_to_retain_test.append(death_name)
            mature_f_to_retain_train.append(death_name)
            df[death_name].astype(int)
            df2[death_name].astype(int)
            

        
        if mature_mirs is not None:
            mature_f_to_retain_test = mature_f_to_retain_test + mature_mirs 
            mature_f_to_retain_train = mature_f_to_retain_train + mature_mirs

        df2 = df2[mature_f_to_retain_test]
        df = df[mature_f_to_retain_train]
  
        df2.rename(columns=mature_test_2_train_col_name_change, inplace=True)   # also in train float file - has - etc.
        df['ismale'] = df['sex'] == 'M'  # one hot encoding: 0 is female 1 is male
        df['istreatment'] = df['treatment'] == 'Yes'
        df['bulbar_onset'] = df['onset'] == 'Bulbar'

        df2['bulbar_onset'] = df2['onset'].str.contains('bulbar')
        df2['istreatment'] = df2['treatment'] == 1.00
        df2 = df2.rename(columns = {'sex':'ismale'})


        for col in mature_mirs_prefix_col:
            df.rename(columns={col: mature_prefix_2_add + col}, inplace=True)
        for col in mature_mirs_prefix_col:
            df2.rename(columns={col: mature_prefix_2_add + col}, inplace=True)

        #enforce indeces as strings
        df.drop(columns=['sex'], inplace=True)
        df.drop(columns=['treatment'], inplace=True)
        df2.drop(columns=['treatment'], inplace=True)
        df.drop(columns=['onset'], inplace=True)
        df2.drop(columns=['onset'], inplace=True)

        df2 = df2.reindex(columns=df.columns)  # align col names
        return df, df2
    except Exception as ex:
        print(''.join(traceback.format_exception(type=type(ex), value=ex, tb=ex.__traceback__)))
        print('bad mature_data_load')


def combine_multiple_inputs(X_train, X_test,second_input = 'mature'):
    try:
        print(f'Input size:{len(X_train)} & {len(X_test)}')
        train_mature, test_mature = mature_data_load()

        # Loop through the indices and compare
        X_train_sorted = X_train.sort_index()
        train_mature_sorted = train_mature.sort_index()
        for index1, index2 in zip(X_train_sorted.index, train_mature_sorted.index):
            if index1 != index2:
                print(f"Train: Indices {index1} and {index2} are not equal.")

        X_test_sorted = X_test.sort_index()
        test_mature_sorted = test_mature.sort_index()
        for index1, index2 in zip(X_test_sorted.index, test_mature_sorted.index):
            if index1 != index2:
                print(f"Test: Indices {index1} and {index2} are not equal.")

        assert (X_test_sorted.index.equals(test_mature_sorted.index))
        assert (X_train_sorted.index.equals(train_mature_sorted.index))

        combined_X_train = pd.concat([X_train, train_mature], axis=1)
        combined_X_test = pd.concat([X_test, test_mature], axis=1)

        print(f'after combine. size:{combined_X_train.shape} & {combined_X_test.shape}')
        return combined_X_train,combined_X_test

    except Exception as ex:
        print(''.join(traceback.format_exception(type=type(ex), value=ex, tb=ex.__traceback__)))
        print('bad combine_multiple_inputs')




if __name__ == "__main__":

    train_mature, test_mature = mature_data_load()
    print(train_mature)
