import debugpy
import copy
#debugpy.listen(('0.0.0.0', 5678))
import sys,os
import platform
import numpy as np
import matplotlib.pyplot as plt
import threading
import pandas as pd
import traceback
import datetime
import re
import mlflow
import mlflow.sklearn
import inspect
import time
import warnings


from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
#from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer 

from sklearn.feature_selection import f_regression, r_regression
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest  #not RandomForestClassifier - for main model. ok for f selection

from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
#from sklearn.feature_selection import SequentialFeatureSelector as SFS  # -- mlxtend SFS instead

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_tweedie_deviance, mean_pinball_loss, mean_absolute_error

#from isomiRsII_main import data_load
from data_loaders import data_load  # replacing
from isomiRsII_main import Get_sorted_features_range, Load_sorted_features_range
from isomiRsII_main import normalize
from isomiRsII_main import CoxPH_single
from save_model_results import save_results

from data_view_lib import auc_time_dependant_tool
from data_view_lib import f_importance_tool


from models_helpers import coxph_fit_and_score_features, best_model_preparations
from models_helpers import calculate_survival_scores, select_features, my_IBS, my_ImBS, InclusiveSurvivalKFold, ci_score
from data_loaders import select_least_correlated_features
from data_loaders import Unified_features, smooth_data_load, re_break_to_xy
from data_loaders import extract_dict_items_generator
from data_loaders import second_round_selection, extract_features_indexes_from_names

from features_combinations import generate_features_combinations_file, remove_features_combinations_files, fill_random_numbers, read_features_combinations_file

from Extract_mature_data import combine_multiple_inputs, mature_data_load
from correlations_helpers import anchors_finder, smooth_data_prepare


#from  globals import savefig_path
import globals as globals
from globals import os_name
random_seed = 0
#DT_token = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
from globals import Usecase
from globals import Usecase_main_model as main_model_type
from globals import Usecase_features_selection as features_selection
from globals import first_round_DT_token as first_round_DT_token
#from globals import savefig_path, savefig_path_internal, folder_path, folder_path_internal

from stats_on_cohort import cohort_summary_features_stats, cohort_summary_records_stats, cohort_summary_features_stats_deaths_seperated
from stats_on_cohort import survival_grouping_on_events
from save_model_results import save_results_file_name, save_events_file_name
from summerize_multi_runs_main import summerize_multi_runs_main
from features_combinations import rearrange_rerun, associate_groups_into_unified_thread_run

from globals import isomiRsII_data_train, isomiRsII_data_test, mature_train, mature_test

survival_name = 'Survival from onset'
survival_name2 = 'Survival from enrolment'
death_name = 'outcome'

log_output_file_name = 'log_{0}_{1}.txt'

def isomiRs_main_worker_thread(Usecase_main_model, Usecase_pipeline, Usecase_feature_selection,
                               use_different_score_metrics, second_round_features_selection,multi_files_input, features,
                             features_combinations = None, thread_name = None, final_model = False,actual_survival_times = False):

    try:
        #for fix number of features: model_max_num_f = 14 model_min_num_f = 15  --> gives 15
        model_max_num_f = 14 # 6 #21       #14     #0  # 14 # 20 # 9 #40 #60 # 42 # 20  # 1552  # in bestkfolds
        model_min_num_f = 15 # 17  # 10  # 1  #15 # 15 #20
        model_step_num_f = 1 #20
        verbose_level = 1 #10
        ###################################  inital
        b_perform_decorr_sort = False
        b_perform_stats_analysis = False
        ################################

            ########### gaussian data smoothing
        
        window_size_smoother = 5  # 20 # 50# 2  # windif half size
        minimal_range_smooth_indeces_setting = 3
        b_smooth_data = False
        b_load_saved_smooth = True
        ####################################################
        #todo - move this to a specific place
        main_model_param_grid = {'learning_rate': [0.1, 0.5, 0.7],
                    'subsample': [0.3, 0.5, 0.7],
                    'n_estimators': [10, 20, 50],
                    'dropout_rate': [0.3,0.7],
                    'loss': ['coxph','squared', 'ipcwls'],
                    'random_state': [0,42],
                    'max_depth': [2, 3]
                    }

        main_model_param_grid = {}

        run_name = 'pipe_{0}_model_{1}'.format(Usecase_pipeline.value, str(Usecase_main_model.name))

        if b_use_generated_features_combinations:
            if b_generate_features_combinations:
                print("---------------use & generate_features_combinations ---------------")
            else:
                print("---------------use *previously* generate_features_combinations ---------------")

        X_train, X_test, y_train, y_test = data_load()
        if multi_files_input:
            X_train, X_test = combine_multiple_inputs(X_train, X_test,second_input = 'mature')   # this is non permutated !

        if b_perform_stats_analysis:
            cohort_summary_records_stats(X_train, y_train[death_name], name = 'train')
            cohort_summary_records_stats(X_test, y_test[death_name], name='test')
            cohort_summary_features_stats(X_train, std_amount = 4, name = 'train')
            cohort_summary_features_stats(X_test, std_amount=4, name='test')
            cohort_summary_features_stats_deaths_seperated(X_train, y_train[death_name], std_amount=4, name='train')
            cohort_summary_features_stats_deaths_seperated(X_test, y_test[death_name], std_amount=4, name='test')

        num_pateints,num_f_before = X_train.shape

        if b_perform_decorr_sort:
                X_train, X_test = select_least_correlated_features(feature_train = X_train, feature_test = X_test, criteria = num_f_least_corr_to_select, name = run_name, token = DT_token)


    
        num_pateints, num_f_after =  X_train.shape
        X_train, X_test = normalize(X_train, X_test, y_train, y_test, normalize_gt=False)


        lower_time_perc = 10; upper_time_perc = 90
        lower_test_perc, upper_test_perc = np.percentile(y_test[survival_name2], [lower_time_perc, upper_time_perc])
        lower_train_perc, upper_train_perc = np.percentile(y_train[survival_name2], [lower_time_perc, upper_time_perc])
        lower_bound_prediction_time = y_train[survival_name2].min()
        upper_bound_prediction_time = y_train[survival_name2].max()
        lower_time = y_test[survival_name2].min()
        upper_time = y_test[survival_name2].max()
        if not actual_survival_times:
            lower_bound_prediction_time = max(lower_bound_prediction_time+2,lower_time) 
            upper_bound_prediction_time = min(upper_bound_prediction_time - 2, upper_time)
            prediction_vector_times = np.arange(round(lower_bound_prediction_time), round(upper_bound_prediction_time))
        else:
            if y_train[survival_name2].max() > y_test[survival_name2].max():
                prediction_vector_times_test = y_test[survival_name2]
                prediction_vector_times_train = y_train[survival_name2][y_train[survival_name2] <= y_test[survival_name2].max()]
            else:
                prediction_vector_times_test = y_test[survival_name2][y_test[survival_name2] <= y_train[survival_name2].max()]
                prediction_vector_times_train = y_train[survival_name2]
            
            if prediction_vector_times_train.min() <= prediction_vector_times_test.min():
                prediction_vector_times_train = prediction_vector_times_train[prediction_vector_times_train >= prediction_vector_times_test.min()]
            else:
                prediction_vector_times_test = prediction_vector_times_test[prediction_vector_times_test >= prediction_vector_times_train.min()]
            
            prediction_vector_times_train = np.unique(prediction_vector_times_train)
            prediction_vector_times_train = np.sort(prediction_vector_times_train)
            prediction_vector_times_train = np.delete(prediction_vector_times_train, -1)
            
            prediction_vector_times_test = np.unique(prediction_vector_times_test)
            prediction_vector_times_test = np.sort(prediction_vector_times_test)
            prediction_vector_times_test = np.delete(prediction_vector_times_test, -1)
            

        if use_different_score_metrics:
            scoring_type = 'Brier_Family'
            scoring_name = 'ImBS'  # brier   #should be the same as the my_ImBS in SFS or other objects that activate models (SFS,GridSearchCV etc.)
        else:
            scoring_type = 'ci'
            scoring_name = 'ci'

        print(f'Scoring type: {scoring_type}. Specificlly: {scoring_name}')
        ########################  smooth  input   ##########
        if b_smooth_data:
            save_path = f'c:\AlgoCode\Hornstein\isoMIRsII\outputs\corr_play_output'

            isomiRsII_smooth_data_train = save_path + '\smoooth_corrected_umi_UK_2132023.txt'
            isomiRsII_smooth_data_test = save_path + '\smoooth_corrected_umi_replication_new_analysis_censored_by_Endpt_reason.txt'

            helper = '_win_{0}_minrange_{1}.txt'.format(window_size_smoother, minimal_range_smooth_indeces_setting)
            helper_train = re.sub('.txt$', helper, isomiRsII_smooth_data_train)  # $ remove substring
            helper_test = re.sub('.txt$', helper, isomiRsII_smooth_data_test)

            if not b_load_saved_smooth:

                train_df = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)

                train_df = train_df.sort_values(by=[survival_name2], ascending=False)
                test_df = test_df.sort_values(by=[survival_name2], ascending=False)

                train_df, test_df = smooth_data_prepare(train_df, test_df, survival_name2,
                                                        smooth_range=window_size_smoother,minimal_range_smooth =  minimal_range_smooth_indeces_setting,
                                                        symmetric_gausssian_range=[3, 101])

                train_df.to_csv(helper_train)
                test_df.to_csv(helper_test)
                print('smooth data')

            else:

                train_df, test_df = smooth_data_load(train_asource=helper_train, test_source=helper_test)
                print('load smoothed data')

            x_train, y_train, x_test, y_test = re_break_to_xy(train_df, test_df)
        #########################################

        [num_patients ,max_f_num] = X_train.shape

        f_ranges_start = np.arange(first_group_f_start, last_group_f_start, 1)

        gen_object = None

        #### preproc - specific for each usecase" pipeline + main_model
        if Usecase_main_model == main_model_type.Cox:
            #main_model = CoxPHSurvivalAnalysis(alpha=0, ties='breslow', n_iter=100, tol=1e-09, verbose=1)
            n_alphas = 1  #, alphas = [0.001]
            main_model = CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True , alphas = [0.001])
            # fit_baseline_model=True to enable base & predict_survival_function vector
            if n_alphas > 1:  #was alpha_min_ratio=0.01 'auto'  , alphas = [0.001] alpha_min_ratio=0.01,
                print('need n_alphas optimization')

            main_model_params = {}
            main_model.set_params(**main_model_params)
        if Usecase_main_model == main_model_type.Cox_Regulation_free:
            main_model = CoxPHSurvivalAnalysis(alpha=0, ties='breslow', n_iter=100, tol=1e-09, verbose=1)
            main_model_params = {}
            main_model.set_params(**main_model_params)

        elif Usecase_main_model == main_model_type.Boosting:
            main_model = GradientBoostingSurvivalAnalysis()
            main_model.set_params(n_estimators= 20, learning_rate=0.5,  max_depth=2, subsample = 0.3,
                                loss= 'coxph',
                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0,
                                max_features=None, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=None,
                                tol=0.0001, dropout_rate=0.3, verbose=1, ccp_alpha=0.0,
                                random_state=42 )

        elif Usecase_main_model == main_model_type.Bagging:
            main_model = RandomSurvivalForest()
            main_model.set_params(n_estimators= 2 , max_depth = 4,  
                                min_samples_split=2, min_samples_leaf=2,
                                max_features= 0.5,max_samples= 0.5,
                                min_weight_fraction_leaf=0.0,max_leaf_nodes=None,
                                bootstrap=True,  
                                n_jobs=-1, random_state=42, oob_score=False, verbose=verbose_level)



        elif Usecase_main_model == main_model_type.Without:
            main_model = None
        else:
            print('wrong- no main_model type was set')

        ##### features selection
        selector = None
        param_grid = {}

        if Usecase_feature_selection == features_selection.Without:

            selector = None

            param_grid = {}

        elif Usecase_feature_selection == features_selection.Survival_Univariate_cox:

            selector = SelectKBest(score_func=coxph_fit_and_score_features, k=model_max_num_f)  # bug
            param_grid = {"select__k": np.arange(model_min_num_f, model_max_num_f+1, model_step_num_f)}


        elif Usecase_feature_selection == features_selection.Stats:

            selector =  SelectKBest(score_func =r_regression, k=model_max_num_f) #f_classif   f_regression
            param_grid = {"select__k": np.arange(model_min_num_f, model_max_num_f+1, model_step_num_f)}


        elif Usecase_feature_selection == features_selection.Extra_tree:

            clf = ExtraTreesClassifier(n_estimators=100,random_state=42)
            selector = SelectFromModel(estimator=clf, threshold="1.0 * median")

            param_grid = {"select__max_features": np.arange(model_min_num_f, model_max_num_f + 1, model_step_num_f),
                        "select__threshold": ["0.05 * median", "0.1 * median", "0.3 * median", "1.0 * median"]}

        elif Usecase_feature_selection == features_selection.RF_Classification:


            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator=clf, threshold="1.0 * median")

            param_grid = {"select__max_features": np.arange(model_min_num_f, model_max_num_f + 1, model_step_num_f), # [0.3,0.5, 0.7], #
                        "select__threshold": ["0.05 * median", "0.1 * median", "0.3 * median", "1.0 * median"]}

        elif Usecase_feature_selection == features_selection.RF_Regression:

            #RandomSurvivalForest  RandomForestClassifier
            clf = RandomForestRegressor(n_estimators=10,  criterion='absolute_error', max_depth  = 3, min_samples_split=2, min_samples_leaf=1, max_features  = 0.5,
                                    max_leaf_nodes = None, bootstrap  = True, max_samples  = 0.5, # max_leaf_nodes - define
                                        min_weight_fraction_leaf = 0.0, min_impurity_decrease = 0.0,
                                    n_jobs=-1,random_state=42, oob_score=True, verbose = verbose_level)

            selector = SelectFromModel(estimator=clf)
            param_grid = {}

        elif Usecase_feature_selection == features_selection.RF_Permutation:

            clf = RandomSurvivalForest(n_estimators=10, min_samples_split=5, min_samples_leaf=5, n_jobs=-1, random_state=42)


            param_grid = {"select__max_features": np.arange(model_min_num_f, model_max_num_f + 1, model_step_num_f),
                        "select__threshold": ["0.05 * median", "0.1 * median", "0.3 * median", "1.0 * median"]}

        elif Usecase_feature_selection == features_selection.SFS_Extra_tree:

            clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

            selector = SFS(estimator=clf, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        scoring= scoring_type,   # 'r2',  # 'neg_mean_squared_error',
                        verbose=2,
                        n_jobs=-1,
                        cv=3)
            param_grid = {}

        elif Usecase_feature_selection == features_selection.SFS_KNN:

            knn = KNeighborsClassifier(n_neighbors=3)
            selector = SFS(estimator=knn, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        scoring= scoring_type,   #'r2',  # 'neg_mean_squared_error',
                        verbose=2,
                        n_jobs=-1,
                        cv=3)

            param_grid = {}

        elif Usecase_feature_selection == features_selection.SFS_RF_Regression:
            'friedman_mse', 'poisson', 'squared_error', 'absolute_error'
            clf = RandomForestRegressor(n_estimators=15, criterion='absolute_error', max_depth=2, min_samples_split=3,
                                        min_samples_leaf=1, max_features=0.5,
                                        max_leaf_nodes=None, bootstrap=True, max_samples=0.25,  # max_leaf_nodes - define
                                        min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0,
                                        n_jobs=-1, random_state=42, oob_score=True, verbose=verbose_level)

            selector = SFS(estimator=clf,  k_features=(model_min_num_f, model_max_num_f + 1),
                    forward=True,
                    floating=False,
                    scoring= scoring_type,   # 'neg_mean_squared_error', # 'r2', #'neg_mean_squared_error',
                    verbose=2,
                    n_jobs=-1,
                    cv=3)

            param_grid = {}


        elif Usecase_feature_selection == features_selection.SFS_RF_Classification:
            #RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            selector = SFS(estimator=clf, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        scoring= scoring_type,   # 'r2',  # 'neg_mean_squared_error',
                        verbose=2,
                        n_jobs=-1,
                        cv=0)



            param_grid = {}


        elif Usecase_feature_selection == features_selection.SFS_RF_Permutation:


            clf = permutation_importance(n_repeats = 10, random_state=42)

            selector = SFS(estimator=clf, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        scoring=scoring_type,   # 'r2',  # 'neg_mean_squared_error',
                        verbose=2,
                        n_jobs=-1,
                        cv=3)



            param_grid = {}

        elif Usecase_feature_selection == features_selection.SFS_CoxPH:

            coxph = CoxPHSurvivalAnalysis(alpha=0, ties='breslow', n_iter=100, tol=1e-09, verbose=1)
            selector = SFS(estimator=coxph, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        scoring= scoring_type,   # 'r2',  # 'neg_mean_squared_error',  'neg_brier_score', #
                        verbose=2,
                        n_jobs=-1,
                        cv=0)  #3

            param_grid = {}

        elif Usecase_feature_selection == features_selection.SFS_CoxNet:
            coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True)
            selector = SFS(estimator=coxnet, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        #scoring= 'r2',  # 'neg_mean_squared_error',  'neg_brier_score', #
                        scoring= scoring_type,
                        verbose=2,
                        n_jobs=-1,
                        cv=0)  #3

            param_grid = {}

        elif Usecase_feature_selection == features_selection.SFS_RSF:
            rsf = RandomSurvivalForest(max_depth=2, random_state=1)
            selector = SFS(estimator=rsf , k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        #scoring= 'neg_brier_score', #'r2',  # 'neg_mean_squared_error',  'neg_brier_score', #
                        scoring= scoring_type,
                        verbose=2,
                        n_jobs=1,  
                        cv=0)  
            # selector = SFS(estimator=rsf, n_features_to_select='auto',
            #                direction='forward',
            #                #scoring='neg_brier_score',  # 'r2',  # 'neg_mean_squared_error',  'neg_brier_score', #
            #                 scoring= scoring_type,
            #                n_jobs=-1,
            #                # Error: i think my scoring_type is not thread safe/sfs mxlmn dont know how to eat it cv> 0 or  n_jobs<>1
            #                cv=2)  # 3

            param_grid = {}

        elif Usecase_feature_selection == features_selection.SFS_XGB_Survival:
            xgb = GradientBoostingSurvivalAnalysis()
            xgb.set_params(n_estimators=20, learning_rate=0.5, max_depth=2, subsample=0.3,
                                # scale_pos_weight = 0.1 - non v
                                loss=score_type,  # 'squared', # 'coxph',
                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0,
                                max_features=None, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=None,
                                tol=0.0001, dropout_rate=0.3, verbose=1, ccp_alpha=0.0,
                                random_state=0)
            selector = SFS(estimator=xgb, k_features=(model_min_num_f, model_max_num_f + 1),
                        forward=True,
                        floating=False,
                        # scoring= 'r2',  # 'neg_mean_squared_error',  'neg_brier_score', #
                        scoring=scoring_type,
                        verbose=2,
                        n_jobs=-1,
                        cv=3)  # 3

            param_grid = {}

        else:
            print('wrong- no features selection was set')

        ##################### loops
        ii = np.arange(len(features_combinations))
        extract_group_gen = extract_dict_items_generator(features_combinations, ii)
        for i in f_ranges_start:
            if not b_use_generated_features_combinations:
                ggi  = i
                start_n_f = i
                end_n_f = min(i + f_ranges_len, max_f_num)
                print('features: {0} - {1}'.format(start_n_f,end_n_f))
                mlflow.log_metrics({"features-start":start_n_f, "features-end": end_n_f})

                X_train_filtered = X_train.iloc[:, start_n_f:end_n_f]
                X_test_filtered = X_test.iloc[:, start_n_f:end_n_f]
                y_train_filtered = y_train
                y_test_filtered = y_test
                input_features_names = X_train.columns[start_n_f:end_n_f]
                run_name = 'pipe_{0}_model_{1}_fs{2}_fe{3}_ggi{4}_{5}'.format(Usecase_pipeline.value, str(Usecase_main_model.name + '_scoring_' + scoring_name), start_n_f, end_n_f,ggi, thread_name)

            else:
                try:
                    ggi, features_combinations = next(extract_group_gen)
                except StopIteration:
                    break
                
                
                print('thread:{1}.group#{0}.global group index#{2}'.format(i,thread_name,ggi))

                X_train_filtered = X_train.iloc[:, features_combinations]
                X_test_filtered = X_test.iloc[:, features_combinations]
                y_train_filtered = y_train
                y_test_filtered = y_test
                input_features_names = X_train.columns[features_combinations]
                print(f'f index:{features_combinations[0]}. name:{input_features_names[0]}')
                run_name = 'pipe_{0}_model_{1}_group{2}_oo{3}_ggi{4}_{5}'.format(Usecase_pipeline.value, str(Usecase_main_model.name + '_scoring_' + scoring_name), i, len(features_combinations), ggi, thread_name)


            if selector is not None:

                num_patients, n_f_before = X_train_filtered.shape
                col_names_before = X_train_filtered.columns

                if Usecase_feature_selection in (features_selection.Survival_Univariate_cox,
                        features_selection.SFS_CoxPH, features_selection.SFS_CoxNet, features_selection.SFS_RSF, features_selection.SFS_XGB_Survival):
                    selector.fit(X_train_filtered, y_train_filtered)

                else:
                    selector.fit(X_train_filtered, y_train_filtered[survival_name2])

                save_results(DT_token).features_selection(selector, Usecase_feature_selection , name=run_name, name2 = 'preporc', scoring_family = scoring_type)
                X_train_filtered, X_test_filtered = select_features(selector = selector, X_train = X_train_filtered, X_test = X_test_filtered)

            ############### init for main model part
            if main_model == None:
                print('Pre-proceesing only')
                continue

            if Usecase_pipeline == Usecase.Simple:
                gen_object = main_model  # CoxPHSurvivalAnalysis()

            elif Usecase_pipeline == Usecase.Features_Select:
                X = X_train_filtered
                y = y_train_filtered
                gen_object = GridSearchCV(main_model, main_model_param_grid,cv = 2,
                                        scoring=lambda estimator, X, y: my_ImBS(estimator, X, y,  times=prediction_vector_times, y_train=y_train_filtered),
                                        return_train_score=True, verbose=verbose_level, n_jobs=-1)

            elif Usecase_pipeline == Usecase.Intra_Batch:

                cv = KFold(n_splits=3, random_state=1, shuffle=True)
                gen_object = GridSearchCV(main_model, main_model_param_grid, return_train_score=True, cv=cv, error_score='raise',
                                        verbose=verbose_level, n_jobs=-1)

            elif Usecase_pipeline == Usecase.SFS_Inter_Batch:
                
                # Use custom KFold for ensuring train-test survival overlap
                inclusive_kfold = InclusiveSurvivalKFold(n_splits=5)
                
                X = X_train_filtered
                y = y_train_filtered
                if use_different_score_metrics:
                    gen_object = SFS(estimator=main_model, k_features='best',  # k_features= (model_min_num_f, model_max_num_f + 1) #
                        forward=True,
                        floating=False,
                        scoring=lambda estimator, X, y: my_ImBS(estimator, X, y, times=None, surv_name = survival_name2 ,y_train=y_train_filtered),
                        verbose=1,
                        n_jobs=-1,
                        cv=inclusive_kfold)  # change to 3 if using for feature selection 3 
                else:
                    gen_object = SFS(estimator=main_model, k_features='best',  # k_features= (model_min_num_f, model_max_num_f + 1) #
                        forward=True,
                        floating=False,
                        scoring= ci_score,
                        verbose=1,
                        n_jobs=-1,
                        cv=inclusive_kfold)  # change to 3 if using for feature selection 3 


            gen_object.fit(X_train_filtered, y_train_filtered)

   ######### post proc

            if Usecase_pipeline == Usecase.SFS_Inter_Batch:
                save_results(DT_token).features_selection(selector = gen_object, features_selection = Usecase_feature_selection, name=run_name, name2='main_model', use_different_score_metrics = use_different_score_metrics, scoring_family = scoring_type)
                X_train_filtered, X_test_filtered = select_features(selector=gen_object, X_train=X_train_filtered, X_test=X_test_filtered)
            elif Usecase_pipeline != Usecase.Simple:
                save_results(DT_token).models_cv(gen_object=gen_object, param_grid = param_grid, name=run_name)

            #### focus on best model

            if Usecase_pipeline != Usecase.Simple:
                run_name = run_name +'_best_model'
            else:
                run_name = run_name + '_single_model'

        
            if (Usecase_main_model != main_model_type.Cox) & (Usecase_main_model != main_model_type.Cox_Regulation_free):
                if not actual_survival_times:
                    score_results, score_results_vectors, params = best_model_preparations(feature_train=X_train_filtered, feature_test=X_test_filtered, y_train=y_train_filtered,y_test=y_test_filtered, gen_object=gen_object,
                                                                    Usecase_main_model = Usecase_main_model, name=run_name, use_different_score_metrics = use_different_score_metrics,
                                                                    prediction_vector_times = prediction_vector_times)
                    
                    scorer_results = save_results(DT_token).summary(score_results=score_results, score_results_vectors =score_results_vectors, params = params, input_features_names = input_features_names, y_train=y_train_filtered, y_test=y_test_filtered,
                            feature_test=X_test_filtered, feature_train=X_train_filtered,
                            model_trained=gen_object,
                            name=run_name, use_different_score_metrics = use_different_score_metrics, optimization = scoring_name,
                                                prediction_vector_times = prediction_vector_times )
                    if final_model:
                        save_results(DT_token).various_predictions_probabilities_per_patient(features=X_train_filtered, y=y_train_filtered, trained_model=gen_object, 
                                                                                             prediction_vector_times = prediction_vector_times , name=run_name + '_train')
                        
                        save_results(DT_token).various_predictions_probabilities_per_patient(features=X_test_filtered, y=y_test_filtered, trained_model=gen_object,
                                                                                             prediction_vector_times = prediction_vector_times , name=run_name + '_test')
                        
                        f_importance_tool.show(input_features_names = input_features_names, results = score_results, name=run_name, bshow_images=False, bsave_image = False, token = DT_token)
                else:
                    score_results, score_results_vectors, params = best_model_preparations(feature_train=X_train_filtered, feature_test=X_test_filtered, y_train=y_train_filtered,y_test=y_test_filtered, gen_object=gen_object,
                                                                    Usecase_main_model = Usecase_main_model, name=run_name, use_different_score_metrics = use_different_score_metrics,
                                                                    prediction_vector_times = prediction_vector_times_train, prediction_vector_times_test = prediction_vector_times_test)
                    
                    scorer_results = save_results(DT_token).summary(score_results=score_results, score_results_vectors =score_results_vectors, params = params, input_features_names = input_features_names, y_train=y_train_filtered, y_test=y_test_filtered,
                            feature_test=X_test_filtered, feature_train=X_train_filtered,
                            model_trained=gen_object,
                            name=run_name, use_different_score_metrics = use_different_score_metrics, optimization = scoring_name,
                                                prediction_vector_times = prediction_vector_times_train, prediction_vector_times_test=  prediction_vector_times_test)
                    
                    f_importance_tool.show(input_features_names = input_features_names, results = score_results, name=run_name, bshow_images=False, bsave_image = False, token = DT_token)
                    
                    if final_model:
                        save_results(DT_token).various_predictions_probabilities_per_patient(features=X_train_filtered, y=y_train_filtered, trained_model=gen_object, 
                                                                                            prediction_vector_times = prediction_vector_times_train , name=run_name + '_train')
                    
                        save_results(DT_token).various_predictions_probabilities_per_patient(features=X_test_filtered, y=y_test_filtered, trained_model=gen_object,
                                                                                            prediction_vector_times = prediction_vector_times_test , name=run_name + '_test')
                    
            else:
                if not actual_survival_times:
                    score_results, score_results_vectors, params ,p_values_stats= best_model_preparations(feature_train=X_train_filtered, feature_test=X_test_filtered, y_train=y_train_filtered,y_test=y_test_filtered, gen_object=gen_object,
                                                    Usecase_main_model = Usecase_main_model, name=run_name, use_different_score_metrics = use_different_score_metrics,
                                                    prediction_vector_times = prediction_vector_times )
                    
                    scorer_results = save_results(DT_token).summary(score_results=score_results, score_results_vectors =score_results_vectors, params = params,
                                                                    input_features_names = input_features_names, y_train=y_train_filtered, y_test=y_test_filtered,
                                                                    feature_test=X_test_filtered, feature_train=X_train_filtered,
                                                                    model_trained=gen_object,
                                                                    name=run_name, use_different_score_metrics = use_different_score_metrics, optimization = scoring_name,
                                                                    prediction_vector_times = prediction_vector_times )
                    
                    f_importance_tool.show(input_features_names = input_features_names, results = score_results, name=run_name, bshow_images=False, bsave_image = False, token = DT_token,  p_values_stats = p_values_stats)
                    
                    if final_model:
                        save_results(DT_token).various_predictions_probabilities_per_patient(features=X_train_filtered, y=y_train_filtered, trained_model=gen_object, 
                                                                                            prediction_vector_times = prediction_vector_times , name=run_name + '_train')
                    
                        save_results(DT_token).various_predictions_probabilities_per_patient(features=X_test_filtered, y=y_test_filtered, trained_model=gen_object,
                                                                                            prediction_vector_times = prediction_vector_times , name=run_name + '_test')
                    
                else:
                    score_results, score_results_vectors, params ,p_values_stats= best_model_preparations(feature_train=X_train_filtered, feature_test=X_test_filtered, y_train=y_train_filtered,y_test=y_test_filtered, gen_object=gen_object,
                                Usecase_main_model = Usecase_main_model, name=run_name, use_different_score_metrics = use_different_score_metrics,
                                prediction_vector_times = prediction_vector_times_train, prediction_vector_times_test = prediction_vector_times_test )
                    
                    # for cox alone to get HR and coefficients
                    save_results(DT_token).model( best_model_results = score_results,  name=run_name)



        print("---------------Pipe {0} - End---------------".format(Usecase_pipeline))


    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(type=type(ex), value=ex, tb=ex.__traceback__)))  # was etype=type(ex)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print(current_function_name)


if __name__ == "__main__":
    try:
    
        DT_token = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        warnings.filterwarnings("error", message="invalid value encountered in scalar divide")
        warnings.filterwarnings("error", message="Degrees of freedom <= 0 for slice")
        print('starting')
        original_stdout = sys.stdout
        
        try:
            model_choice = int(sys.argv[1])  # First argument after the script name
        except (IndexError, ValueError):
            print("Please provide a valid integer (0, 1, or 2) for model selection.")
            sys.exit(1)
        
        if model_choice == 0:
            Usecase_main_model = main_model_type.Bagging
        elif model_choice == 1:
            Usecase_main_model = main_model_type.Boosting
        elif model_choice == 2:
            Usecase_main_model = main_model_type.Cox_Regulation_free
        
        Usecase_pipeline = Usecase.SFS_Inter_Batch
        
        Usecase_feature_selection = features_selection.Without # model based feature selectiion not used in paper
        use_different_score_metrics = False  # non ci i.e. censored brier,always true!
        second_round_features_selection  = globals.second_round_features_selection.from_top_runs_file         #from_list_of_f_names_file        #from_f_stats_over_multiple_runs_file     #from_top_runs_file   # from_f_importance_files  #none
        multi_files_input = False  # used when i want to concatenate multi  files for the same users e.g.  isomiRs and mature

        num_f_least_corr_to_select =  1484 #num of f to run with

        # 15 for end results , 20 for whole pipeline
        f_ranges_len =  20  # of groups per thread

        # to test runs on subset of small groups
        first_group_f_start = 0
        last_group_f_start =  100   # now we have 1484 f
        
        # for multithread use - paper pipeline, - True
        b_use_multi_threading = True
        f_repetition_num =   50  # how many times each f repeat
        b_generate_features_combinations = True
        b_use_generated_features_combinations = True  #True   relevant also  when b_relapse_run = True
        allready_generated_features_combinations_token = '2024_11_15_11_51_51'   # put DT_token if b_generate_features_combinations = False
        b_relapse_run = False

        b_immdiate_buffering = True  # 1 slows output
        b_print_log_to_screen = True
        if os_name == 'Linux':
            #save_path = f'/outputs/pipelines_output/'
            save_path = f'/home/projects/hornsteinlab/yahelc/R/isoMIRs/isomiRs_II/results' # ImBS_XGBOOST_iteration/
            save_path_internal =  os.path.join(save_path,f'r_{DT_token}')
            globals.savefig_path_internal = globals.savefig_path_internal.format(DT_token)
            globals.folder_path_internal = globals.folder_path_internal.format(DT_token)
        else:
            save_path = f"D:\\yahel\\phd\isomiRs\\Ilan's\\IsomiRsII\\results"
            save_path_internal =  os.path.join(save_path,f'r_{DT_token}')
            globals.savefig_path_internal = globals.savefig_path_internal.format(DT_token)
            globals.folder_path_internal = globals.folder_path_internal.format(DT_token)
            
   
        script_name = os.path.basename(__file__)
        log_output_file_name2 = log_output_file_name.format(script_name, DT_token)
        path_n_name = os.path.join(save_path, log_output_file_name2)
        print(f'log file:{path_n_name}')

        with open(path_n_name, 'w', buffering=b_immdiate_buffering) as file:
            # Redirect stdout to both console and the output file
            if b_print_log_to_screen:
                sys.stdout = original_stdout
            else:
                sys.stdout = file


            if b_generate_features_combinations == True:
                allready_generated_features_combinations_token = DT_token
            if not (b_relapse_run) and not (os.path.exists(save_path_internal)):
                os.makedirs(save_path_internal)
                print(f"Folder created: {save_path_internal}")

            features_combinations = []
            participating_features = []
            
            # This could be precentage if globals.second_round_features_selection is from_f_stats_over_multiple_runs_file
            # this could also be a cutoff for score if globals.second_round_features_selection is from_top_runs_file
                # can also be None in this case than the elbow method is applied
            numerical_threshold =   None  #0.9 # 
            
            #select_single_feature_index =8
            sort_by_score = 'ci_train'
            if second_round_features_selection != globals.second_round_features_selection.none:
                
                participating_features_names= second_round_selection(second_round_features_selection, DT_token, first_round_DT_token, numerical_threshold, sort_by_score)
                participating_features = extract_features_indexes_from_names(participating_features_names)
                
                if multi_files_input:
                    train_mature, test_mature = mature_data_load()
                    addtional_input_f_names = train_mature.columns
                    addtional_input_f_psaudo_indeces = np.arange(1, len(addtional_input_f_names) + 1)
                    addtional_input_f_psaudo_indeces = list((num_f_least_corr_to_select -1) + addtional_input_f_psaudo_indeces)
                    participating_features = participating_features + addtional_input_f_psaudo_indeces

            if b_generate_features_combinations:
                #remove_features_combinations_files()
                if second_round_features_selection == globals.second_round_features_selection.none:
                    fixed_numbers_pos = copy.copy(list(np.arange(num_f_least_corr_to_select)))
                    generate_features_combinations_file(numbers=list(fixed_numbers_pos),fixed_numbers_pos = fixed_numbers_pos, group_size=f_ranges_len, token = DT_token, repetition=f_repetition_num)                                       
                    fill_random_numbers(numbers_range = list(np.arange(num_f_least_corr_to_select)), group_size = f_ranges_len, token = DT_token)
                else:
                    fixed_numbers_pos = copy.copy(tuple(participating_features))
                    generate_features_combinations_file(numbers=tuple(participating_features),fixed_numbers_pos = fixed_numbers_pos, group_size=f_ranges_len, token = DT_token, repetition=f_repetition_num)
                    fill_random_numbers(numbers_range = list(participating_features), group_size = f_ranges_len, token = DT_token)
                    
            if b_use_generated_features_combinations:
                            features_combinations = read_features_combinations_file(allready_generated_features_combinations_token)
                            
            if b_relapse_run:
                if b_generate_features_combinations:
                    print('can not run relapse. new features combinations generated')
                    raise ValueError("can not run relapse. new features combinations generated.")
                features_combinations = rearrange_rerun(past_token = allready_generated_features_combinations_token, new_token = DT_token)
                DT_token = allready_generated_features_combinations_token   # since i want to cont. the prev. run

            num_groups = len(features_combinations)
            if num_groups == 0:
                num_groups = num_f_least_corr_to_select

            num_threads = int(np.ceil(num_groups/last_group_f_start))
            num_groups_per_thread = int(np.ceil(num_groups / num_threads))

            workers_f_groups = associate_groups_into_unified_thread_run(dictionary = features_combinations, chunk_size = num_groups_per_thread)

            if b_use_multi_threading:
                print(f'groups:{num_groups}.# threads:calc-{num_threads},created-{len(workers_f_groups)}. #groups per thread:{num_groups_per_thread}')
                threads = []
                for worker_number, worker_f_groups in enumerate(workers_f_groups):
                    print(worker_number)
                    thread = threading.Thread(target=isomiRs_main_worker_thread,
                                            args=(Usecase_main_model, Usecase_pipeline, Usecase_feature_selection,
                                                    use_different_score_metrics, second_round_features_selection,
                                                    multi_files_input, participating_features,
                                                    workers_f_groups[worker_number], f'T{worker_number}', True, False))
                    thread.start()
                    threads.append(thread)

                # Wait for all threads to finish
                for thread in threads:
                    thread.join()
            else:
                isomiRs_main_worker_thread(Usecase_main_model = Usecase_main_model, 
                                            Usecase_pipeline = Usecase_pipeline, 
                                            Usecase_feature_selection = Usecase_feature_selection,
                                            use_different_score_metrics = use_different_score_metrics, 
                                            second_round_features_selection = second_round_features_selection,
                                            multi_files_input = multi_files_input,
                                            features = participating_features, 
                                            features_combinations = features_combinations,
                                            thread_name = 'SingleT')
                                    

            summerize_multi_runs_main(DT_token,use_different_score_metrics)
            
    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(type=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print(current_function_name)

    
    finally:
        # Restore the original stdout to avoid issues with other parts of your code
        sys.stdout = original_stdout