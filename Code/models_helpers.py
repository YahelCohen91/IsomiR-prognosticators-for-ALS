import inspect
import sys,os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import traceback
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from contextlib import redirect_stdout

from lifelines import CoxPHFitter   #  fellow of sksurv CoxPHSurvivalAnalysis

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from globals import Usecase_main_model as main_model
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_multilabel_classification
from sklearn.feature_selection import SelectKBest , f_classif

from globals import summary_tab_name
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
    brier_score,
)

from sksurv.metrics import check_y_survival, _check_estimate_2d
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import globals as globals

survival_name2 = 'Survival from enrolment'
death_name = 'outcome'

def reduce_float_resolution(data, X):
    # Create an array of intervals by dividing the float values by X, and taking the floor
    # Then multiply back by X to get the rounded value to the nearest X
    data2 =data.copy()
    data2[:] = np.floor(data[:] / X) * X
    return data2




def calc_p_values(X_train,y_train, X_test,y_test):

    '''
    mode of operation
    coefficients = coxmodel.coef_
    standard_errors = np.sqrt(np.diag(coxmodel.variance_))
    # Calculate z-scores
    z_scores = coefficients / standard_errors
    # Calculate p-values (two-tailed)
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
    '''
    ##################
    y_train_df = pd.DataFrame(y_train)
    y_train_df.index = X_train.index
    unified_train = pd.concat([X_train, y_train_df], axis=1)


    estimator = CoxPHFitter()
    estimator.fit(unified_train, duration_col='Survival from enrolment', event_col='outcome')
    
    '''
    estimator_null = CoxPHFitter()
    estimator_null.fit(unified_train, duration_col='Survival from enrolment', event_col='outcome', formula='1')  # '1' sets all coefficients to zero except the intercept
    # estimator.print_summary()
    # estimator.fit(rossi, duration_col = 'week', event_col = 'arrest', formula = "fin")# + wexp + age * prio")
    estimator_null.print_summary()

    #########################
    # Obtain log-likelihoods of the full and reduced models
    ll_full = estimator._log_likelihood
    ll_reduced = estimator_null._log_likelihood

    # Calculate the likelihood ratio test statistic
    lr_test_statistic = 2 * (ll_full - ll_reduced)

    # Calculate the degrees of freedom for the chi-square distribution
    df_chi_square = estimator.params_.shape[0] - estimator_null.params_.shape[0]

    # Calculate the p-value using the chi-square distribution
    from scipy.stats import chi2
    overall_model_p_value = 1 - chi2.cdf(lr_test_statistic, df_chi_square)

    print("overall model: Likelihood ratio test statistic:", lr_test_statistic)
    print("overall model: p-value:", overall_model_p_value)
    ###################
    '''
    '''
    lr_test =  estimator.statistics_['ll_null'] - estimator.statistics_['llr']
    p_value = estimator.statistics_['p-value']
    print("option1: Likelihood ratio test statistic:", lr_test)
    print("option1: overall p-value:", p_value)
    lr_test2 = estimator.summary['lrx2']
    p_value2 = estimator.summary['p']
    print("option2: Likelihood ratio test statistic:", lr_test2)
    print("option2: overall p-value:", p_value2)
    '''
    
    summary_df = pd.DataFrame(estimator.summary)

    
    summary_df = summary_df['p']
    

    return summary_df

def coxph_fit_and_score_features(X, y):  # assume marginal contri - since linear it is ok
    try:
       
        n_features = X.shape[1]
        scores = np.empty(n_features)
        m = CoxPHSurvivalAnalysis()
        for j in range(n_features):
            Xj = X[:, j : j + 1]
            m.fit(Xj, y)
            scores[j] = m.score(Xj, y)
        return scores

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad coxph_fit_and_score_features')




def best_model_preparations(feature_train, feature_test, y_train,y_test, gen_object, Usecase_main_model, name, use_different_score_metrics
                            ,prediction_vector_times, prediction_vector_times_test = None):
    try:

        feature_names = feature_train.columns
        p_values_stats = None

        if isinstance(gen_object, SFS):
            gen_object = gen_object.estimator  # strip out the sfs wrapper
            gen_object.fit(feature_train, y_train)

        if isinstance(gen_object, GridSearchCV):
            model_object = gen_object.best_estimator_
            model_object.set_params(**gen_object.best_params_)
            model_object.fit(feature_train, y_train)  

            selected_feature_names = feature_names

            feature_train_selected = feature_train[selected_feature_names]
            feature_test_selected = feature_test[selected_feature_names]

            if Usecase_main_model == main_model.Bagging:  # namely RandomSurvivalBagging - dont have features importance - so we permute to immitate

                results = permutation_importance(model_object, feature_test, y_test, n_repeats=10, random_state = 0)
                coefs = results.importances_mean   

            elif hasattr(model_object, "coef_"):
                    coefs = model_object.coef_
            elif hasattr(model_object, "feature_importances_"):
                    coefs = model_object.feature_importances_

            hr = np.round(np.exp(coefs), 3)
            results = pd.DataFrame({
                'features':selected_feature_names,
                'coeff': coefs,
                'hazard_ratio': hr,
                'train_ci_score': model_object.score(feature_train_selected.values, y_train),
                'test_ci_score': model_object.score(feature_test_selected.values, y_test)
            })

            params = gen_object.get_params()
            return results, params

        if isinstance(gen_object, RandomSurvivalForest):  # namely RandomSurvivalBagging - dont have features importance - so we permute to immitate

            results = permutation_importance(gen_object, feature_test, y_test, n_repeats=10, random_state = 0)
            coeff = results.importances_mean  

            feature_names = gen_object.feature_names_in_


            hr = np.round(np.exp(coeff), 3)

            if use_different_score_metrics:
                if prediction_vector_times_test is None:
                    different_loss_metrics = calculate_survival_scores(model=gen_object,
                                                                       prediction_vector_times=prediction_vector_times,
                                                                       X_train=feature_train,
                                                                       y_train=y_train, X_test=feature_test,
                                                                       y_test=y_test)

                    mean_score_test = different_loss_metrics['mean_score_test']
                    mean_score_train = different_loss_metrics['mean_score_train']
                    type = different_loss_metrics['method']

                    results2 = different_loss_metrics
                    
                else:
                    different_loss_metrics = calculate_survival_scores(model=gen_object,
                                                    prediction_vector_times=prediction_vector_times,
                                                    X_train=feature_train,
                                                    y_train=y_train, X_test=feature_test,
                                                    y_test=y_test,
                                                    prediction_vector_times_test = prediction_vector_times_test)

                    mean_score_test = different_loss_metrics['mean_score_test']
                    mean_score_train = different_loss_metrics['mean_score_train']
                    type = different_loss_metrics['method']

                    results2 = different_loss_metrics
            else:
                mean_score_test = gen_object.score(feature_test, y_test)
                mean_score_train = gen_object.score(feature_train, y_train)
                type = 'ci_index'
                results2 = None

            results = pd.DataFrame({
                'name': name,
                'coeff': coeff,
                'hazard_ratio': hr,
                'features': feature_names,
                'mean_score_test': mean_score_test,
                'mean_score_train': mean_score_train,
                'score_type': type
            })

            params = gen_object.get_params()

            return results, results2, params

        if isinstance(gen_object, Pipeline):  # here you have only the best model - not exist any more
            if 'select' in pipe.named_steps:
                feature_names = gen_object.named_steps['select'].feature_names_in_
                selected_feature_names = feature_names[gen_object.named_steps['select'].get_support(indices=True)]
            else:
                selected_feature_names = feature_names

            model_object = pipe['model']
            if isinstance(model_object, CoxPHSurvivalAnalysis):
                coeff = model_object.coef_
            elif isinstance(model_object, GradientBoostingSurvivalAnalysis):
                coeff = model_object.feature_importances_
            elif (model_object, RandomSurvivalForest):
                coeff = 0

            hr = np.round(np.exp(coeff),3)
            ci_index_test = gen_object.score(feature_test, y_test)
            ci_index_train = gen_object.score(feature_train, y_train)
            results = pd.DataFrame({
                'name': name,
                'coeff': coeff,
                'hazard_ratio': hr,
                'features': selected_feature_names,
                'test_ci_score': ci_index_test,
                'train_ci_score': ci_index_train
            })

            params = gen_object['model'].get_params()

        if isinstance(gen_object, CoxnetSurvivalAnalysis)  or isinstance(gen_object, CoxPHSurvivalAnalysis):  # here you have only the best model
            feature_names = gen_object.feature_names_in_
            coeff = gen_object.coef_
            hr = np.round(np.exp(coeff), 3)
            p_values_stats = calc_p_values(X_train=feature_train,y_train=y_train, X_test=feature_test,y_test=y_test)

            if use_different_score_metrics:
                if prediction_vector_times_test is None:
                    
                    different_loss_metrics = calculate_survival_scores(model=gen_object,
                                                                       prediction_vector_times=prediction_vector_times,
                                                                       X_train=feature_train,
                                                                       y_train=y_train, X_test=feature_test,
                                                                       y_test=y_test)

                    mean_score_test = different_loss_metrics['mean_score_test']
                    mean_score_train = different_loss_metrics['mean_score_train']
                    type = different_loss_metrics['method']

                    results2 = different_loss_metrics
                    
                else:
                    different_loss_metrics = calculate_survival_scores(model=gen_object,
                                                    prediction_vector_times=prediction_vector_times,
                                                    X_train=feature_train,
                                                    y_train=y_train, X_test=feature_test,
                                                    y_test=y_test,
                                                    prediction_vector_times_test = prediction_vector_times_test)

                    mean_score_test = different_loss_metrics['mean_score_test']
                    mean_score_train = different_loss_metrics['mean_score_train']
                    type = different_loss_metrics['method']

                    results2 = different_loss_metrics
            else:
                mean_score_test = gen_object.score(feature_test, y_test)
                mean_score_train = gen_object.score(feature_train, y_train)
                type = 'ci_index'
                results2 = None

            if (isinstance(gen_object, CoxnetSurvivalAnalysis) and gen_object.n_alphas == 1) or (isinstance(gen_object, CoxPHSurvivalAnalysis) and gen_object.alpha == 0):

                results = pd.DataFrame({
                    'name': name,
                    'coeff': coeff.ravel(),
                    'hazard_ratio': hr.ravel(),
                    'features': feature_names,
                    'mean_score_test': mean_score_test,
                    'mean_score_train': mean_score_train,
                    'score_type': type
                })

            else:

                results = pd.DataFrame({
                    'name': name,
                    'features': feature_names,
                    'mean_score_test': mean_score_test,
                    'mean_score_train': mean_score_train,
                    'score_type': type

                })

                for i, alpha in enumerate(gen_object.alphas_):
                    results[f'coeff of alphas: {alpha}']  = np.round(coeff[:,i],6)
                    results[f'hazard_ratio of alphas: {alpha}'] = np.round(hr[:,i],6)

            params = gen_object.get_params()

            return results, results2, params, p_values_stats

        if isinstance(gen_object, GradientBoostingSurvivalAnalysis):  # here you have only the best model
            feature_names = gen_object.feature_names_in_
            feature_importances = gen_object.feature_importances_
            coeff = feature_importances
            hr = np.round(np.exp(coeff), 3)
            if use_different_score_metrics:
                if prediction_vector_times_test is None:
                    different_loss_metrics = calculate_survival_scores(model=gen_object,
                                                                       prediction_vector_times=prediction_vector_times,
                                                                       X_train=feature_train,
                                                                       y_train=y_train, X_test=feature_test,
                                                                       y_test=y_test)

                    mean_score_test = different_loss_metrics['mean_score_test']
                    mean_score_train = different_loss_metrics['mean_score_train']
                    type = different_loss_metrics['method']

                    results2 = different_loss_metrics
                    
                else:
                    different_loss_metrics = calculate_survival_scores(model=gen_object,
                                                    prediction_vector_times=prediction_vector_times,
                                                    X_train=feature_train,
                                                    y_train=y_train, X_test=feature_test,
                                                    y_test=y_test,
                                                    prediction_vector_times_test = prediction_vector_times_test)

                    mean_score_test = different_loss_metrics['mean_score_test']
                    mean_score_train = different_loss_metrics['mean_score_train']
                    type = different_loss_metrics['method']

                    results2 = different_loss_metrics
            else:  
                mean_score_test = gen_object.score(feature_test, y_test)
                mean_score_train = gen_object.score(feature_train, y_train)
                type = 'ci_index'
                results2 = None
            
            results = pd.DataFrame({
                'name': name,
                'coeff': coeff,
                'hazard_ratio': hr,
                'features': feature_names,
                'mean_score_test': mean_score_test,
                'mean_score_train': mean_score_train,
                'score_type': type
            })

            params = gen_object.get_params()

            return results, results2, params
            
    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad best_model_preparations')



def calculate_survival_scores(model, prediction_vector_times, X_train, y_train, X_test, y_test, prediction_vector_times_test = None):
    try:
        """
        Calculate various survival analysis scores for train and test sets.
    
        Parameters:
        - model: Fitted survival model from scikit-survival
        - X_train: Features of the training set
        - y_train: Survival data of the training set
        - X_test: Features of the testing set
        - y_test: Survival data of the testing set
        
        prediction_vector_times
    
        Returns:
        - Dictionary containing different metrics for train and test sets
        """


        if prediction_vector_times_test is None:
            y_times = prediction_vector_times

            # Predict survival function for training set

            survival_train = np.row_stack([fn(y_times) for fn in model.predict_survival_function(X_train)])
            # Predict survival function for testing set
            survival_test = np.row_stack([fn(y_times) for fn in model.predict_survival_function(X_test)])


            # Brier Score
            brier_train_time, brier_train_score = brier_score(survival_train = y_train, survival_test = y_train, estimate =survival_train, times = y_times)
            brier_test_time, brier_test_score = brier_score(survival_train = y_train, survival_test = y_test, estimate =survival_test, times = y_times)
            assert brier_train_time.any() == y_times.any()
            assert brier_test_time.any() == y_times.any()

            #Integrated Median Brier score ImBS
            median_brier_train_time, median_brier_train_score = median_brier_score(survival_train=y_train, survival_test=y_train,estimate=survival_train, times=y_times)
            median_brier_test_time, median_brier_test_score = median_brier_score(survival_train=y_train, survival_test=y_test,estimate=survival_test, times=y_times)
            assert median_brier_train_time.any() == y_times.any()
            assert median_brier_test_time.any() == y_times.any()
            ImBS_train = np.trapz(median_brier_train_score, median_brier_train_time) / (median_brier_train_time[-1] - median_brier_train_time[0])
            ImBS_test = np.trapz(median_brier_test_score, median_brier_test_time) / (median_brier_test_time[-1] - median_brier_test_time[0])


            # Integrated Brier Score

            integrated_brier_train = integrated_brier_score(survival_train = y_train, survival_test = y_train, estimate = survival_train, times = y_times)
            integrated_brier_test = integrated_brier_score(survival_train=y_train, survival_test=y_test, estimate=survival_test,  times = y_times)


            interval_integrated_brier_train = {}
            interval_integrated_brier_test = {}
            index_4_survival_month_start = [12, 9, 15,17,21,27]  # if start by 3 and jump by 1. index 0 is 3m survival
            for  i in index_4_survival_month_start:
                interval_times = y_times[i:-1]
                inverval_name = 'brier_m:{0}-{1}'.format(y_times[i],y_times[-1])
                interval_integrated_brier_test[inverval_name] =  np.trapz(brier_test_score[i:-1], brier_test_time[i:-1]) / (brier_test_time[-1] - brier_test_time[i])
                interval_integrated_brier_train[inverval_name] = np.trapz(brier_train_score[i:-1], brier_train_time[i:-1]) / (  brier_train_time[-1] - brier_train_time[i])
                
        else:
            y_times_train = prediction_vector_times
            y_times_test = prediction_vector_times_test
            # Predict survival function for training set
            survival_train = np.row_stack([fn(y_times_train) for fn in model.predict_survival_function(X_train)])
            # Predict survival function for testing set
            survival_test = np.row_stack([fn(y_times_test) for fn in model.predict_survival_function(X_test)])
            
            # Brier Score
            brier_train_time, brier_train_score = brier_score(survival_train = y_train, survival_test = y_train, estimate =survival_train, times = y_times_train)
            brier_test_time, brier_test_score = brier_score(survival_train = y_train, survival_test = y_test, estimate =survival_test, times = y_times_test)
            assert brier_train_time.any() == y_times_train.any()
            assert brier_test_time.any() == y_times_test.any()

            #Integrated Median Brier score ImBS
            median_brier_train_time, median_brier_train_score = median_brier_score(survival_train=y_train, survival_test=y_train,estimate=survival_train, times=y_times_train)
            median_brier_test_time, median_brier_test_score = median_brier_score(survival_train=y_train, survival_test=y_test,estimate=survival_test, times=y_times_test)
            assert median_brier_train_time.any() == y_times_train.any()
            assert median_brier_test_time.any() == y_times_test.any()
            
            ImBS_train = np.trapz(median_brier_train_score, median_brier_train_time) / (median_brier_train_time[-1] - median_brier_train_time[0])
            ImBS_test = np.trapz(median_brier_test_score, median_brier_test_time) / (median_brier_test_time[-1] - median_brier_test_time[0])

            # Integrated Brier Score
            integrated_brier_train = integrated_brier_score(survival_train = y_train, survival_test = y_train, estimate = survival_train, times = y_times_train)
            integrated_brier_test = integrated_brier_score(survival_train=y_train, survival_test=y_test, estimate=survival_test,  times = y_times_test)
            
            interval_integrated_brier_train = {}
            interval_integrated_brier_test = {}
            index_4_survival_month_start = [12, 9, 15,17,21,27]  # if start by 3 and jump by 1. index 0 is 3m survival .ugly
            for  i in index_4_survival_month_start:
                interval_times_train = y_times_train[i:-1]
                interval_times_test = y_times_test[i:-1]
                inverval_name_train = 'brier_m:{0}-{1}'.format(y_times_train[i],y_times_train[-1])
                inverval_name_test = 'brier_m:{0}-{1}'.format(y_times_test[i],y_times_test[-1])
                interval_integrated_brier_test[inverval_name_test] =  np.trapz(brier_test_score[i:-1], brier_test_time[i:-1]) / (brier_test_time[-1] - brier_test_time[i])
                interval_integrated_brier_train[inverval_name_train] = np.trapz(brier_train_score[i:-1], brier_train_time[i:-1]) / (  brier_train_time[-1] - brier_train_time[i])


        # Concordance Index for Censored Data
        if 'time' in y_train.dtype.names:
            concordance_train = concordance_index_censored(y_train['cens'], y_train['time'], model.predict(X_train))
            concordance_test = concordance_index_censored(y_test['cens'], y_test['time'], model.predict(X_test))
        elif 'Survival from enrolment' in y_train.dtype.names:
            concordance_train = concordance_index_censored(y_train['outcome'], y_train['Survival from enrolment'], model.predict(X_train))
            concordance_test = concordance_index_censored(y_test['outcome'], y_test['Survival from enrolment'], model.predict(X_test))
        else:
            print('error')
            raise  KeyError(f"y Columns does not exist in the structured array.")

        if False:

            print( f"CI train {concordance_index_censored(y_train['outcome'], y_train['Survival from enrolment'], model.predict(X_train))}")
            print( f"CI test {concordance_index_censored(y_test['outcome'], y_test['Survival from enrolment'], model.predict(X_test))}")
            requested_binning_resuolation = 3
            y_test_new = y_test.copy()
            new_y_test = reduce_float_resolution(y_test['Survival from enrolment'], requested_binning_resuolation)
            y_test_new['Survival from enrolment'] = new_y_test
            print(f"binned CI test {concordance_index_censored(y_test_new['outcome'], y_test_new['Survival from enrolment'],    model.predict(X_test))}")


        # Integrated Concordance Index for Censored Data

        # Create a dictionary to store the results
        scores_dict = {
            'Concordance Index (Train)': concordance_train[0],
            'Concordance Index (Test)': concordance_test[0],


            'brier_train_score': brier_train_score,
            #'brier_train_time': brier_train_time,
            'brier_test_score': brier_test_score,
            #'brier_test_time': brier_test_time,
            'mean_score_train': integrated_brier_train,
            'mean_score_test': integrated_brier_test,
            'median_brier_train_score': median_brier_train_score,
            'norm_median_brier_train_score': (1.0-np.sqrt(median_brier_train_score)),
            'median_brier_train_time': median_brier_train_time,
            'median_brier_test_score': median_brier_test_score,
            'norm_median_brier_test_score': (1.0 - np.sqrt(median_brier_test_score)),
            'median_brier_test_time': median_brier_test_time,
            'ImBS_train': ImBS_train,
            'ImBS_test': ImBS_test,
            'method':summary_tab_name,
        }

        for key, value in interval_integrated_brier_train.items():
            scores_dict[key+'_train'] = value
        for key, value in interval_integrated_brier_test.items():
            scores_dict[key+'_test'] = value


        return scores_dict

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print(current_function_name)

def my_IBS(estimator, X, y, times, y_train):
    try:
        linear_complement_IBS_score = -1.1  # to answer for higher is better
        try:
            N, l = X.shape
        except ValueError:
            l = 1
    # Predict survival function for
        survival = np.row_stack([fn(times) for fn in estimator.predict_survival_function(X)])
    # Brier Score
        IBS_score = integrated_brier_score(survival_train=y_train, survival_test=y, estimate=survival, times=times)
        linear_complement_IBS_score = 1.0 - np.sqrt(IBS_score)  #to answer for higher is better I linerazie and inverse
        print(f'current: {IBS_score:.{5}f}.linear_complement_IBS_score:{linear_complement_IBS_score:.{5}f}.#f:{l}')
        print(f'current: {IBS_score:.{5}f}. Best: {my_IBS.best_ibs:.{3}f}')
        my_IBS.best_ibs = min(IBS_score, my_IBS.best_ibs)
    except AttributeError:
        my_IBS.best_ibs = 1.1 #IBS_score
        print('AttributeError')
    except Exception as ex:
         print(ex.__context__)
         print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
         exc_type, exc_obj, exc_tb = sys.exc_info()
         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
         print(exc_type, fname, exc_tb.tb_lineno)
         current_function_name = inspect.currentframe().f_back.f_code.co_name
         print(current_function_name)
    finally:
        return linear_complement_IBS_score  # returning along "higher is better" guidelines


# define a cross-validtion split to ensure training folds survival range always include testing fold survival range
class InclusiveSurvivalKFold:
    def __init__(self, n_splits, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups=None):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Extract survival times from y (assuming y is a structured array)
        survival_times = np.array([time for event, time in y])
        
        # Sort data by survival times
        sorted_indices = np.argsort(survival_times)
        if isinstance(X, pd.DataFrame):  # If X is a DataFrame, use iloc
            X_sorted = X.iloc[sorted_indices]
            y_sorted = y[sorted_indices]
        else:  # For other array-like data structures
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]
        survival_times_sorted = survival_times[sorted_indices]
        
        for train_index, test_index in kf.split(X_sorted):
            # Check the range of survival times in the test set
            train_times = survival_times_sorted[train_index]
            test_times = survival_times_sorted[test_index]
            
            # Adjust training set if necessary
            if train_times.min() > test_times.min() or train_times.max() < test_times.max():
                # Find the missing survival times in the training set and add them
                train_index, test_index = self.adjust_train_set(train_index, test_index, survival_times_sorted)
            
            yield list(train_index), list(test_index)
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def adjust_train_set(self, train_index, test_index, survival_times_sorted):
        # Get min and max survival times of the test set
        min_test_time, max_test_time = survival_times_sorted[test_index].min(), survival_times_sorted[test_index].max()
        
        # Get min and max survival times of the train set
        min_train_time, max_train_time = survival_times_sorted[train_index].min(), survival_times_sorted[train_index].max()
        
        # Lists to store indices to be swapped between train and test sets
        test_remove_indices = []
        train_remove_indices = []
        train_add_indices = []
        test_add_indices = []

        # If test set min is smaller than train set min, move the min from test to train
        if min_test_time < min_train_time:
            test_min_index = np.where(survival_times_sorted[test_index] == min_test_time)[0][0]
            train_min_index = np.where(survival_times_sorted[train_index] == min_train_time)[0][0]
            
            # Swap indices between test and train sets
            test_remove_indices.append(test_index[test_min_index])
            train_remove_indices.append(train_index[train_min_index])
            
            test_add_indices.append(train_index[train_min_index])
            train_add_indices.append(test_index[test_min_index])

        # If test set max is greater than train set max, move the max from test to train
        if max_test_time > max_train_time:
            test_max_index = np.where(survival_times_sorted[test_index] == max_test_time)[0][0]
            train_max_index = np.where(survival_times_sorted[train_index] == max_train_time)[0][0]
            
            # Swap indices between test and train sets
            test_remove_indices.append(test_index[test_max_index])
            train_remove_indices.append(train_index[train_max_index])
            
            test_add_indices.append(train_index[train_max_index])
            train_add_indices.append(test_index[test_max_index])

        # Remove the indices from test set and train set
        new_test_index = np.setdiff1d(test_index, test_remove_indices)
        new_train_index = np.setdiff1d(train_index, train_remove_indices)

        # Add the new indices to the respective sets
        new_train_index = np.unique(np.concatenate([new_train_index, train_add_indices]))
        new_test_index = np.unique(np.concatenate([new_test_index, test_add_indices]))

        return new_train_index, new_test_index



def  my_IBS_inverse(linear_complement_IBS_score):
    try:
        IBS = np.power(1-linear_complement_IBS_score,2)

        return IBS

    except Exception as ex:
         print(ex.__context__)
         print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
         exc_type, exc_obj, exc_tb = sys.exc_info()
         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
         print(exc_type, fname, exc_tb.tb_lineno)
         current_function_name = inspect.currentframe().f_back.f_code.co_name
         print(current_function_name)


def my_ImBS(estimator, X, y, y_train, times=None, surv_name = None):
    try:
        linear_complement_ImBS_score = -1.1  # to answer for higher is better
        try:
            N, l = X.shape
        except ValueError:
            l = 1
    
    # for CV purposes make sure the times tested for prediction are just the ones in test
        if times is None:
            surv_times = y[surv_name]
            times = np.unique(surv_times)
            times = np.sort(times)
            # removing max time as a workaround sk-surv code that won't allow max time of survival to be equal/greater than evaluating times
            times = np.delete(times, -1)

            
    # Predict survival function for each patient
        survival = np.row_stack([fn(times) for fn in estimator.predict_survival_function(X)])
    # Brier Score

        # Integrated Median Brier score ImBS
        median_brier_time, median_brier = median_brier_score(survival_train=y_train, survival_test=y, estimate=survival,times=times)
        ImBS_score = np.trapz(median_brier, median_brier_time) / ( median_brier_time[-1] - median_brier_time[0])

        linear_complement_ImBS_score = 1.0 - np.sqrt(ImBS_score)  #to answer for higher is better I linerazie and inverse
        print(f'current: {ImBS_score:.{5}f}.linear_complement_ImBS_score:{linear_complement_ImBS_score:.{5}f}.#f:{l}')

    except AttributeError:
        my_ImBS.best_ibs = 1.1
        print('AttributeError')
    except Exception as ex:
         print(ex.__context__)
         print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
         exc_type, exc_obj, exc_tb = sys.exc_info()
         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
         print(exc_type, fname, exc_tb.tb_lineno)
         current_function_name = inspect.currentframe().f_back.f_code.co_name
         print(current_function_name)
    finally:
        return linear_complement_ImBS_score  # returning along "higher is better" guidelines


def select_features(selector, X_train, X_test ):
    
    try:
        num_patients, n_f_before = X_train.shape
        col_names_before = X_train.columns

        X_train_filtered = pd.DataFrame(selector.transform(X_train))
        X_test_filtered = pd.DataFrame(selector.transform(X_test))
        if isinstance(selector, SFS):
            X_train_filtered.columns = selector.k_feature_names_
            X_test_filtered.columns = selector.k_feature_names_
        if isinstance(selector, SelectFromModel) or isinstance(selector, SelectKBest):
            X_train_filtered.columns = col_names_before[selector.get_support()]
            X_test_filtered.columns = col_names_before[selector.get_support()]

        num_patients, n_f_new = X_train_filtered.shape
        print('\n SFS f selection: was {0}. selected: {1} features'.format(n_f_before, n_f_new))
        # now we can use "regular" model/pipeline
        return X_train_filtered, X_test_filtered

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        print(current_function_name)


def  median_brier_score(survival_train, survival_test, estimate, times):
    """Estimate the time-dependent Brier score for right censored data - with median instead of average. based on standard brier score  """
    try:
        test_event, test_time = check_y_survival(survival_test)
        estimate, times = _check_estimate_2d(estimate, test_time, times, estimator="brier_score")
        if estimate.ndim == 1 and times.shape[0] == 1:
            estimate = estimate.reshape(-1, 1)

        # fit IPCW estimator
        cens = CensoringDistributionEstimator().fit(survival_train)
        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t[prob_cens_t == 0] = np.inf
        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(test_time)
        prob_cens_y[prob_cens_y == 0] = np.inf

        # Calculating the brier scores at each time point
        median_brier_scores = np.empty(times.shape[0], dtype=float)
        for i, t in enumerate(times):
            est = estimate[:, i]
            is_case = (test_time <= t) & test_event
            is_control = test_time > t

            median_brier_scores[i] = np.median(  # this is the only change - was np.mean
                np.square(est) * is_case.astype(int) / prob_cens_y
                + np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]
            )

        return times, median_brier_scores
    
    
    except Exception as ex:
         print(ex.__context__)
         print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
         exc_type, exc_obj, exc_tb = sys.exc_info()
         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
         print(exc_type, fname, exc_tb.tb_lineno)
         current_function_name = inspect.currentframe().f_back.f_code.co_name
         print(current_function_name)
         
def ci_score(estimator, X, y,event_name= 'outcome', time_name= 'Survival from enrolment'):
        survival_pred = estimator.predict(X)
        return concordance_index_censored(y[event_name], y[time_name], survival_pred)[0]