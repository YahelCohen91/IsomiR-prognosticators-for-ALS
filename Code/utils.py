import debugpy
import copy
#debugpy.listen(('0.0.0.0', 5678))
import sys,os
import platform
import numpy as np
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
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

import seaborn as sns

from brokenaxes import brokenaxes

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, cumulative_dynamic_auc
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

import lifelines
from lifelines.statistics import logrank_test

from sklearn.metrics import make_scorer
from sklearn.feature_selection import f_regression, r_regression, chi2, f_classif ,SelectFromModel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import optuna 
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import RandomSampler

from scipy.stats import pearsonr, spearmanr
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.spatial.distance import euclidean

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

import lime
import lime.lime_tabular

import shap
from functools import partial
#from survshap import SurvivalModelExplainer, PredictSurvSHAP, ModelSurvSHAP


from data_loaders import data_load
from isomiRsII_main import Get_sorted_features_range, Load_sorted_features_range,normalize, CoxPH_single
from save_model_results import save_results

from data_view_lib import auc_time_dependant_tool
from data_view_lib import f_importance_tool


from models_helpers import coxph_fit_and_score_features, best_model_preparations
from models_helpers import calculate_survival_scores, select_features, my_IBS, my_ImBS,  ci_score, InclusiveSurvivalKFold

from data_loaders import extract_dict_items_generator
from data_loaders import second_round_selection, extract_features_indexes_from_names

from features_combinations import read_features_combinations_file
from features_combinations import rearrange_rerun, associate_groups_into_unified_thread_run

from Extract_mature_data import combine_multiple_inputs, mature_data_load
from Extract_mature_data import mature_data_load

#from  globals import savefig_path
import globals as globals
from globals import os_name, summary_tab_name, summary_tab_name_ci
from globals import isomiRsII_data_train, isomiRsII_data_test, mature_train, mature_test
from globals import Usecase
from globals import Usecase_main_model as main_model_type
from globals import Usecase_features_selection as features_selection
from globals import first_round_DT_token as first_round_DT_token
#from globals import savefig_path, savefig_path_internal, folder_path, folder_path_internal

from stats_on_cohort import cohort_summary_features_stats, cohort_summary_records_stats, cohort_summary_features_stats_deaths_seperated
from stats_on_cohort import survival_grouping_on_events

from save_model_results import save_results_file_name, save_events_file_name
from summerize_multi_runs_main import summerize_multi_runs_main, multiple_runs_stats_file_name


random_seed = 0
survival_name = 'Survival from onset'
survival_name2 = 'Survival from enrolment'
death_name = 'outcome'

test_2_train_col_name_change = {'Survival.from.onset':'Survival from onset', 'Survival.from.enrolment':'Survival from enrolment', 'Sex_c':'sex', 'CollAge':'Age of enrolment', 'deltaFRS':'ALSFRS slope', 'age_onset':'Age of onset'}

actual_survival_times = False
prediction_vector_times_train = []
tuned_model = []



# define objective for optuna optimization for Random forest
def print_best_callback(study, trial):
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

# define objective for optuna optimization for Random forest
def objective_xgb(trial, X,y, cv,eval_metric = 'ImBS',times = None ,surv_name='Survival from enrolment'):
    """
    trial - 
    X - pandas df of the feautures to predict survival
    y - survival boject that include tuples of (survival time, censorship) same in sksurv package
    folds - a list of fold train/test indexes outputed by sklearn Kfold object or similar
    eval_metric - either "ImBS"/"CI"/"AUC" to determine the score to optimize by (defualts to "ImBS")
    times = times corresponding to y to check AUC in each and average 
    surv_name= for ImBS function, need the name of the survival times (defualts to "Survival from enrolment" for convenience)
    """
    
    # Suggest parameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 18, 25),
        "learning_rate" : trial.suggest_float("learning_rate",0.24,0.27),
        "max_depth": trial.suggest_int("max_depth", 20, 25),
        "subsample" : trial.suggest_float("subsample", 0.8, 0.83),
        "min_impurity_decrease" : trial.suggest_float("min_impurity_decrease", 0, 0.01),
        'max_features' :trial.suggest_float("max_features", 0.04, 0.06),
        'min_weight_fraction_leaf' :trial.suggest_float("min_weight_fraction_leaf", 0.04 ,0.06),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 8),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 5000),
        'dropout_rate' :trial.suggest_float("dropout_rate", 0.35 ,0.5),
        'ccp_alpha' :trial.suggest_float("ccp_alpha", 0.0 ,0.02),
        #'loss' : trial.sugget_categorical('loss',['coxph', 'squared', 'ipcwls'])
    }
    
    # List to store the scores for each fold
    scores = []
    folds = list(cv.split(X, y))
    # Iterate over each fold
    for train_idx, test_idx in folds:

        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Initialize the RandomForestClassifier with suggested parameters
        
        clf = GradientBoostingSurvivalAnalysis(**params, random_state=42, n_iter_no_change=None,
                                               criterion='friedman_mse', verbose=1, loss = 'coxph')
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        
        # Calculate the accuracy (or any other metric)
        if eval_metric == "ImBS":
            score = my_ImBS(clf, X_test, y_test, y_train, surv_name=surv_name)
        elif eval_metric == "CI":
             score, *_ = concordance_index_ipcw(y_train, y_test, clf.predict(X_test))
        elif eval_metric == "AUC":
            if times is None:
                times = np.linspace(y_test[surv_name].min(), y_test[surv_name].max(),50)
            aucs, _ = cumulative_dynamic_auc(y_train, y_test, clf.predict(X_test), times)
            score = np.nanmean(aucs) # if there is a NaN value in one of the months than igonre it in the mean
            

        scores.append(score)

    # Calculate the average score across all folds
    average_score = np.mean(scores)
    return average_score


# hyperparamer tuning for XGBOOST
def tune_hyperparameters_xgb(X,y, cv,eval_metric ='ImBS',surv_name='Survival from enrolment',n_trials=200):
    
    # tuning trails for optuna
    n_trials = n_trials
    
    # ---------- #
    # base model #
    # ---------- #
    
    
    base_params = {'n_estimators': 20,
                   'learning_rate': 0.5,
                   'max_depth' : 2,
                   'subsample' : 0.3,
                   'loss': 'coxph',
                   'criterion' : 'friedman_mse',
                   'min_samples_split' : 2,
                   'min_samples_leaf' : 1,
                   'min_weight_fraction_leaf' : 0.0, 
                   'min_impurity_decrease' : 0.0,
                   'max_features' : None,
                   'max_leaf_nodes' : None,
                   'validation_fraction' : 0.1,
                   'n_iter_no_change' : None,
                   'tol' : 0.0001,
                   'dropout_rate' : 0.3,
                   'verbose' : 1,
                   'ccp_alpha' : 0.0,
                   'random_state' : 42 
                  }
    
    
    
    # set basic parameters 
    params = base_params
    
    scores_base =[]
    folds = list(cv.split(X, y))
    for train_idx, test_idx in folds:
        
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Initialize the RandomForestClassifier with suggested parameters
        clf = GradientBoostingSurvivalAnalysis(**params)
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        # Calculate the accuracy (or any other metric)
        if eval_metric == "ImBS":
            score = my_ImBS(clf, X_test, y_test, y_train, surv_name=surv_name)
        elif eval_metric == "CI":
             score, *_ = concordance_index_ipcw(y_train, y_test, clf.predict(X_test))
        elif eval_metric == "AUC":
            if times is None:
                times = np.linspace(y_test[surv_name].min(), y_test[surv_name].max(),50)
            aucs, _ = cumulative_dynamic_auc(y_train, y_test, clf.predict(X_test), times)
            score = np.nanmean(aucs) # if there is a NaN value in one of the months than igonre it in the mean
        
        scores_base.append(score)

    # Calculate the average score across all folds
    baseline_score = np.mean(scores_base)
    print(baseline_score)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    func = lambda trial: objective_xgb(trial,X,y,cv,eval_metric)
    study.optimize(func, n_trials=n_trials,
                   show_progress_bar=True,callbacks=[print_best_callback]
                   )
    
    best_params = study.best_params
    base_params.update(best_params)

    print('params', params)
    print('Best value', study.best_value)
    
    return base_params, study.best_value,baseline_score




# plot final models AUC over time for train and test cohorts based on tuned model
def plot_AUC(times,times_test, auc_train,auc_test,
             ylab = None,xlab = None,legend_text = [14,12],
             ref_auc_train = None, ref_auc_test = None, model = None, pad = [0.9, 1.12],
             ticks_size = 14,labels_size = 18):
    
    plt.plot(times, auc_train[0], marker="o",label='UK discovery')
    plt.plot(times_test, auc_test[0], marker="o",label='PGB replication', color = 'purple')
    
    if ref_auc_train is not None:
        plt.plot(times, ref_auc_train[0],linestyle =':',color = '#1f77b4', lw = 2.5)
    if ref_auc_test is not None:
        plt.plot(times_test, ref_auc_test[0],linestyle =':',color = 'purple', lw = 2.5)
    if (ref_auc_test is not None) | (ref_auc_test is not None):
        full_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label=model)    
        dashed_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='isomiR prediction')
    if xlab is not None:
        plt.xlabel(xlab,fontsize=labels_size)
    if ylab is not None:
        plt.ylabel(ylab,fontsize=labels_size)
    plt.ylim(0.3,1)
    plt.grid(True,linestyle="--", alpha=0.7)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    if (ref_auc_test is not None) | (ref_auc_test is not None):
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.extend([full_line, dashed_line])
        labels.extend([model, 'isomiR prediction'])
        plt.legend(handles=handles, labels=labels,title_fontsize=legend_text[0],fontsize=legend_text[1],
                   bbox_to_anchor=(pad[0], pad[1]),ncol = 2)
    else:
        plt.legend(title_fontsize=14,fontsize=ticks_size,bbox_to_anchor=(pad[0], pad[1]),ncol = 2)
    

    
def rsf_predict_risk(X, times = prediction_vector_times_train,model =  tuned_model):
    """
    Custom prediction function for SHAP.
    Calculates risk score for each instance in X.
    """
    risks =model.predict(X)
    return risks

def round_to_nearest_half(x, base=0.005):
    return round(x / base) * base

def plot_forest(df,ci_low,ci_high,hr,middle_line = 0, y_ticks = "features",
                X_title = "$\log_2({HR})$", main_title = None, highlight_y_ticks = None,
                yticks_size = 14, xticks_size = 14,title_size = 22,x_size = 18,dotsize = 20,
                bold_ticks = False,x_tick_gap = 0.5,remove_frame= False, limit_num = False,
               sort_HR = False,log2= True, xlim = None):
    """
    Hazard ratio forest plot based on a dataframe provided

    Parameters:
    - df (pd.DataFrame): DataFrame containing Hazard ratios data (HR, 95% CI lower / upper, feature names in a column).
    - ci_low (str): Name of column for lower bound 95% HR condifdance interval values.
    - ci_high (str): Name of column for upper bound 95% HR condifdance interval values.
    - hr (str): Name of column for HRs values.
    - y_ticks (str): Name of column for feature name values.
    - middle_line (int): position to draw vertical lines indicating lack of signicance (defualt to 0, in log2 should be 1).
    - X_title (str): Label for the x-axis defualts to "$\log_2({HR})$".
    - main_title (str): A title to supply to plot (defualts to None).
    - highlight_y_ticks (str): features to highlight names by color (must be found in column y_ticks), defualts to None i.e. no highlight.
    - yticks_size (float): size of Y axis ticks in plot (defualts to 11).
    - xticks_size (float): size of X axis ticks in plot (defualts to 10).
    - bold_ticks (bool): whether to plot the Y axis ticks in "bold" (defaults to True).
    - x_tick_gap (float): steps between xlim to plot x ticks (defualt to 0.5)
    - sort_HR (bool): whether to plot features sorted by ascending HR (default to False).
    - log2 (bool): whether to plot log2 HR values (default to True)
    - xlim (list): default to None, if list takes the first and second arguments as range of X axis values
    - title_size (int): size of figure title, defaults to 22.
    - x_size (int): size of X axis label, defaults to 18.
    - remove_frame (bool): whether to remove the outline of the graph- defaults to False
    - limit_num (bool): whether to plot only edges and center X axis tick values.

    Returns:
    - If show_plot True: return KM curve/s
    - A list of dicts with the following stats per patient ['Feature','Log-rank p-value','C-index',
    'Univarite Cox Hazard Ratio','Univarite Cox p-value',
    'HR 95% confidance interval' (DF)  0 - lower range, 1 - upper range}]
    """
    new_df = df
    if sort_HR:
        new_df = df.sort_values(by = hr,ascending = True)
        new_df = new_df.reset_index(drop = True)
    
    # Create a forest plot
   
    # Plot the confidence intervals as horizontal lines
    for i, row in new_df.iterrows():
        if log2:
            plt.plot([np.log2(row[ci_low]), np.log2(row[ci_high])], [i, i], color="black", linewidth=1)
            plt.scatter(np.log2(row[hr]), i, color="black", zorder=3)
        else:
            plt.plot([row[ci_low], row[ci_high]], [i, i], color="black", linewidth=1)
            plt.scatter(row[hr], i, color="black",zorder=3,s=dotsize)
    # Add a vertical dotted line at HR=1
    plt.axvline(middle_line, color="gray", linestyle="dotted", linewidth=1)

    # Set y-axis ticks and labels
    y_labels = new_df[y_ticks] if y_ticks is not None else new_df.index
    y_positions = range(len(new_df))
        
    # Apply colors to specific y-ticks
    tick_colors = ['#32CD32' if label in (highlight_y_ticks or []) else 'black' for label in y_labels]
    
    if bold_ticks:
        plt.yticks(y_positions, y_labels, fontsize=yticks_size, fontweight='bold')
    else:
        plt.yticks(y_positions, y_labels, fontsize=yticks_size)
        
    plt.xticks(fontsize=xticks_size)

    # Manually set the color of each tick label
    ax = plt.gca()
    for tick_label, color in zip(ax.get_yticklabels(), tick_colors):
        tick_label.set_color(color)
    
    plt.xlabel(X_title,fontsize=x_size,family='sans-serif')
    plt.ylabel("")
    plt.title(main_title, fontsize=title_size,fontweight = 'bold')
    
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
        x_ticks =  x_ticks = np.arange(xlim[0], xlim[1] + x_tick_gap, x_tick_gap)
        x_ticks = np.round(x_ticks, 2)
        plt.xticks(x_ticks, fontsize=xticks_size)
        plt.xticks(x_ticks,fontsize=xticks_size)
    if remove_frame:
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y', length=0)
    if limit_num:
        xmin, xmax = plt.xlim()  # or manually: xmin = 0; xmax = 2
        # Round to nearest 0.5 or 0.1 as needed
        xmin = round_to_nearest_half(xmin)
        xmax = round_to_nearest_half(xmax)
        # Compute middle value
        xmid = middle_line

        # Set ticks at edges and middle
        plt.xticks([xmin, xmid, xmax])
    plt.tight_layout()
#     plt.show()
    

    
def format_p_value(p):
    """Formats p-values in scientific notation with 2 significant figures"""
    if p < 0.001:  # Use scientific notation for very small p-values
        return f"{p:.1e}"
    elif (p > 0.001) & (p < 0.01):  # Use standard decimal notation
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"

def KM_plots(df:pd.DataFrame, feature_cols, x_title,
             cohort_title = None,
             censorship_col = "outcome",
             survival_col = "Survival.from.enrolment",
             km_stats_loc:list = [0.3, 0.95],
             high_text_loc:list = [0.3, 0.8],
             low_text_loc:list = [0.3, 0.75],
             text_size = 14,
             xlim = None,
             cols = 4,
             sub_width = 15,
             sub_height = 30,
             hspace = 0.35,
             wsapce = 0.1,
             combine_title_x = [44,0.5, 0.04],
             combine_title_y = [44,0.08, 0.5],
             combine_plots = False, show_plot = True,  axs=None):
    
    """
    Kaplan-Meier survival analysis for binary features. Supports subplots when `ax` is provided.

    Parameters:
    - df (pd.DataFrame): DataFrame containing survival data.
    - feature_cols (list): List of features to plot.
    - x_title (str): Label for the x-axis.
    - cohort_title (str, optional): Title prefix for the plot.
    - censorship_col (str): Column name for censorship (event occurrence).
    - survival_col (str): Column name for survival time.
    - combine_plots (bool): Whether to combine multiple feature plots in a grid.
    - show_plot (bool): Whether to show the plot (set to False for subplot integration).
    - ax (matplotlib.axes.Axes, optional): Predefined Matplotlib axis to plot in.
    - xlim (list): defualt to None, if list takes the first and second arguments as range of X axis values 
    - cols (int): number of columns when combine = True in subplots (deafults to 4)
    - sub_width (float): the width of each subplot when combine = True
    - sub_height (float): the height of each subplot when combine = True
    - hspace (float): verical space between subplots when combine = True
    - wspace (float): horizontal space between subplots when combine = True
    - combine_title_x (list): a list of floats of [title size, x position, y position] - parameters for overall X title when combine = True defualts to [44,0.5, 0.04]
    - combine_title_y (list): a list of floats of [title size, x position, y position] - parameters for overall y title when combine = True defualts to [44,0.5, 0.04]

    Returns:
    - If show_plot True: return KM curve/s
    - A list of dicts with the following stats per patient ['Feature','Log-rank p-value','C-index',
    'Univarite Cox Hazard Ratio','Univarite Cox p-value',
    'HR 95% confidance interval' (DF)  0 - lower range, 1 - upper range}]
    """
        
    results_univariate = []
    if combine_plots:
        # Create subplots grid
        n_features = len(feature_cols)
        n_cols = cols  # Number of columns in the grid
        n_rows = (n_features + n_cols - 1) // n_cols  # Calculate required rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(sub_width, sub_height), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=hspace, wspace=wsapce)
        axes = axes.flatten()  # Flatten axes for easy iteration

    for i, feature in enumerate(feature_cols):
        # Initialize KM Fitter
        kmf = lifelines.KaplanMeierFitter()

        # Create groups based on feature
        group_0 = df[df[feature] == 0]
        group_1 = df[df[feature] == 1]

        # Log-rank test for KM
        logrank_result = lifelines.statistics.logrank_test(
            group_0[survival_col],
            group_1[survival_col],
            event_observed_A=group_0[censorship_col],
            event_observed_B=group_1[censorship_col],
        )
        logrank_p = logrank_result.p_value
        logrank_stat = logrank_result.test_statistic

        # Cox proportional hazards model
        cph = lifelines.CoxPHFitter()
        df_for_cox = df[[feature, survival_col, censorship_col]]
        cph.fit(df_for_cox, duration_col=survival_col, event_col=censorship_col)
        c_index = cph.concordance_index_
        hr = cph.summary.loc[feature, "exp(coef)"]
        ci = pd.DataFrame({cph.summary.loc[feature, "exp(coef) lower 95%"],
                           cph.summary.loc[feature, "exp(coef) upper 95%"]})
        cox_p = cph.summary.loc[feature, "p"]
        
        # Save results
        results_univariate.append({
            "Feature": feature,
            "Log-rank p-value": logrank_p,
            "C-index": c_index,
            "Hazard Ratio": hr,
            "HR p-value" : cox_p,
            "Confidance interval" : ci
        })

        if show_plot:
            if combine_plots:
                ax = axes[i]
                kmf.fit(group_0[survival_col], group_0[censorship_col])
                kmf.plot_survival_function(ax =ax,ci_show=False, show_censors= True,
                                           color="blue", linewidth=2,legend=False)
                median_0 = kmf.median_survival_time_

                kmf.fit(group_1[survival_col], group_1[censorship_col])
                kmf.plot_survival_function(ax =ax,ci_show=False,show_censors= True,
                                           color="red", linewidth=2,legend=False)
                median_1 = kmf.median_survival_time_

                # Annotate chi-squared statistic and median survival
                ax.text(
                    km_stats_loc[0], km_stats_loc[1],
                    f"$\\chi^2$: {logrank_stat:.2f}, P: {format_p_value(logrank_p)}",
                    transform=ax.transAxes,
                    fontsize=text_size,
                    ha="left",
                    va="top"
                )
                ax.text(
                    low_text_loc[0],low_text_loc[1],  # Position for group 0 legend entry
                    f"Low Levels, Median Survival = {median_0:.1f}",
                    fontsize=text_size,
                    color="blue",
                    ha="left",
                    transform=ax.transAxes
                )
                ax.text(
                    high_text_loc[0],high_text_loc[1],  # Position for group 1 legend entry
                    f"High Levels, Median Survival = {median_1:.1f}",
                    fontsize=text_size,
                    color="red",
                    ha="left",
                    transform=ax.transAxes
                )
                ax.set_title(feature, fontsize=30)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.set_xlabel("")
                ax.tick_params(axis = 'both',labelsize=22)
                
            else:
                 # Individual plots
                # Fit survival curves
                if axs is None:
                    plt.figure()
                    axs = plt.gca()  # Get current axis
                    
                kmf.fit(group_0[survival_col], group_0[censorship_col])
                kmf.plot_survival_function(ci_show=False, show_censors= True,
                                           color="blue", linewidth=2,legend=False)
                median_0 = kmf.median_survival_time_

                kmf.fit(group_1[survival_col], group_1[censorship_col])
                kmf.plot_survival_function(ci_show=False,show_censors= True,
                                           color="red", linewidth=2,legend=False)
                median_1 = kmf.median_survival_time_
                
                if xlim is not None:
                    plt.xlim(xlim[0],xlim[1])
                    
                plt.text(
                    km_stats_loc[0], km_stats_loc[1],  # Position on the plot (adjust as needed)
                    f"$\\chi^2$ =  {logrank_stat:.2f}, P-value =  {format_p_value(logrank_p)}",
                    fontsize=text_size,
                    ha="left",
                    va="top",
                    transform=plt.gca().transAxes,
                )
                
                plt.text(
                    low_text_loc[0],low_text_loc[1],  # Position for group 0 legend entry
                    f"Low Levels, Median Survival = {median_0:.1f}",
                    fontsize=text_size,
                    color="blue",
                    ha="left",
                    transform=plt.gca().transAxes
                )
                plt.text(
                    high_text_loc[0],high_text_loc[1],  # Position for group 1 legend entry
                    f"High Levels, Median Survival = {median_1:.1f}",
                    fontsize=text_size,
                    color="red",
                    ha="left",
                    transform=plt.gca().transAxes
                )

                if cohort_title is None:
                    plt.title(f"{feature}", fontsize=22,fontweight = 'bold')
                elif cohort_title == "No":
                    plt.title(f"", fontsize=22,fontweight = 'bold')
                else:
                    plt.title(f"{cohort_title}", fontsize=22,fontweight = 'bold')
                plt.xlabel("Months from enrollment",fontsize=18)
                plt.ylabel("Survival Probability",fontsize=18)
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                
    if combine_plots:
        # Hide unused subplots
        for j in range(len(feature_cols), len(axes)):
            fig.delaxes(axes[j])

#           # Add shared X and Y labels
        fig.text(combine_title_x[1], combine_title_x[2], f'{x_title}', ha="center", fontsize=combine_title_x[0],fontweight = 'bold')
        fig.text(combine_title_y[1], combine_title_y[2], "Survival Probability", va="center", rotation="vertical", fontsize=combine_title_y[0],fontweight = 'bold')
#         plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])


    return results_univariate


def convert_to_quantiles(df,feature_cols, survival_col, event_col,q):
    """
    convert continous varaibles to category ones based on thier chosen qunatiles 

    Parameters:
    df (pd.DataFrame): DataFrame containing survival data.
    feature_cols (list): Columns containing features
    survival_col (str): Column name for survival time.
    event_col (str): Column name for event occurrence (1 = event, 0 = censored).
    q (int) : Number of quantiles to cut

    Returns:
    Dataframe: converted dataframe to categories.
    """
    df_new = df.copy()
    survival_vars = df_new[[survival_col, event_col]]
    thresholds = {}
    for feat in feature_cols:
        cat, bins = pd.qcut(df_new[feat],q,labels=range(q),retbins=True)
        df_new[feat] = cat
        thresholds[feat] = bins
    return(df_new,thresholds)


def cox_univariate_binarization(df1, df2, survival_col, event_col, cutpoint = None, return_matrix = False):
    """
    Iterates over all columns in df1, binarizing each column at all unique values.
    Runs Cox Univariate analysis on df1 and df2 (using df1's thresholds).
    
    Parameters:
    df1 (pd.DataFrame): Discovery dataset.
    df2 (pd.DataFrame): Replication dataset.
    survival_col (str): Name of the survival time column.
    event_col (str): Name of the event (censorship) column.
    cutpoint (str): if median, than cut point for the specific median of each feature from df1
    
    Returns:
    pd.DataFrame: DataFrame with Cox univariate results, including feature name, binarization threshold, 
                  and results from both discovery and replication datasets.
    """
    results = []

    for i,feature in enumerate(df1.columns):
        if feature in [survival_col, event_col]:  
            continue  # Skip survival and event columns
        if cutpoint is None:
            unique_values = np.sort(df1[feature].dropna().unique())  # Get sorted unique values from discovery dataset
            
            failed_runs = []
            for threshold in unique_values:
                # Binarize feature at this threshold for both datasets
                df1_binarized = (df1[feature] > threshold).astype(int)
                df2_binarized = (df2[feature] > threshold).astype(int)

                # Prepare dataset for Cox analysis (Discovery)
                df1_for_cox = df1[[survival_col, event_col]].copy()
                df1_for_cox["binarized_feature"] = df1_binarized  

                # Prepare dataset for Cox analysis (Replication)
                df2_for_cox = df2[[survival_col, event_col]].copy()
                df2_for_cox["binarized_feature"] = df2_binarized  

                # Run Cox Univariate Analysis (Discovery)
                cph1 = lifelines.CoxPHFitter()
                try:
                    cph1.fit(df1_for_cox, duration_col=survival_col, event_col=event_col)
                    c_index1 = cph1.concordance_index_
                    hr1 = cph1.summary.loc["binarized_feature", "exp(coef)"]
                    ci_lower1 = cph1.summary.loc["binarized_feature", "exp(coef) lower 95%"]
                    ci_upper1 = cph1.summary.loc["binarized_feature", "exp(coef) upper 95%"]
                    cox_p1 = cph1.summary.loc["binarized_feature", "p"]
                except Exception as e:
                    print(f"Skipping {feature} (Threshold {threshold}, Discovery): {e}")
                    failed_runs.append(np.nan)
                    if len(failed_runs) == len(unique_values):
                        c_index1, hr1, ci_lower1, ci_upper1, cox_p1 = [np.nan] * 5  # Fill missing values with NaN
                    else:
                        continue  # Skip this threshold if an error occurs

                # Run Cox Univariate Analysis (Replication)
                cph2 = lifelines.CoxPHFitter()
                try:
                    cph2.fit(df2_for_cox, duration_col=survival_col, event_col=event_col)
                    c_index2 = cph2.concordance_index_
                    hr2 = cph2.summary.loc["binarized_feature", "exp(coef)"]
                    ci_lower2 = cph2.summary.loc["binarized_feature", "exp(coef) lower 95%"]
                    ci_upper2 = cph2.summary.loc["binarized_feature", "exp(coef) upper 95%"]
                    cox_p2 = cph2.summary.loc["binarized_feature", "p"]
                except Exception as e:
                    print(f"Skipping {feature} (Threshold {threshold}, Replication): {e}")
                    c_index2, hr2, ci_lower2, ci_upper2, cox_p2 = [np.nan] * 5  # Fill missing values with NaN

                # Store results
                results.append({
                    "Feature": feature,
                    "Threshold": threshold,
                    "C-index": c_index1,
                    "HR": hr1,
                    "95% CI Lower": ci_lower1,
                    "95% CI Upper": ci_upper1,
                    "P-value": cox_p1,
                    "C-index_replication": c_index2,
                    "HR_replication": hr2,
                    "95% CI Lower_replication": ci_lower2,
                    "95% CI Upper_replication": ci_upper2,
                    "P-value_replication": cox_p2
                })
                
        elif cutpoint == 'median':
            
            threshold = df1[feature].median()
            df1_binarized = (df1[feature] > threshold).astype(int)
            df2_binarized = (df2[feature] > threshold).astype(int)

            # Prepare dataset for Cox analysis (Discovery)
            df1_for_cox = df1[[survival_col, event_col]].copy()
            df1_for_cox["binarized_feature"] = df1_binarized  

            # Prepare dataset for Cox analysis (Replication)
            df2_for_cox = df2[[survival_col, event_col]].copy()
            df2_for_cox["binarized_feature"] = df2_binarized  
            
            # Run Cox Univariate Analysis (Discovery)
            cph1 = lifelines.CoxPHFitter()
            try:
                cph1.fit(df1_for_cox, duration_col=survival_col, event_col=event_col)
                c_index1 = cph1.concordance_index_
                hr1 = cph1.summary.loc["binarized_feature", "exp(coef)"]
                ci_lower1 = cph1.summary.loc["binarized_feature", "exp(coef) lower 95%"]
                ci_upper1 = cph1.summary.loc["binarized_feature", "exp(coef) upper 95%"]
                cox_p1 = cph1.summary.loc["binarized_feature", "p"]
            except Exception as e:
                print(f"Skipping {feature} (Threshold {threshold}, Discovery): {e}")
                continue  # Skip this threshold if an error occurs

            # Run Cox Univariate Analysis (Replication)
            cph2 = lifelines.CoxPHFitter()
            try:
                cph2.fit(df2_for_cox, duration_col=survival_col, event_col=event_col)
                c_index2 = cph2.concordance_index_
                hr2 = cph2.summary.loc["binarized_feature", "exp(coef)"]
                ci_lower2 = cph2.summary.loc["binarized_feature", "exp(coef) lower 95%"]
                ci_upper2 = cph2.summary.loc["binarized_feature", "exp(coef) upper 95%"]
                cox_p2 = cph2.summary.loc["binarized_feature", "p"]
            except Exception as e:
                print(f"Skipping {feature} (Threshold {threshold}, Replication): {e}")
                c_index2, hr2, ci_lower2, ci_upper2, cox_p2 = [np.nan] * 5  # Fill missing values with NaN
            
            # rename feature after analysis
            df1_for_cox = df1_for_cox.rename(columns={"binarized_feature": feature})
            df2_for_cox = df2_for_cox.rename(columns={"binarized_feature": feature})
            
            # Store results
            results.append({
                "Feature": feature,
                "Threshold": threshold,
                "C-index": c_index1,
                "HR": hr1,
                "95% CI Lower": ci_lower1,
                "95% CI Upper": ci_upper1,
                "P-value": cox_p1,
                "C-index_replication": c_index2,
                "HR_replication": hr2,
                "95% CI Lower_replication": ci_lower2,
                "95% CI Upper_replication": ci_upper2,
                "P-value_replication": cox_p2
            })

    return pd.DataFrame(results)



def generate_features(df, col1, col2, rolling_window=3, scale = None):
    """
    Generate new features using mathematical operations between two continuous columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    col1 (str): Name of the first continuous column.
    col2 (str): Name of the second continuous column.
    scale (str): if None no scaling, if "Z" than StandardScaler, if "minmax" than MinMaxScaler

    Returns:
    pd.DataFrame: DataFrame with new engineered features.
    """
    
    df_new = df.copy()
    df_new = df_new.round()
    
    # Min-Max Normalization
    if scale == 'minmax':
        min_max_scaler = MinMaxScaler()
        df_new[[col1,col2]] = min_max_scaler.fit_transform(df_new[[col1,col2]])
    # Standardization (Z-score)    
    elif scale == 'Z':
        standard_scaler = StandardScaler()
        df_new[[col1,col2]] = standard_scaler.fit_transform(df_new[[col1,col2]])

    # Basic Arithmetic Operations
    df_new[f"{col1}_{col2}_sum"] = df[col1] + df[col2]
    
    if df[col1].mean() > df[col2].mean():
        df_new[f"{col1}_{col2}_diff"] = df[col1] - df[col2]
    else:
        df_new[f"{col2}_{col1}_diff"] = df[col2] - df[col1]
        
    df_new[f"{col1}_{col2}_product"] = df[col1] * df[col2]
    df_new[f"{col1}_{col2}_ratio"] = df[col1] / (df[col2] + 1e-9)  # Avoid division by zero
    df_new[f"{col2}_{col1}_ratio"] = df[col2] / (df[col1] + 1e-9)
    
    df_new[f"{col1}_{col2}_mean"] = df[[col1,col2]].mean(axis = 1)
    df_new[f"{col1}_{col2}_abs_diff"] = abs(df[col1] - df[col2])

    # Logarithmic & Exponential Transformations
    df_new[f"{col1}_{col2}_log_ratio"] = np.log2(df[col1]) - np.log2(df[col2])
    df_new[f"{col1}_{col2}_log_sum"] = np.log2(df[col1] + df[col2])
    df_new[f"{col1}_{col2}_log_product"] = np.log2(df[col1]) + np.log2(df[col2])
#     df_new[f"{col1}_{col2}_exp_sum"] = np.exp(df[col1]) + np.exp(df[col2])
#     df_new[f"{col1}_{col2}_exp_diff"] = np.exp(df[col1]) - np.exp(df[col2])

    # Polynomial & Power Transformations
    df_new[f"{col1}_{col2}_squared_diff"] = (df[col1] - df[col2])**2
    df_new[f"{col1}_{col2}_sqrt_sum"] = np.sqrt(df[col1] + df[col2])  # Avoid sqrt of negative
    df_new[f"{col1}_squared"] = df[col1] ** 2
    df_new[f"{col2}_squared"] = df[col2] ** 2
    df_new[f"{col1}_{col2}_sum_sqrt"] = df[col1] ** 2 + df[col2] ** 2
    df_new[f"{col1}_{col2}_sum_cubed"] = df[col1]**3 + df[col2]**3
    df_new[f"{col1}_cubed"] = df[col1] ** 3
    df_new[f"{col2}_cubed"] = df[col2] ** 3
    df_new[f"{col1}_{col2}_interaction"] = (df[col1] * df[col2])**0.5  # Geometric mean
    df_new[f"{col1}_{col2}_sqrt_interaction"] = (df[col1] * df[col2])**2

    # Statistical & Rank-Based Operations
    df_new[f"{col1}_{col2}_rank_diff"] = rankdata(df[col1]) - rankdata(df[col2])
    df_new[f"{col1}_{col2}_rank_sum"] = rankdata(df[col1]) + rankdata(df[col2])
    df_new[f"{col1}_{col2}_quantile_diff"] = df[col1].rank(pct=True) - df[col2].rank(pct=True)
    
    # Rolling Sum Difference
    df_new[f"{col1}_{col2}_rolling_sum_abs_diff"] = (
        abs(df[col1].rolling(rolling_window, min_periods=1).sum() - df[col2].rolling(rolling_window, min_periods=1).sum())
    )

    # Distance & Similarity Measures
    df_new[f"{col1}_{col2}_euclidean"] = np.sqrt((df[col1] - df[col2])**2)
    df_new[f"{col1}_{col2}_cosine_sim"] = (df[col1] * df[col2]) / (np.linalg.norm(df[col1]) * np.linalg.norm(df[col2]) + 1e-9)
    
    #Mahalanobis distance of each value from its column distribution
    # this is not actually a combination of the two columns
    for col in [col1,col2]:
        df_new[f"{col}_mahalanobis"] = np.sqrt(((df[col] - df_new[col].mean()) ** 2) / (df_new[col].var() + 1e-9))  # Avoid div by zero
    
    return df_new


def rank_thresholds_per_feature(df):
    """
    Rank thresholds per feature based on HR direction and replication consistency.

    1. If HR > 1, prefer highest HR.
    2. If HR < 1, prefer lowest HR.
    3. Minimize absolute difference between HR and HR_replication.

    Parameters:
    df (pd.DataFrame): DataFrame containing "Feature", "HR", "HR_replication", and "Threshold".

    Returns:
    pd.DataFrame: DataFrame with an additional column "Rank" (lower rank is better).
    """
    def rank_within_feature(sub_df):
        # Compute absolute difference between HR and HR_replication
        sub_df["HR_diff"] = abs(sub_df["HR"] - sub_df["HR_replication"])

        # Create a ranking column based on HR direction
        sub_df["HR_rank"] = sub_df.apply(lambda row: -row["HR"] if row["HR"] > 1 else row["HR"], axis=1)

        # Rank first by HR direction preference, then by replication consistency (HR_diff)
        sub_df = sub_df.sort_values(["HR_rank", "HR_diff"], ascending=[True, True])
        sub_df["Rank"] = range(1, len(sub_df) + 1)  # Assign rank starting from 1

        # Drop auxiliary columns used for ranking
        sub_df = sub_df.drop(columns=["HR_rank", "HR_diff"])

        return sub_df

    # Apply ranking function per feature
    return df.groupby("Feature", group_keys=False).apply(rank_within_feature)
