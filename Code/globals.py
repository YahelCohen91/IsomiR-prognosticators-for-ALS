import platform
import threading
from enum import Enum
import os
global savefig_path
global os_name
os_name = platform.system()
if os_name == 'Linux':
    savefig_path = f'/home/projects/hornsteinlab/yahelc/R/isoMIRs/isomiRs_II/results' #RSF_imbs_iterations , ImBS_XGBOOST_iteration/
    savefig_path_internal =  os.path.join(savefig_path,'r_{0}')
    
else:

    savefig_path = f"D:\\yahel\\phd\\isomiRs\\Ilan's\\IsomiRsII\\results"
    savefig_path_internal =  os.path.join(savefig_path,'r_{0}')


feature_name_name = 'feature_importances_'
second_round_file_prefix = 'f_importance_multiple_alpha'

# Set the folder path and prefix for your files
if os_name == 'Linux':
    folder_path = f'/home/projects/hornsteinlab/yahelc/R/isoMIRs/isomiRs_II/results' #RSF_imbs_iterations , ImBS_XGBOOST_iteration/
    folder_path_internal = os.path.join(folder_path,'r_{0}')
else:

    folder_path = "D:\\yahel\\phd\\isomiRs\\Ilan's\\IsomiRsII\\results"  
    folder_path_internal = os.path.join(folder_path,'r_{0}')



isomiRsII_data_train_sheet_name = 'corrected_isomiR_UK_154p_09_n2'   


if os_name == 'Linux':
    root = '/home/projects/hornsteinlab/yahelc/R/isoMIRs/isomiRs_II/'
    isomiRsII_data_train = os.path.join(root,'uk_us_corrected_data_121123.xlsx')
    isomiRsII_data_test = os.path.join(root, 'corrected_umi_replication_new_analysis_censored_by_Endpt_reason.txt')
    
else:
    root = "D:\\yahel\\phd\isomiRs\\Ilan's\\"
    isomiRsII_data_train = f"D:\\yahel\\phd\isomiRs\\Ilan's\\git_repo\\uk_us_corrected_data_121123.xlsx"
    isomiRsII_data_test = f"D:\\yahel\\phd\\isomiRs\\Ilan's\\git_repo\\corrected_umi_replication_new_analysis_censored_by_Endpt_reason.txt"

mature_train = 'Mature_miRNA_train.txt'

mature_test =  'Mature_miRNA_test.txt'

file_prefix = 'f_importance_multiple_alpha'  # 'f_importance_sorted_pipe'   #'your_prefix_here'

# for second floor
#first_round_DT_token = ['2024_11_11_15_41_01','2024_11_11_19_22_28', '2024_11_13_19_42_56'] #['2024_11_02_18_17_46','2024_11_02_18_49_21', '2024_11_03_10_34_57']
# for RSF
#first_round_DT_token = '2024_11_07_15_37_59'# '2024_11_09_08_36_31' # #'2024_11_07_15_37_59' #'2024_11_07_14_34_39' #'2024_11_06_18_07_04'  
# for XGBOOST
# first_round_DT_token = '2024_11_10_11_32_18' #'2024_11_09_09_49_36' # '2024_11_10_13_23_55' # '2024_11_10_11_32_18' - had 15 features did not improve #'2024_11_09_09_49_36' #'2024_11_09_08_56_37' #'2024_11_06_18_29_58'
# for XGBOOST CI
first_round_DT_token = '2024_11_15_11_57_03' #'2024_11_17_16_00_05' #'2024_11_17_09_38_45' #'2024_11_16_11_28_43' #

multiple_runs_stats_file_name =  'features_stats_over_multiple_runs_{}.xlsx'
f_w_tab_name =  'gen f importance'
m_w_tab_name =  'gen m importance'
m_f_inc_tab_name = 'f_inc'

filter_col_name = 'features'
numerical_threshold = 0.02  # Adjust this threshold as needed
summary_tab_name = 'summary_mBS_family' #'summary_IBS_family'
summary_tab_name_ci = 'AUC-t'

#summary_file_lock = threading.Lock()

excel_file_lock = threading.Lock()  #  #So pandas to_excel is thread safe while openpyxl and xlsxwriter are not - this is true even if you write to different excel files
# class syntax
class Usecase(Enum):  #pipelines
    Simple = 1
    Features_Select = 2
    Intra_Batch = 3
    SFS_Inter_Batch = 4 # what was used to select isomiR panel

class Usecase_main_model(Enum):
    Without = 1
    Cox = 2
    Boosting = 3
    Bagging = 4
    Cox_Regulation_free = 5   # without regularization


class Usecase_features_selection(Enum):
    Without = 1
    Survival_Univariate_cox = 2  #survival - the only model
    Stats = 3
    Extra_tree = 4   #ExtraTreesClassifier
    RF_Classification = 5   #RandomForestClassifier
    RF_Regression = 6
    RF_Permutation = 7 # permutation_importance
    SFS_KNN = 8
    SFS_Extra_tree = 9  # ExtraTreesClassifier
    SFS_RF_Classification = 10  # RandomForestClassifier
    SFS_RF_Regression = 11
    SFS_RF_Permutation = 12  # permutation_importance
    SFS_CoxPH = 13
    SFS_CoxNet = 14
    SFS_RSF = 15  # bagging
    SFS_XGB_Survival = 16 #boosting
    

class second_round_features_selection(Enum):
    """
    based on first floor results
    feature selection mathods

    each is a paramter to read different files and extract relevant features based on files
    """
    none = 1 # start from raw/ first floor
    from_f_importance_files= 2
    from_f_stats_over_multiple_runs_file = 3
    from_list_of_f_names_file = 4
    from_top_runs_file = 5