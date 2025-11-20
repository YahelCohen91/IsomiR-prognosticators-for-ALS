import sys,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import csv
import xlsxwriter

#matplotlib inline
#try to avoid lifelines if you dont have to (different pack. actually the lifeline has some f which lacl in sckit but less models)
#see https://gist.github.com/jackyko1991/bd0e605fa03b2c3e244d08db2b68edd8
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import survival_table_from_events
from lifelines.utils import restricted_mean_survival_time
from lifelines.plotting import rmst_plot
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from itertools import zip_longest
import scipy.stats

#savefig_path = f'c:\AlgoCode\Hornstein\isoMIRsII\outputs\data_analysis_tool_output'
import globals as globals
import itertools

random_seed = 0

survival_name = 'Survival from onset'
survival_name2 = 'Survival from enrolment'
death_name = 'outcome'


#from isomiRsII_main import data_load
from data_loaders import data_load
def KM_specific_time_point(y_train, y_test, time_limit = 50, bshow_images = False, bclear_momory = True):
    try:
        fitter_train = KaplanMeierFitter()
        fitter_test = KaplanMeierFitter()


        fitter_train.fit(y_train[survival_name2], event_observed=y_train[death_name], label="train")
        restr_mean_survival_time_train = restricted_mean_survival_time(fitter_train, t=time_limit)
        #print(restr_mean_survival_time_train)
        fitter_test.fit(y_test[survival_name2], event_observed=y_test[death_name], label="test")
        restr_mean_survival_time_test = restricted_mean_survival_time(fitter_test, t=time_limit)
        #print(restr_mean_survival_time_test)

        dif_test_train = restr_mean_survival_time_test - restr_mean_survival_time_train
        dif_perc = 100*2*dif_test_train / (restr_mean_survival_time_test + restr_mean_survival_time_train)
        ax = plt.subplot(311)
        ax.set_title("Time point limit: %s. RMST Delta %4.2f .%%  %3i" % (time_limit, dif_test_train, round(dif_perc)))
        rmst_plot(fitter_train, t=time_limit, ax=ax)

        ax = plt.subplot(312)
        rmst_plot(fitter_test, t=time_limit, ax=ax)

        ax = plt.subplot(313)
        rmst_plot(fitter_train, model2=fitter_test, t=time_limit, ax=ax)
        #ax.set_title("Time limit: %s. RMST %s - %s" % (time_limit, str(restr_mean_survival_time_train),str(restr_mean_survival_time_test)))
        plt.tight_layout()
        if bclear_momory:
            plt.savefig(os.path.join(globals.savefig_path, 'KM_specific_time_point_{0}.png'.format(time_limit), bbox_inches='tight'))
        else:
            plt.savefig(os.path.join(globals.savefig_path, 'KM_specific_time_point_cont_{0}.png'.format(time_limit), bbox_inches='tight'))
        if bshow_images:
            plt.show()
        if bclear_momory:
            plt.close()

        datapoint = {
            'time_limit': time_limit,
            'restr_mean_survival_time_train': restr_mean_survival_time_train,
            'restr_mean_survival_time_test': restr_mean_survival_time_test
        }

        return datapoint

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad KM_specific_time_point')



def KM_survival_plot(y_train, y_test, feature_train, feature_test, bshow_images = False):
    try:

        fitter_train = KaplanMeierFitter()
        fitter_test = KaplanMeierFitter()

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        # patients GT
        fitter_train.fit(y_train[survival_name2], event_observed= y_train[death_name], label="train")
        fitter_train.plot_survival_function(ax=ax1, show_censors=True, censor_styles={'ms': 3, 'marker': 's', 'markeredgecolor': '#ff0000'},color='#ff9900')
        fitter_test.fit(y_test[survival_name2], event_observed=y_test[death_name], label="test")
        fitter_test.plot_survival_function(ax=ax1, show_censors=True, censor_styles={'ms': 3, 'marker': 's', 'markeredgecolor': '#00ff00'},color='#99ff00')
        ax1.set_ylim(0, 1.05)
        # patients different features
        # watch out - entering f1 as ranked (i.e. fit it to survi) after doing so for any GT will break the concordance/same record linkage
        #fitter_train_feature.fit(feature_train, event_observed=y_train['outcome'], label="feature train") - i.e. this is wrong
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        #for i in range(len(feature_train)):
        i = 0
        ax2.scatter(y_train[survival_name2], feature_train, c=colors[2*i], alpha=0.3)
        ax2.scatter(y_test[survival_name2], feature_test, c=colors[2*i+1], alpha=0.3)
        red_patch = mpatches.Patch(color='red', label='train')   #feature_train.name, feature_test.name
        ax2.legend(handles=[red_patch])
        # Test pairwise differences between all classes
        #p_train = logrank_test( feature_train, y_train['Survival from onset'], y_train['outcome'], y_train['outcome']).p_value
        #p_test = logrank_test(feature_test, y_test['Survival from onset'], y_test['outcome'],y_test['outcome']).p_value

        # Add p-values as anchored text
        #ax1.add_artist(AnchoredText("\n".join([p_train, p_test]), loc=8, frameon=False))

        gt_train_name = y_train.dtype.names[1]
        gt_test_name = y_test.dtype.names[1]
        feature_train_name = feature_train.name # feature_train.columns[0]
        feature_test_name = feature_test.name # feature_test.columns[0]

        ax1.set_title("GT: %s - %s" % (gt_train_name, gt_test_name))
        ax2.set_ylabel("f: %s - %s" % (feature_train_name, feature_test_name))
        ax2.set_xlabel("timeline: %s - %s" % (gt_train_name, gt_test_name))
        ax2.margins(x=0)
        ax1.margins(x=0)
        #add_at_risk_counts(fitter_train, fitter_test, ax=ax1)
        plt.tight_layout()
        plt.savefig(globals.savefig_path+'\KM_survival_plot_{0}_{1}.png'.format(feature_train_name,feature_test_name), bbox_inches='tight')
        if bshow_images:
            plt.show()
    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad survival_plot')

def survival_grouping(y_train, y_test, feature_train, feature_test):
    try:
        gt_train_name = y_train.dtype.names[1]
        gt_test_name = y_test.dtype.names[1]

        table_train = survival_table_from_events(y_train[survival_name2], y_train[death_name])
        table_test = survival_table_from_events(y_test[survival_name2], y_test[death_name])

        survival_bin = [-1,10,20,30,40,50,60,70,80,90,100,500]
        table_train['bucket'] = pd.cut(table_train.index,survival_bin)
        table_train_bin = table_train[['bucket','observed']].groupby('bucket').sum()
        helper = table_train[['bucket', 'censored']].groupby('bucket').sum()  # was 'censored' by mistake
        table_train_bin = pd.concat([helper, table_train_bin], axis=1)

        table_test['bucket'] = pd.cut(table_test.index, survival_bin)
        table_test_bin = table_test[['bucket', 'observed']].groupby('bucket').sum()
        helper = table_test[['bucket', 'censored']].groupby('bucket').sum()
        table_test_bin = pd.concat([helper, table_test_bin], axis=1)

        #feature_train_name = feature_train.name  # feature_train.columns[0]
        #feature_test_name = feature_test.name  # feature_test.columns[0]

        with pd.ExcelWriter(os.path.join(globals.savefig_path, 'KM_survival_events_bins.xlsx')) as writer:  # doctest: +SKIP
            table_train.to_excel(writer, sheet_name=('Surv train'), engine='xlsxwriter')
            table_test.to_excel(writer, sheet_name=('Surv test'), engine='xlsxwriter')
            table_train_bin.to_excel(writer, sheet_name=('bin train'), engine='xlsxwriter')
            table_test_bin.to_excel(writer, sheet_name=('bin  test'), engine='xlsxwriter')

        #now do it with dous of survival time & f value - f after f
        col_names = y_train.dtype.names
        df = pd.DataFrame(y_train, columns = col_names)
        feature_train['index'] = np.arange(len(feature_train))
        feature_train.set_index('index',inplace=True)
        helper= pd.concat([df, feature_train], axis=1)
        helper['bucket'] = pd.cut(helper[survival_name2], survival_bin)

        df2 = pd.DataFrame(y_test, columns=col_names)
        feature_test['index'] = np.arange(len(feature_test))
        feature_test.set_index('index', inplace=True)
        helper2 = pd.concat([df2, feature_test], axis=1)
        helper2['bucket'] = pd.cut(helper2[survival_name2], survival_bin)

        dict_dp = {}
        for i,name in enumerate(feature_train.columns):
            f_train_bin = helper[['bucket', feature_train.columns[i]]].groupby('bucket').mean()
            f_train_bin.rename(columns={feature_train.columns[i]: (feature_train.columns[i] + '_mean')}, inplace = True)
            f_train_bin2 = helper[['bucket', feature_train.columns[i]]].groupby('bucket').std()
            f_train_bin2.rename(columns={feature_train.columns[i]: (feature_train.columns[i] + '_std')}, inplace = True)
            table_train_features_bin = pd.concat([f_train_bin, f_train_bin2], axis=1)

            f_test_bin = helper2[['bucket', feature_test.columns[i]]].groupby('bucket').mean()
            f_test_bin.rename(columns={feature_test.columns[i]: feature_test.columns[i] + '_mean'})
            f_test_bin2 = helper2[['bucket', feature_test.columns[i]]].groupby('bucket').std()
            f_test_bin2.rename(columns={feature_test.columns[i]: feature_test.columns[i] + '_std'})
            table_test_features_bin = pd.concat([f_test_bin, f_test_bin2], axis=1)
            dict_dp[feature_train.columns[i]]= table_train_features_bin
            dict_dp[feature_test.columns[i]] =  table_test_features_bin
            with pd.ExcelWriter(globals.savefig_path + '\KM_features_by_survival_events_bins_{}.xlsx'.format(name), mode = 'w+') as writer:  # doctest: +SKIP
                table_train_features_bin.to_excel(writer, sheet_name=('train_'+feature_train.columns[i]), engine='xlsxwriter')
                table_test_features_bin.to_excel(writer, sheet_name=('test_'+feature_test.columns[i]), engine='xlsxwriter')

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad survival_grouping')


def features_connections_1D(y_train, y_test, feature_train, feature_test):
    try:

        n_train = len(y_train[survival_name2])
        n_test = len(y_test[survival_name2])
        if n_train > n_test:
            y_corr_concor, y_p_val = scipy.stats.kendalltau(y_train[survival_name2][:n_test], y_test[survival_name2])
        else:
            y_corr_concor, y_p_val = scipy.stats.kendalltau(y_test[survival_name2][:n_train], y_train[survival_name2])

        list_dp_train = []
        list_dp_test = []
        for i,n in enumerate(feature_train):
            #print(i,n)
            yf_corr_concor, yf_p_val = scipy.stats.kendalltau(y_train[survival_name2], feature_train[n])
            #print('spearmanr',scipy.stats.spearmanr(y_train['Survival from onset'], feature_train[n]))
            #print('pearsonr', scipy.stats.pearsonr(y_train['Survival from onset'], feature_train[n]))
            #print('kendalltau', yf_corr_concor, yf_p_val)
            datapoint = {
                'y': survival_name2,
                'f': n,
                'cohort': 'train',
                'conc_corr': yf_corr_concor,
                'p_val_cond': yf_p_val
            }
            list_dp_train.append(datapoint)
        for i,n in enumerate(feature_test):
            yf_corr_concor, yf_p_val = scipy.stats.kendalltau(y_test[survival_name2], feature_test[n])
            datapoint = {
                'y': survival_name2,
                'f': n,
                'cohort': 'test',
                'conc_corr': yf_corr_concor,
                'p_val_cond': yf_p_val
            }
            list_dp_test.append(datapoint)

        datapoint = {
            'y': survival_name2,
            'f': survival_name2,
            'cohort': 'train-test',
            'conc_corr': y_corr_concor,
            'p_val_cond': y_p_val
        }
        list_dp_test.append(datapoint)
        list_dp_train.append(datapoint)

        PARAMS.save(list_dic=list_dp_train, file_name=os.path.join(globals.savefig_path, 'features_connections_1D_train.csv'))
        PARAMS.save(list_dic=list_dp_test, file_name=os.path.join(globals.savefig_path, 'features_connections_1D_test.csv'))

    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad features_connections_1D')


def top_correlated_features(train_file_name, test_file_name):
    try:

        df_train = pd.read_csv(train_file_name, sep=',', lineterminator='\r')
        df_test = pd.read_csv(test_file_name, sep=',', lineterminator='\r')

        # Sort features based on their CCC values in descending order.
        sorted_features_train = df_train.sort_values(by='conc_corr', ascending=False, key=lambda x: abs(x))
        sorted_features_test = df_test.sort_values(by='conc_corr', ascending=False, key=lambda x: abs(x))
        sorted_features_test = sorted_features_test.rename(columns = {"p_val_cond":"p_val_cond_test","conc_corr":"conc_corll_test"})
        sorted_features = pd.merge(left = sorted_features_train, right = sorted_features_test, how = 'left', left_on = 'f',right_on = 'f')

        with pd.ExcelWriter(os.path.join(globals.savefig_path , f'corr_train_features_corresponded_test.xlsx'), mode='w') as writer:  # doctest: +SKIP
            sorted_features.to_excel(writer, sheet_name=('f'),engine='xlsxwriter')
        with pd.ExcelWriter(os.path.join(globals.savefig_path ,f'corr_train_features.xlsx'), mode='w') as writer:  # doctest: +SKIP
            sorted_features_train.to_excel(writer, sheet_name=('f'),engine='xlsxwriter')



    except Exception as ex:
        print(ex.__context__)
        print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('bad features_connections_1D')


## create csv out of list of dicts
class PARAMS(object):
    @classmethod
    def save(cls, list_dic, file_name):
        try:
            header = list_dic[0].keys()
            with open(file_name,'w', newline='') as file:
                writer = csv.DictWriter(file,fieldnames = header)
                writer.writeheader()
                writer.writerows(list_dic)

        except Exception as ex:
            print(ex.__context__)
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('bad PARAMS')


if __name__ == "__main__":
    print("---------------Tool Start---------------")

    print("---------------KM_specific_time_point Start---------------")
    X_train, X_test, y_train, y_test = data_load()

    time_points = [10,20,30,40,50,60,70,80,90,100,150,200,300]
    list_dp = []
    for time_limit in time_points:
            datapoint= KM_specific_time_point(y_train, y_test, time_limit=time_limit,bshow_images = False, bclear_momory = True)
            list_dp.append(datapoint)
    PARAMS.save(list_dic = list_dp, file_name = os.path.join(globals.savefig_path,'KM_specific_time_point.csv'))     ## create csv out of list of dicts
    #os.system("pause - KM_specific_time_point concluded")

    print("---------------KM_survival_plot Start---------------")
    # For each clinical feature
    features = ['hsa-let-7a-5p']#, 'hsa-miR-99b-5p.T']  #TT is not mutual
    features = ['hsa-miR-16-5p.T']
    features = ['hsa-miR-660-5p']
    features = ['hsa-miR-301a-3p']
    #features11_notinboth = ['miR.191.5p.A19G','let.7a.5p.A17G','miR.221.3p.A1G.T','miR.103a.3p.A1G','let.7b.5p.T1A.T','miR.1.3p.ts.G1T.A3G','miR.146a.5p','miR.1.3pT20G.T22G','miR.339.5p.g','let.7b.5p.T.G','miR.16.5p.T18A']
    for feature in features:
        feature_train = X_train[feature]
        feature_test = X_test[feature]
        KM_survival_plot(y_train, y_test, feature_train, feature_test, bshow_images = True)
        #survival_plot(clinical, NelsonAalenFitter(), "hazard", feature, "duration")
    #os.system("pause - KM_survival_plot concluded")

    print("---------------survival_grouping Start---------------")
    #features = ['hsa-let-7a-5p', 'hsa-miR-99b-5p.T']
    features = ['hsa-miR-16-5p.T']
    features = ['hsa-miR-660-5p']
    features = ['hsa-miR-301a-3p']
    feature_train = X_train[features]
    feature_test = X_test[features]
    survival_grouping(y_train, y_test, feature_train = feature_train, feature_test = feature_test)

    print("---------------corr_p_val Start---------------")
    features = ['hsa-let-7a-5p', 'hsa-miR-99b-5p.T']
    features = ['hsa-miR-301a-3p', 'hsa-miR-181a-5p.gt']
    feature_train = X_train# [features]
    feature_test = X_test# [features]
    features_connections_1D(y_train, y_test, feature_train=feature_train, feature_test=feature_test)
    print("---------------dual correlated choice  Start---------------")
    train_file_name =os.path.join(globals.savefig_path , f'features_connections_1D_train.csv')
    test_file_name = os.path.join(globals.savefig_path , f'features_connections_1D_test.csv')

    top_correlated_features(train_file_name, test_file_name)

    print("---------------bye bye---------------")