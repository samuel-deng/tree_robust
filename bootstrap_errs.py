from data import preprocess_adult, preprocess_communities,preprocess_compas, preprocess_german, preprocess_employment, preprocess_income
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import pickle
import os
import time
from experiment_utils import error_trials

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import numpy as np
import argparse

# Parse which dataset we're using
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('n_bootstraps')
parser.add_argument('-s', '--save',
                    action='store_true')  # on/off flag
args = parser.parse_args()
dataset = args.dataset
n_bootstraps = int(args.n_bootstraps)
save_errs = args.save

# Preprocess the data
if dataset == 'adult':
    X, y, col_transf, group_memberships = preprocess_adult()
elif dataset == 'communities':
    X, y, col_transf, group_memberships = preprocess_communities()
elif dataset == 'compas':
    X, y, col_transf, group_memberships = preprocess_compas()
elif dataset == 'german':
    X, y, col_transf, group_memberships = preprocess_german()
elif dataset == 'employment':
    X, y, col_transf, _, group_memberships = preprocess_employment()
elif dataset == 'income':
    X, y, col_transf, _, group_memberships = preprocess_income()
else:
    raise ValueError("dataset not supported. check input to program.")
X = col_transf.transform(X)
num_groups = len(group_memberships)

# Get parameters for each model
SAVE_DATA_PATH = '{}_agreement_data/'.format(dataset)
with open(os.path.join(SAVE_DATA_PATH, 'dectree_params.pkl'), 'rb') as f:
    dt_params = pickle.load(f)
with open(os.path.join(SAVE_DATA_PATH, 'rf_params.pkl'), 'rb') as f:
    rf_params = pickle.load(f)
with open(os.path.join(SAVE_DATA_PATH, 'gbm_params.pkl'), 'rb') as f:
    gb_params = pickle.load(f)
with open(os.path.join(SAVE_DATA_PATH, 'xgb_params.pkl'), 'rb') as f:
    xgb_params = pickle.load(f)

# Run bootstraps in parallel
start_time = time.time()
print("Number of CPUs: {}".format(cpu_count()))
cpus = cpu_count()/2
if dataset == 'employment' or dataset == 'income':
    results = error_trials(n_bootstraps, X, y, group_memberships, 
                        dt_params, rf_params, gb_params, xgb_params, bootstrap=False)
else:
    results = error_trials(n_bootstraps, X, y, group_memberships, 
                        dt_params, rf_params, gb_params, xgb_params)
dt_errs_stats = results[0]
dt_erm_errs_stats = results[1]
rf_errs_stats = results[2]
gb_errs_stats = results[3]
xgb_errs_stats = results[4]
end_time = time.time()
print("Total time elapsed for {} bootstraps= {} seconds".format(n_bootstraps,
                                                                end_time - start_time))

# Save to pickle
if save_errs:
    ERRS_PATH = 'final_errs/'
    dt_errs_path = os.path.join(ERRS_PATH, '{}_dt_errs.pkl'.format(dataset))
    with open(dt_errs_path, 'wb') as handle:
            pickle.dump(dt_errs_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dt_erm_errs_path = os.path.join(ERRS_PATH, '{}_dt_erm_errs.pkl'.
                                    format(dataset))
    with open(dt_erm_errs_path, 'wb') as handle:
            pickle.dump(dt_erm_errs_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    rf_errs_path = os.path.join(ERRS_PATH, '{}_rf_errs.pkl'.format(dataset))
    with open(rf_errs_path, 'wb') as handle:
            pickle.dump(rf_errs_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gb_errs_path = os.path.join(ERRS_PATH, '{}_gb_errs.pkl'.format(dataset))
    with open(gb_errs_path, 'wb') as handle:
            pickle.dump(gb_errs_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    xgb_errs_path = os.path.join(ERRS_PATH, '{}_xgb_errs.pkl'.format(dataset))
    with open(xgb_errs_path, 'wb') as handle:
            pickle.dump(xgb_errs_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)