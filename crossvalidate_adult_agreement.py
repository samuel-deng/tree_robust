import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle
import time

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Adult Dataset
adult_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country", 
                "income"]
adult_data = pd.read_csv("./datasets/adult/adult.data", header=None, names=adult_names, na_values=' ?')
adult_data = adult_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
adult_test = pd.read_csv("./datasets/adult/adult.test", header=None, names=adult_names, na_values=' ?')
adult_test = adult_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
dfs = [adult_data, adult_test]
adult_df = pd.concat(dfs)
adult_df = adult_df.dropna()
adult_df = adult_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
print("Adult Shape: {}".format(adult_df.shape))

# last column in adult has some textual discrepancy
adult_df = adult_df.replace(">50K.", ">50K")
adult_df = adult_df.replace("<=50K.", "<=50K")

# Split into X and y
X, y = adult_df.drop("income", axis=1), adult_df["income"]

# Select categorical and numerical features
cat_idx = X.select_dtypes(include=["object", "bool"]).columns
num_idx = X.select_dtypes(include=['int64', 'float64']).columns
steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', MinMaxScaler(), num_idx)]
col_transf = ColumnTransformer(steps)

# label encoder to target variable so we have classes 0 and 1
assert(len(np.unique(y)) == 2)
y = LabelEncoder().fit_transform(y)
print("% examples >=50k (y=1): {}".format(100 * len(np.where(y == 1)[0])/len(y)))
print("% examples <50k (y=0): {}".format(100 * len(np.where(y == 0)[0])/len(y)))

def adult_gp_indices(df, race_val, sex_val):
    if race_val == "NotWhite":
        return np.where((df['race'] != 'White') & (df['sex'] == sex_val))
    else:
        return np.where((df['race'] == race_val) & (df['sex'] == sex_val))

group_names = ["ALL", "W,M", "W,F", "nW,M", "nW,F", "W", "nW", "M", "F"]
group_memberships = []
group_memberships.append([True] * y.shape[0])
race_gps_coarse = ["White", "NotWhite"]
sex_gps = ["Male", "Female"]

# Traditional disjoint groups
for race in race_gps_coarse:
    for sex in sex_gps:
        indices = adult_gp_indices(X, race, sex)[0]
        membership = np.zeros(y.shape[0], dtype=bool)
        membership[indices] = True
        group_memberships.append(membership)

# Add 4 overlapping groups
w_indices = np.where(X['race'] == 'White')
w_membership = np.zeros(y.shape[0], dtype=bool)
w_membership[w_indices] = True
group_memberships.append(w_membership)

nw_indices = np.where(X['race'] != 'White')
nw_membership = np.zeros(y.shape[0], dtype=bool)
nw_membership[nw_indices] = True
group_memberships.append(nw_membership)

m_indices = np.where(X['sex'] == 'Male')
m_membership = np.zeros(y.shape[0], dtype=bool)
m_membership[m_indices] = True
group_memberships.append(m_membership)

f_indices = np.where(X['sex'] == 'Female')
f_membership = np.zeros(y.shape[0], dtype=bool)
f_membership[f_indices] = True
group_memberships.append(f_membership)

num_groups = len(group_memberships)
print('num_groups = {0}'.format(num_groups))

# Fit the ColumnTransformer to X
X_transf = col_transf.fit_transform(X)
print("Column-transformed X has shape: {}".format(X_transf.shape))

# Train-test split
splits = train_test_split(*tuple([X, y] + group_memberships), test_size=0.2, random_state=0)
X_train = splits[0]
X_test = splits[1]
y_train = splits[2]
y_test = splits[3]

# group_train
group_train = splits[4::2]
group_test = splits[5::2]

# group_train and group_test have the indices in X_train, X_test (respectively)
# for each group, as a binary mask.
num_group_train = {}
num_group_test = {}

# TODO: fix column alignment issue :(
print('Group\t\t\ttrain\ttest')
for g in range(num_groups):
    num_group_train[g] = np.sum(group_train[g])
    num_group_test[g] = np.sum(group_test[g])
    print('{0} ({3})\t\t\t{1}\t{2}'.format(g, num_group_train[g], num_group_test[g], group_names[g]))

for i in range(num_groups):
    print('P(Y=1 | group {0}) = {1}'.format(i, np.mean(y_test[group_test[i]])))

import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from train_utils import cross_validate_pergroup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='crossvalidate_adult_agreement',
                        description='Cross-validator for DecisionTree, Gradient-Boosted Trees, XGBoost.')
    parser.add_argument('--dec_tree', help='cross-validate DecisionTreeClassifier.', action='store_true')
    parser.add_argument('--gbm', help='cross-validate GradientBoostingClassifier.', action='store_true')
    parser.add_argument('--xgb', help='cross-validate XGBoostClassifier', action='store_true')
    parser.add_argument('--ada', help='cross-validate AdaBoost', action='store_true')
    parser.add_argument('--rf', help='cross-validate RandomForest', action='store_true')
    args = parser.parse_args()
    SAVE_DATA_PATH = 'adult_agreement_data/'

    '''
    CROSS-VALIDATION FOR DECISION TREES
    Cross-validate the best decision tree per group.
    '''
    if args.dec_tree:
        best_params_path = os.path.join(SAVE_DATA_PATH, 'dectree_params.pkl')

        param_grid = {
            'max_depth': [2, 4, 6, 8, 10, None],
            'min_samples_split': [2, 4, 8, 16, 32],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32],
            'ccp_alpha': [0, 0.01, 0.1, 1.0],
            'random_state': [0]
        }
        best_params_pergroup = cross_validate_pergroup(X_train, y_train, col_transf, 
                                                    group_train, num_groups, 
                                                    DecisionTreeClassifier, param_grid)
        with open(best_params_path, 'wb') as handle:
            pickle.dump(best_params_pergroup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    CROSS-VALIDATION FOR GRADIENT BOOSTING CLASSIFIER
    Cross-validate the best gradient-boosted trees per group.
    '''
    if args.gbm:
        best_params_path = os.path.join(SAVE_DATA_PATH, 'gbm_params.pkl')

        param_grid = {
            'max_depth': [2, 4, 8, 16],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'random_state': [0]
        }
        best_params_pergroup = cross_validate_pergroup(X_train, y_train, col_transf, 
                                                    group_train, num_groups, 
                                                    GradientBoostingClassifier, param_grid)
        with open(best_params_path, 'wb') as handle:
            pickle.dump(best_params_pergroup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    CROSS-VALIDATION FOR GRADIENT BOOSTING CLASSIFIER
    Cross-validate the best XGBoost classifier per group.
    '''
    if args.xgb:
        best_params_path = os.path.join(SAVE_DATA_PATH, 'xgb_params.pkl')

        param_grid = {
            'max_depth': [2, 4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'random_state': [0]
        }
        best_params_pergroup = cross_validate_pergroup(X_train, y_train, col_transf, 
                                                    group_train, num_groups, 
                                                    XGBClassifier, param_grid)
        with open(best_params_path, 'wb') as handle:
            pickle.dump(best_params_pergroup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    CROSS-VALIDATION FOR ADABOOST ClASSIFIER
    Cross-validate the best AdaBoost classifier per group.
    '''
    if args.ada:
        best_params_path = os.path.join(SAVE_DATA_PATH, 'ada_params.pkl')

        param_grid = {
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'random_state': [0]
        }
        best_params_pergroup = cross_validate_pergroup(X_train, y_train, col_transf, 
                                                        group_train, num_groups, 
                                                        AdaBoostClassifier, param_grid)
        with open(best_params_path, 'wb') as handle:
            pickle.dump(best_params_pergroup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    CROSS-VALIDATION FOR RANDOM FOREST CLASSIFIER
    Cross-validate the best RandomForestClassifier per group.
    '''
    if args.rf:
        best_params_path = os.path.join(SAVE_DATA_PATH, 'rf_params.pkl')

        param_grid = {
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'ccp_alpha': [0, 0.001, 0.01, 0.1],
            'random_state': [0]
        }
        best_params_pergroup = cross_validate_pergroup(X_train, y_train, col_transf, 
                                                        group_train, num_groups, 
                                                        RandomForestClassifier, param_grid)
        with open(best_params_path, 'wb') as handle:
            pickle.dump(best_params_pergroup, handle, protocol=pickle.HIGHEST_PROTOCOL)
