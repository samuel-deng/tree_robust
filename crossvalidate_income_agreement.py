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

from folktables import ACSDataSource, ACSIncome

# Preprocess Employment Dataset (folktables)

# Download data and define groups
data_source = ACSDataSource(survey_year='2016', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSIncome.df_to_numpy(acs_data)
sex = features[:, -2]
old = (features[:,0] > 65)
print("ACS Income Features: {}".format(ACSIncome.features))
print("ACS Income Shape {}".format(features.shape))

# Define groups 
group_names = []
group_memberships = []
group_memberships.append([True] * label.shape[0])
group_names.append('ALL')
for g in np.unique(group):
    if g == 4 or g == 5: # group is too small
        continue
    group_memberships.append(group == g)
    group_names.append('R{0}'.format(g))
group_memberships.append(sex == 1)
group_names.append('S1')
group_memberships.append(sex == 2)
group_names.append('S2')
group_memberships.append(old == False)
group_names.append('A1')
group_memberships.append(old == True)
group_names.append('A2')
num_groups = len(group_memberships)
print('num_groups = {0}'.format(num_groups))

to_one_hot = set(['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P'])
to_leave_alone = set(ACSIncome.features) - to_one_hot
one_hot_inds = [i for i, x in enumerate(ACSIncome.features) if x in to_one_hot]
leave_alone_inds = [i for i, x in enumerate(ACSIncome.features) if x in to_leave_alone]

steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', MinMaxScaler(), leave_alone_inds)]
col_transf = ColumnTransformer(steps)
features_t = col_transf.fit_transform(features).toarray()
print("Column-transformed X has shape: {}".format(features_t.shape))

# Train-test split
splits = train_test_split(*tuple([features, label] + group_memberships), test_size=0.2, random_state=0)
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

print('Group\ttrain\ttest')
for g in range(num_groups):
    num_group_train[g] = np.sum(group_train[g])
    num_group_test[g] = np.sum(group_test[g])
    print('{0} ({3})\t{1}\t{2}'.format(g, num_group_train[g], num_group_test[g], group_names[g]))

for i in range(num_groups):
    print('P(Y=1 | group {0}) = {1}'.format(i, np.mean(y_test[group_test[i]])))



import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from train_utils import cross_validate_pergroup
if __name__ == "__main__":
    SAVE_DATA_PATH = 'income_agreement_data/'
    parser = argparse.ArgumentParser(
                        prog='crossvalidate_income_agreement',
                        description='Cross-validator for DecisionTree, Gradient-Boosted Trees, XGBoost.')
    parser.add_argument('--dt', help='cross-validate DecisionTreeClassifier.', action='store_true')
    parser.add_argument('--gbm', help='cross-validate GradientBoostingClassifier.', action='store_true')
    parser.add_argument('--xgb', help='cross-validate XGBoostClassifier', action='store_true')
    parser.add_argument('--ada', help='cross-validate AdaBoost', action='store_true')
    parser.add_argument('--rf', help='cross-validate RandomForest', action='store_true')
    args = parser.parse_args()

    '''
    CROSS-VALIDATION FOR DECISION TREES
    Cross-validate the best decision tree per group.
    '''
    if args.dt:
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