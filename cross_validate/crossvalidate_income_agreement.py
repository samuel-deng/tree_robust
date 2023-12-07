import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from data import preprocess_income
from train_utils import cross_validate_pergroup

# Preprocess Adult Dataset (folktables)
X, y, col_transf, _, group_memberships = preprocess_income()
num_groups = len(group_memberships)

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
        best_params_pergroup = cross_validate_pergroup(X, y, col_transf, 
                                                    group_memberships, num_groups, 
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
        best_params_pergroup = cross_validate_pergroup(X, y, col_transf, 
                                                    group_memberships, num_groups, 
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
        best_params_pergroup = cross_validate_pergroup(X, y, col_transf, 
                                                    group_memberships, num_groups, 
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
        best_params_pergroup = cross_validate_pergroup(X, y, col_transf, 
                                                        group_memberships, num_groups, 
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
        best_params_pergroup = cross_validate_pergroup(X, y, col_transf, 
                                                        group_memberships, num_groups, 
                                                        RandomForestClassifier, param_grid)
        with open(best_params_path, 'wb') as handle:
            pickle.dump(best_params_pergroup, handle, protocol=pickle.HIGHEST_PROTOCOL)