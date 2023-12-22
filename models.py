"""
Utility functions for handling all the model types used in the experiments.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mlp import MLPClassifier

import numpy as np
import os
import pickle
import multiprocessing

from train_utils import std_err

PARAM_PATH = "params/"

def name_to_model(model_name, X_dim=None, params=None):
    """
    Takes a model name from the specified model names and outputs the sklearn model object initialized with appropriate hyperparameters.

    Args:
        default_params: initializes the model with the default sklearn parameters. If False, initialize with the parameters we found by cross-validating on each group.
    """
    cores = multiprocessing.cpu_count()

    if params == None:
        params = {}
    if model_name == 'LogisticRegression':
        params['dual'] = False
        model = LogisticRegression(**params)
    elif model_name == 'LogisticRegressionSGD':
        params['loss'] = 'log_loss'
        params['penalty'] = None
        model = SGDClassifier(**params)
    elif model_name == 'SVMClassifier':
        params['dual'] = False
        model = LinearSVC(**params)
    elif model_name == 'DecisionTree2':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 2
        model = DecisionTreeClassifier(**params)
    elif model_name == 'DecisionTree4':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 4
        model = DecisionTreeClassifier(**params)
    elif model_name == 'DecisionTree8':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 8
        model = DecisionTreeClassifier(**params)
    elif model_name == 'DecisionTree16':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 16
        model = DecisionTreeClassifier(**params)
    elif model_name == 'DecisionTree':
        params['criterion'] = 'log_loss'
        model = DecisionTreeClassifier(**params)
    elif model_name == 'RandomForest2':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 2
        params['n_jobs'] = int(cores/2)
        model = RandomForestClassifier(**params)
    elif model_name == 'RandomForest4':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 4
        params['n_jobs'] = int(cores/2)
        model = RandomForestClassifier(**params)
    elif model_name == 'RandomForest8':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 8
        params['n_jobs'] = int(cores/2)
        model = RandomForestClassifier(**params)
    elif model_name == 'RandomForest16':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 16
        params['n_jobs'] = int(cores/2)
        model = RandomForestClassifier(**params)
    elif model_name == 'RandomForest':
        params['criterion'] = 'log_loss'
        params['n_jobs'] = int(cores/2)
        model = RandomForestClassifier(**params)
    elif model_name == 'XGBoost':
        params['nthread'] = int(cores/2)
        model = XGBClassifier(**params)
    elif model_name == 'MLP':
        if X_dim == None:
            raise ValueError("X_dim needed for MLP!")
        params['h_sizes'] = [X_dim, 256, 256] # 2 hidden layers
        model = MLPClassifier(**params)
    else:
        raise NotImplementedError("model: {} is not implemented!".format(model))
    
    return model

def load_group_params(model_name, dataset_name, g):
    """
    Loads group's param dictionary for model.

    Args:
        model_name: Name of the model to load the params for.
        dataset_name: Name of the dataset to load the params for.
        g: index of group to load the params for (0 is for ALL).
    """
    file = "{}_params.pkl".format(model_name)
    path = os.path.join(os.path.join(PARAM_PATH, dataset_name), file)
    with open(path, 'rb') as f:
        params = pickle.load(f)

    return params[g]

def prepend(models, X_train, y_train, groups_train, 
            X_test, y_test, groups_test, group_names,
            epsilon=0, verbose=False):
    """
    Runs the Prepend algorithm for already fitted models in `models`.

    Args:
        models: fitted sklearn-type models wth a .fit() and .predict()
        X_train: full training dataset
        y_train: full training labels
        groups_train: list of Boolean arrays for indexing X_train, y_train by group (there are num_groups of them)
        X_test: full test dataset
        y_test: full test labels
        groups_test: list of Boolean arrays for indexing X_test, y_test by group
        group_names: name of each group
        epsilon: tolerance for prepending a new predictor
    """
    # f is a list of indices (0 is the ALL predictor)
    f = [0]
    num_groups = len(groups_train)
    assert(num_groups == len(models))
    assert(num_groups == len(groups_test))
    assert(num_groups == len(group_names))

    H_train = {} # predictions of group-wise models on training data
    H_test = {}  # predictions of group-wise models on test data
    H_train_err = {}
    ng_test = {}
    for g in range(num_groups):
        if models[g]:
            H_train[g] = models[g].predict(X_train)
            H_test[g] = models[g].predict(X_test)
            H_train_err[g] = np.mean(H_train[g][groups_train[g]] != y_train[groups_train[g]])
            ng_test[g] = np.sum(groups_test[g])
        else:
            H_train_err[g] = np.inf
    F_train = H_train[0].copy()
    F_test = H_test[0].copy()
    F_train_err = {}
    for g in range(num_groups):
        F_train_err[g] = np.mean(F_train[groups_train[g]] != y_train[groups_train[g]])
    while True:
        scores = [H_train_err[g] + epsilon - F_train_err[g] for g in range(num_groups)]
        g = np.argmin(scores)
        if scores[g] < 0.:
            f.insert(0,g) # prepend g to the list f
            F_train[groups_train[g]] = H_train[g][groups_train[g]]
            F_test[groups_test[g]] = H_test[g][groups_test[g]]
            for g in range(num_groups):
                F_train_err[g] = np.mean(F_train[groups_train[g]] != y_train[groups_train[g]])
        else:
            break

    F_test_err = {}
    for g in range(num_groups):
        if models[g]:
            F_test_err[g] = np.mean(F_test[groups_test[g]] != y_test[groups_test[g]])
            if verbose:
                print('PREPEND group {0} ({4}): {1} (+/-{2}; n={3})'.format(g, F_test_err[g], std_err(F_test_err[g], ng_test[g]), ng_test[g], group_names[g]))
        elif verbose:
            print("PREPEND group {} had no data!".format(g))

    return f, F_test_err

def treepend(models, tree, X_train, y_train, gps_train,
             X_test, y_test, gps_test, group_names,
             epsilon=0, verbose=False):
    """
    Runs the MGL-Tree algorithm for already fitted models in `model`.

    Args:
        models: fitted sklearn-type models with a .fit() and a .predict()
        tree: a list of lists designating which groups in gps_train and gps_test are in each level of the tree.
        X_train: full training dataset
        y_train: full training labels
        gps_train: list of Boolean arrays for indexing X_train, y_train by group.
        X_test: full test dataset
        y_test: full test labels
        gps_test: list of Boolean arrays for indexing X_test, y_test by group
        group_names: name for each group
        epsilon: tolerance for new predictor
    """
    declist = [0]
    dectree = [[0] * len(level) for level in tree]

    num_groups = len(gps_train)
    assert(num_groups == len(models))
    assert(num_groups == len(gps_test))
    assert(num_groups == len(group_names))

    H_train = {}     # predictions of group-wise models on training data
    H_test = {}      # predictions of group-wise models on test data
    H_train_err = {} # number of groups in test 
    ng_test = {}     # number of samples in test for a group

    # Get predictions for every model on the train and test set
    for g in range(num_groups):
        if models[g]:   # Possible that a group is empty
            H_train[g] = models[g].predict(X_train)
            H_test[g] = models[g].predict(X_test)
            diff = H_train[g][gps_train[g]] != y_train[gps_train[g]]
            H_train_err[g] = np.mean(diff)
            ng_test[g] = np.sum(gps_test[g])
        else:
            H_train_err[g] = np.inf
    
    # Initialize predictions for the tree predictor
    F_train = H_train[0].copy()
    F_test = H_test[0].copy()
    F_train_err = {}
    for g in range(num_groups):
        diff = F_train[gps_train[g]] != y_train[gps_train[g]]
        F_train_err[g] = np.mean(diff)

    # BFS through the tree
    for i, level in enumerate(tree):
        for j, g in enumerate(level):
            if H_train_err[g] < F_train_err[g] + epsilon:
                declist.insert(0, g)
                dectree[i][j] = g
                F_train[gps_train[g]] = H_train[g][gps_train[g]]
                F_test[gps_test[g]] = H_test[g][gps_test[g]]
                for g in range(num_groups):
                    diff = F_train[gps_train[g]] != y_train[gps_train[g]]
                    F_train_err[g] = np.mean(diff)

    # Find test error for each group
    F_test_err = {}
    for g in range(num_groups):
        if models[g]:
            diff = F_test[gps_test[g]] != y_test[gps_test[g]]
            F_test_err[g] = np.mean(diff)
            if verbose:
                print('TREE group {0} ({4}): {1} (+/-{2}; n={3})'.format(
                    g, F_test_err[g], std_err(F_test_err[g], ng_test[g]), ng_test[g], group_names[g]))
            elif verbose:
                print("TREE group {} had no data!".format(g))

    return declist, dectree, F_test_err