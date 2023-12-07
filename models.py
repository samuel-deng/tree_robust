"""
Utility functions for handling all the model types used in the experiments.
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mlp import MLPClassifier

MODELS = [
    'LogisticRegression',
    'SVMClassifier',
    'DecisionTree2',
    'DecisionTree4',
    'DecisionTree',
    'RandomForest',
    'XGBoost',
    #'MLP'
    ]

def name_to_model(model_name, X_dim=None, params=None):
    """
    Takes a model name from the specified model names and outputs the sklearn model object initialized with appropriate hyperparameters.

    Args:
        default_params: initializes the model with the default sklearn parameters. If False, initialize with the parameters we found by cross-validating on each group.
    """
    if params == None:
        params = {}

    if model_name == 'LogisticRegression':
        model = LogisticRegression(**params)
    elif model_name == 'LogisticRegressionSGD':
        params['loss'] = 'log_loss'
        params['penalty'] = None
        model = SGDClassifier(**params)
    elif model_name == 'SVMClassifier':
        model = LinearSVC(**params)
    elif model_name == 'DecisionTree2':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 2
        model = DecisionTreeClassifier(**params)
    elif model_name == 'DecisionTree4':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 4
        model = DecisionTreeClassifier(**params)
    elif model_name == 'DecisionTree':
        params['criterion'] = 'log_loss'
        model = DecisionTreeClassifier(**params)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(**params)
    elif model_name == 'XGBoost':
        model = XGBClassifier(**params)
    elif model_name == 'MLP':
        if X_dim == None:
            raise ValueError("X_dim needed for MLP!")
        params['h_sizes'] = [X_dim, 256, 256] # 2 hidden layers
        model = MLPClassifier(**params)
    else:
        raise NotImplementedError("model: {} is not implemented!".format(model))
    
    return model