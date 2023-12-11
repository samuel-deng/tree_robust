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

from train_utils import std_err

MODELS = [
    #'LogisticRegression',
    #'SVMClassifier',
    'DecisionTree2',
    'DecisionTree4',
    'DecisionTree8',
    'DecisionTree16',
    'DecisionTree',
    'RandomForest2',
    'RandomForest4',
    'RandomForest8',
    'RandomForest16',
    #'RandomForest',
    #'XGBoost',
    #'MLP'
    ]
PARAM_PATH = "params/"

def name_to_model(model_name, X_dim=None, params=None):
    """
    Takes a model name from the specified model names and outputs the sklearn model object initialized with appropriate hyperparameters.

    Args:
        default_params: initializes the model with the default sklearn parameters. If False, initialize with the parameters we found by cross-validating on each group.
    """
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
        model = DecisionTreeClassifier(**params)
    elif model_name == 'RandomForest4':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 4
        model = DecisionTreeClassifier(**params)
    elif model_name == 'RandomForest8':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 8
        model = DecisionTreeClassifier(**params)
    elif model_name == 'RandomForest16':
        params['criterion'] = 'log_loss'
        params['max_depth'] = 16
        model = DecisionTreeClassifier(**params)
    elif model_name == 'RandomForest':
        params['criterion'] = 'log_loss'
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
        H_train[g] = models[g].predict(X_train)
        H_test[g] = models[g].predict(X_test)
        H_train_err[g] = np.mean(H_train[g][groups_train[g]] != y_train[groups_train[g]])
        ng_test[g] = np.sum(groups_test[g])
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
        F_test_err[g] = np.mean(F_test[groups_test[g]] != y_test[groups_test[g]])
        if verbose:
            print('PREPEND group {0} ({4}): {1} (+/-{2}; n={3})'.format(g, F_test_err[g], std_err(F_test_err[g], ng_test[g]), ng_test[g], group_names[g]))

    return f, F_test_err

'''"
Simple tree data structure for MGL-Tree.

Each mglTree object is a node in the tree that keeps track of its parent and
children. 

data: a length-2 list of (group, predictor) that designates the group of the node and its associated predictor. upon initialization (before training), each associated predictor will be None. mglTreePredictor expects an mglTree where each node has None for the predictor.
children: a list of mglTree objects (each child is also an mglTree).
parent: an mglTree object. for the root, parent=None.
root: a Boolean value designating if the mglTree node is the root of the tree.
fitted: a Boolean value 
'''
class mglTree:
    def __init__(self, data, parent=None):
        if len(data) != 2:
            raise Exception("mglTree expects `data` to be a 2-tuple (group, predictor).")
        self.data = data
        # check if the node has a predictor associated to it already
        if data[1] is None:
            self.fitted = False
        else:
            self.fitted = True

        self.children = []
        self.parent = parent
        self.root = False
        self.leaf = False

    def add_child(self, obj):
        obj.parent = self
        self.children.append(obj)

    def is_root(self):
        return self.root
    
    def is_leaf(self):
        return self.leaf

    def is_fitted(self):
        return self.fitted

    def set_predictor(self, predictor):
        self.data[1] = predictor
        self.fitted = True

    def get_group(self):
        return self.data[0]

    def predict(self, X):
        if not self.fitted:
            raise Exception("Cannot predict with mglTree node: not yet fitted!")
        return self.data[1].predict(X)

    def get_predictor(self):
        return self.data[1]
    
def construct_tree(groups, intersections):
    """
    Helper function to create an mglTree out of the disjoint groups in groups and the intersections as the leaves. Only supports treees of 2 levels for now (a bit hacky).

    Args:
        groups: Disjoint group indices.
        intersections: List of tuples of group intersections.
    """
    tree = mglTree([(0, 0), None], parent=None)
    tree.root = True
    for g in groups:
        child_node = mglTree([(g, g), None])
        tree.add_child(child_node)
        for inter in intersections:
            if g in set(inter):
                inter_node = mglTree([inter, None])
                inter_node.leaf = True
                child_node.add_child(inter_node)
    return tree

def treepend(models, tree, X_train, y_train, groups_train, 
            X_test, y_test, groups_test, group_names,
            epsilon=0, verbose=False):
    """
    Runs the Treepend algorithm on a given mglTree and already fitted models in `models`. 
    """
    visited = []
    queue = []
    leaves = []
    num_groups = len(groups_train)
    assert(num_groups == len(models))
    assert(num_groups == len(groups_test))
    assert(num_groups == len(group_names))

    # visit nodes in bfs order
    visited.append(tree)
    queue.append(tree)
    while queue:
        # tree_node is a (group, predictor) tuple. 
        tree_node = queue.pop(0)
        g1, g2 = tree_node.get_group()
        X_g = X_train[np.array(groups_train[g1]) & np.array(groups_train[g2])]
        y_g = y_train[np.array(groups_train[g1]) & np.array(groups_train[g2])]

        if verbose:
            print("== group {} ({} examples) ==".format(g, len(X_g)))

        # calculate L(h | g) for current group
        hg_pred = models[g].predict(X_g)
        # err_hg = self.loss(hg_pred, y_g)
        err_hg = np.mean(hg_pred != y_g)
        if verbose:
            print("h{} error={}".format(g, err_hg))

        # calculate L(f^{pa(g)} | g) for current group, uses parent node
        if tree_node.root:  # default to ERM
            tree_node.set_predictor(models[g])
        else:
            if not tree_node.parent.is_fitted():
                raise Exception("Non-root nodes should all have fitted parents.")
            parg_pred = tree_node.parent.predict(X_g)
            err_parg = np.mean(parg_pred != y_g)
            if verbose:
                print("f^(pa){} error={}".format(g, err_parg))

            # decide whether to keep parent or update to hg
            diff_g = err_parg - err_hg - epsilon
            if diff_g > 0:
                tree_node.set_predictor(models[g])
                if verbose:
                    print("result for group {}: set to h{}.".format(g, g))
            else:
                tree_node.set_predictor(tree_node.parent.get_predictor())
                if verbose:
                    print("result for group {}: set to parent.".format(g))
            
            # in the disjoint case, we can collect the leaves
            if tree_node.is_leaf():
                leaves.append(tree_node)

        for child in tree_node.children:
            if child not in visited:
                visited.append(child)
                queue.append(child)

    # Evaluate the tree (leaves are assumed to be disjoint partitioning)
    # TODO: finish up evaluation code for Treepend
    test_err = {}
    for leaf in leaves:
        print(leaf.get_group())
    for g in range(num_groups):
        X_test[groups_test[g]]

    return tree, test_err