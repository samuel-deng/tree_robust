"""
All utilites and driver function for running the Group Errors experiments.
"""
import warnings
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from models import name_to_model, load_group_params, prepend, treepend
from train_utils import std_err

warnings.simplefilter("ignore", category=ConvergenceWarning)
def run_errors(args, dataset, models):
    """
    Main driver function for running all error experiments for a single dataset and a collection of models.

    Args:
        args: arguments from main() function.
        dataset: Dataset object including X, y, groups, intersections.
        model_names: specified models (refer to MODELS for names).
    """
    results = {}

    # Preprocess data and split into train/test
    X = dataset.X
    y = dataset.y
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    splits = train_test_split(*tuple([X, y] + dataset.groups), 
                              test_size=0.2, random_state=args.random_state)
    X_train = splits[0]
    X_test = splits[1]
    y_train = splits[2]
    y_test = splits[3]
    groups_train = splits[4::2]
    groups_test = splits[5::2]

    X_val = splits[1]
    y_val = splits[3]
    groups_val = splits[5::2]

    for model_name in models:
        # Global ERM and group-wise ERMs
        print("Fitting model={}...".format(model_name))

        group_models = []
        for g, group_name in enumerate(dataset.group_names):
            n_g = np.sum(groups_train[g])
            print("\tOn group={} with n={}...".format(group_name, n_g))

            # Train model on group g
            if args.group_params:
                params = load_group_params(model_name, dataset.name, g)
            else:
                params = None

            # Vary number of epochs per dataset, per Gardner et al. paper.
            # Default number of epochs = 50
            if model_name == 'MLP' and dataset.name == 'adult':
                params = {}
                params['n_epochs'] = 300
            elif model_name == 'MLP' and dataset.name == 'compas':
                params = {}
                params['n_epochs'] = 300
            elif model_name == 'MLP' and dataset.name == 'communities':
                params = {}
                params['n_epochs'] = 100
            elif model_name == 'MLP' and dataset.name == 'german':
                params = {}
                params['n_epochs'] = 50
            model = name_to_model(model_name, X.shape[1], params=params)
            
            if np.sum(groups_train[g]) > 0:
                model.fit(X_train[groups_train[g]], y_train[groups_train[g]])
            else:
                model = None
            group_models.append(model)

        # Prepend
        print("Fitting PREPEND model={}...".format(model_name))
        prepend_results = prepend(group_models, X_val, y_val, groups_val, 
                                  X_test, y_test, groups_test, 
                                  dataset.group_names)
        declist = prepend_results[0]
        declist_errs = prepend_results[1]
        print("\tResulting decision list: {}".format(declist))

        # Treepend
        '''
        print("Fitting TREEPEND model={}...".format(model_name))
        for tree in dataset.trees:
            treepend_results = prepend(group_models, X_test, y_test,
                                       groups_test, X_test, y_test, groups_test, dataset.group_names)
            mgltree = treepend_results[0]
            tree_errs = treepend_results[1]
        '''
            
        # Evaluate on each group
        results[model_name] = {}
        for g, group_name in enumerate(dataset.group_names):
            if np.sum(groups_test[g]) == 0 or group_models[g] is None:
                print("\n=== Error on G{}: {} ===\n".format(g, group_name))
                print("G{} ({}) has no data!".format(g, group_name))
                results[model_name][g] = {}
                results[model_name][g]['ERM_ALL'] = (-1, 0)
                results[model_name][g]['ERM_GROUP'] = (-1, 0)
                results[model_name][g]['PREPEND'] = (-1, 0)
            else:
                print("\n=== Error on G{}: {} ===\n".format(g, group_name))
                y_g = y_test[groups_test[g]]
                erm_pred = group_models[0].predict(X_test[groups_test[g]])
                n_g = np.sum(groups_test[g])
                g_erm_pred = group_models[g].predict(X_test[groups_test[g]])
                
                erm_err = np.mean(y_g != erm_pred)
                erm_std = std_err(n_g, erm_err)
                g_erm_err = np.mean(y_g != g_erm_pred)
                g_erm_std = std_err(n_g, g_erm_err)
                prepend_err = declist_errs[g]
                prepend_std = std_err(n_g, prepend_err)
                #treepend_err = tree_errs[g]
                #treepend_std = std_err(n_g, treepend_err)

                print("\tGlobal ERM = {} +/- {}".format(erm_err, erm_std))
                print("\tGroup ERM = {} +/- {}".format(g_erm_err, g_erm_std))
                print("\tPrepend = {} +/- {}".format(prepend_err, prepend_std))
                #print("\tTreepend = {}".format(treepend_err, treepend_std))
                
                results[model_name][g] = {}
                results[model_name][g]['ERM_ALL'] = (erm_err, erm_std)
                results[model_name][g]['ERM_GROUP'] = (g_erm_err, g_erm_std)
                results[model_name][g]['PREPEND'] = (prepend_err, prepend_std)
                #results[model_name][g]['TREEPEND'] = (treepend_err, treepend_std)

    return results