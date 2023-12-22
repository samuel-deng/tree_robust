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

    # Preprocess data and split into train/test/val
    X = dataset.X
    y = dataset.y
    tree = dataset.tree
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

    splits = train_test_split(*tuple([X_train, y_train] + groups_train),
                            test_size=0.5, random_state=args.random_state)
    X_train = splits[0]
    X_val = splits[1]
    y_train = splits[2]
    y_val = splits[3]
    groups_train = splits[4::2]
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
            
            if np.sum(groups_train[g]) > 0 and len(np.unique(y_train[groups_train[g]])) > 1:
                model.fit(X_train[groups_train[g]], y_train[groups_train[g]])
            else:
                model = None
            group_models.append(model)

        # Prepend
        print("Fitting PREPEND model={}...".format(model_name))
        prepend_results = prepend(group_models, X_train, y_train, groups_train, 
                                  X_test, y_test, groups_test, 
                                  dataset.group_names)
        declist = prepend_results[0]
        declist_errs = prepend_results[1]
        print("\tResulting decision list: {}".format(declist))

        # Treepend
        print("Fitting TREE model...")
        treepend_results = treepend(group_models, tree, X_val, y_val,
                                    groups_val, X_test, y_test, groups_test, dataset.group_names)
        tree_declist = treepend_results[0]
        dectree = treepend_results[1]
        tree_errs = treepend_results[2]
        print("\tResulting decision list: {}".format(tree_declist))
            
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
                yg_train = y_train[groups_train[g]]
                yg = y_test[groups_test[g]]
                erm_pred_tr = group_models[0].predict(X_train[groups_train[g]])
                erm_pred = group_models[0].predict(X_test[groups_test[g]])
                ng_train = np.sum(groups_train[g])
                ng = np.sum(groups_test[g])
                g_erm_predtr = group_models[g].predict(X_train[groups_train[g]])
                g_erm_pred = group_models[g].predict(X_test[groups_test[g]])

                # Training errors
                erm_err_tr = np.mean(yg_train != erm_pred_tr)
                erm_std_tr = std_err(ng_train, erm_err_tr)
                g_erm_err_tr = np.mean(yg_train != g_erm_predtr)
                g_erm_std_tr = std_err(ng_train, g_erm_err_tr)
                
                # Test errors
                erm_err = np.mean(yg != erm_pred)
                erm_std = std_err(ng, erm_err)
                g_erm_err = np.mean(yg != g_erm_pred)
                g_erm_std = std_err(ng, g_erm_err)
                prepend_err = declist_errs[g]
                prepend_std = std_err(ng, prepend_err)
                treepend_err = tree_errs[g]
                treepend_std = std_err(ng, treepend_err)

                print("\tGlobal ERM = {} +/- {}".format(erm_err, erm_std))
                print("\tGroup ERM = {} +/- {}".format(g_erm_err, g_erm_std))
                print("\tPrepend = {} +/- {}".format(prepend_err, prepend_std))
                print("\tTree = {} +/- {}".format(treepend_err, treepend_std))
                
                # Save all test error
                results[model_name][g] = {}
                results[model_name][g]['ERM_ALL'] = (erm_err, erm_std)
                results[model_name][g]['ERM_GROUP'] = (g_erm_err, g_erm_std)
                results[model_name][g]['PREPEND'] = (prepend_err, prepend_std)
                results[model_name][g]['TREE'] = (treepend_err, treepend_std)
                
                # Save training error as well
                results[model_name][g]['ERM_ALL_TRAIN'] = (erm_err_tr,
                                                           erm_std_tr)
                results[model_name][g]['ERM_GROUP_TRAIN'] = (g_erm_err_tr,
                                                             g_erm_std_tr)

                # Prepend/Treepnd data also gets saved
                results[model_name][g]['PREPEND_LIST'] = declist
                results[model_name][g]['TREE_LIST'] = tree_declist
                results[model_name][g]['TREE_TREE'] = dectree

    return results