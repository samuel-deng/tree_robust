"""
All utilities and driver function for running the Group Agreement experiments.
"""
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import time
import warnings

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from models import name_to_model

warnings.simplefilter("ignore", category=ConvergenceWarning)

def eval_agreement(models, X):
    """
    Evaluates the agreement of the binary prediction of multiple models on some data. If len(models) > 2, then this is calculated through elementwise AND through all the predictions.

    Args:
        models: models to evaluate the agreement over.
        X: data to evaluate the agreement over
    """
    if len(models) < 2:
        raise ValueError("Cannot calculate agreement with less than 2 models!")
    agreement = (models[0].predict(X) == models[1].predict(X)).astype(int)
    for i in range(2, len(models)):
        agreement = (agreement == models[i].predict(X)).astype(int)
    return np.sum(agreement)/len(agreement)

def agreement_trial(X, y, dataset, model_name, params=False):
    """
    Gets agreements for each group intersection and for ERM and each group.

    Args:
        X: features for the data to fit the models on
        y: binary labels for the data
        dataset: Dataset object for the groups and group names
        model_name: model to fit
        params: Boolean value to fit group-wise cross-validated parameters
    """
    splits = resample(*tuple([X, y] + dataset.groups), n_samples=X.shape[0])
    X = splits[0]
    y = splits[1]
    groups = splits[2:]
    
    intersect_results = []
    group_results = []

    # Fit each model to each group
    fitted_models = []
    for group, group_name in zip(groups, dataset.group_names):
        if params:
            model = name_to_model(model_name, X_dim=X.shape[1])
        else:
            model = name_to_model(model_name, X_dim=X.shape[1])
        fitted_models.append(model.fit(X[group], y[group]))

    # Get agreements for each group intersection
    for ((g1, g2), name) in zip(dataset.intersections, dataset.inter_names):
        indices = groups[g1] & groups[g2]
        models = [fitted_models[g1], fitted_models[g2]]
        intersect_results.append(eval_agreement(models, X[indices]))

    # Get agreements for the ERM model with each group
    for g in range(1, len(groups)):
        models = [fitted_models[0], fitted_models[g]]
        group_results.append(eval_agreement(models, X[groups[g]]))

    return intersect_results, group_results

def print_agreement(results, model_names):
    """
    Prints agreements for a model in a readable format.

    Args:
        results: the results dictionary from run_agreement. Keys are model names and values are dictionaries.
        model_names: the model names to print the reuslts for.
    """
    for model_name in model_names:
        print("Agreement stats for model={}:".format(model_name))
        for intersection, (agree, err) in results[model_name].items():
            print("\t{}: {:.2f} +/- {:.2f}".format(intersection, agree, err))

def run_agreement(args, dataset, model_names):
    """
    Main driver function for running all the agreement experiments on a single dataset for the specified models.

    Args:
        args: arguments from main() function.
        dataset: Dataset object including X, y, groups, intersections.
        model_names: specified models (refer to MODELS for names).
    """
    results = {}

    X = dataset.X
    y = dataset.y
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    for model_name in model_names:
        start = time.time()
        print("Running {} bootstraps for model={} agreement...".format(args.bootstraps, model_name))
        results[model_name] = {}

        # Run args.bootstraps number of agreement trials
        model_results = Parallel(n_jobs=-1)(delayed(agreement_trial)(
            X, y, dataset, model_name, args.group_params) for _ in range(int(args.bootstraps)))
        
        # Unwrap all the results from the bootstraps
        inter_results_all = [[] for _ in range(len(dataset.inter_names))]
        gp_results_all = [[] for _ in range(len(dataset.groups) - 1)]
        for i in range(int(args.bootstraps)):
            inter_results, gp_results = model_results[i]
            for j in range(len(dataset.inter_names)):
                inter_results_all[j].append(inter_results[j])
            for j in range(len(dataset.groups) - 1): # Excludes the ALL group
                gp_results_all[j].append(gp_results[j])

        # Get the mean and standard error for each intersection
        for i, intersect in enumerate(dataset.inter_names):
            agreement = np.mean(inter_results_all[i])
            std_err = np.std(inter_results_all[i])
            results[model_name][intersect] = (agreement, std_err)

        # Get the mean and standard error for each group and global ERM
        for i, group in enumerate(dataset.group_names[1:]):
            intersect = "(ALL, {})".format(group)
            agreement = np.mean(gp_results_all[i])
            std_err = np.std(gp_results_all[i])
            results[model_name][intersect] = (agreement, std_err)

        end = time.time()
        print("Elapsed time for {} = {} seconds.".format(model_name, end-start))

    return results