#!/user/sd3013/.conda/envs/agreement/bin/python
"""
Main function for running and evaluating (1) group agreement (2) group errors on each dataset and each model class.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import json
import numpy as np
import torch

from agreement import run_agreement, print_agreement
from errors import run_errors

from data import name_to_dataset

RESULTS_PATH = './results'

# List of base models to use. Specific ones to use are passed as an argument.
MODELS = [
    'LogisticRegression',
    'SVMClassifier',
    'DecisionTree2',
    'DecisionTree4',
    'DecisionTree',
    'RandomForest',
    'XGBoost',
    'MLP'
    ]
# LogisticRegressionSGD

DATASETS = [
    #'adult',
    #'communities',
    #'compas',
    #'german',
    #'incomeCA',
    'incomeNY',
    'incomeTX',
    'employmentCA',
    'employmentNY',
    'employmentTX',
    'coverageCA',
    'coverageNY',
    'coverageTX'
    ]

def save_results(args, datasets, models, results):
    """
    Saves final results of the experiments to results/agree/ and results/errors/. The paths for each type of experiment, respectively, are:
        (1) results/agree/<dataset>_<model>.pt
        (2) results/errors/<dataset>_<model>.pt
    """
    if args.agree:
        agree_dir = os.path.join(RESULTS_PATH, args.agree_dir)
        Path(agree_dir).mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            for model in models:
                f = os.path.join(agree_dir, "{}_{}.pt".format(dataset, model))
                torch.save(results['agree'][dataset][model], f)
    if args.errs:
        errs_dir = os.path.join(RESULTS_PATH, args.errs_dir)
        Path(errs_dir).mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            for model in models:
                f = os.path.join(errs_dir, "{}_{}.pt".format(dataset, model))
                torch.save(results['errs'][dataset][model], f)

def save_result(models, results, path):
    """
    Save final results fo the experiments to results/agree/ and results/errors/. The path for each type of experiment, respectively, are:
        (1) results/agree/<dataset>_<model>.pt
        (2) results/errors/<dataset>_<model>.pt
    """
    dir = os.path.join(RESULTS_PATH, path)
    Path(dir).mkdir(parents=True, exist_ok=True)
    for model in models:
        f = os.path.join(dir, "{}.pt".format(model))
        torch.save(results[model], f)
        print("Saved model={} results to {}!".format(model, path))

def main(args, datasets, models):
    """
    Runs agreement experiments and group conditional error experiments for all models and datasets specified.
    """
    results = {}
    results['agree'] = {}
    results['errs'] = {}

    for dataset_name in datasets:
        dataset = name_to_dataset(dataset_name)  # Converts to Dataset object
        if args.agree:
            print("\n=== Agreement on dataset: {} ===".format(dataset_name))
            agree_results = run_agreement(args, dataset, models)
            print_agreement(agree_results, models)
            save_result(models, agree_results, 'agree/{}'.format(dataset_name))
            #results['agree'][dataset_name] = agree_results
        if args.errs:
            print("\n=== Errors on dataset: {} ===".format(dataset_name))
            errs_results = run_errors(args, dataset, models)
            save_result(models, errs_results, 'errors/{}'.format(dataset_name))
            #save_results()
            #results['errs'][dataset_name] = errs_results

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_models', action='store_true', default=False)
    parser.add_argument('--group_params', action='store_true', default=False)
    parser.add_argument('--agree_dir', default="agree")
    parser.add_argument('--bootstraps', default=1000, type=int)
    parser.add_argument('--skip_agree', action='store_true', default=False)
    parser.add_argument('--errs_dir', default='errors')
    parser.add_argument('--skip_errs', action='store_true', default=False)
    parser.add_argument('--dataset', default=-1, type=int)
    parser.add_argument('--model', default=-1)
    parser.add_argument('--random_state', default=0, type=int)
    parser.add_argument('--n_cpus', default=16, type=int)

    args = parser.parse_args()
    args.agree = True
    args.errs = True
    if args.dataset == -1: # Run all datasets
        datasets = DATASETS
    else:
        datasets = [DATASETS[int(args.dataset)]]
    if args.model == -1: # Run all models
        models = MODELS
    else:
        models = [MODELS[int(args.model)]]
    if args.skip_agree:
        args.agree = False
    if args.skip_errs:
        args.errs = False

    results = main(args, datasets, models)
    # save_results(args, datasets, models, results)
    
