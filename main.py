#!/user/sd3013/.conda/envs/agreement/bin/python
"""
Main function for running and evaluating (1) group agreement (2) group errors on each dataset and each model class.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import itertools

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
    'DecisionTree8',
    'DecisionTree16',
    'DecisionTree',
    'RandomForest2',
    'RandomForest4',
    'RandomForest8',
    'RandomForest16',
    'RandomForest',
    'XGBoost',
    #'MLP'
    ]
# LogisticRegressionSGD

DATASETS = [
    'adult',
    'communities',
    'compas',
    'german',
    'incomeCA',
    'incomeNY',
    'incomeTX',
    'employmentCA',
    'employmentNY',
    'employmentTX',
    'coverageCA',
    'coverageNY',
    'coverageTX'
    ]

# For multiple state experiments
STATE_DATASETS = [
    'income_ST_race',
    'income_ST_sex',
    'employment_ST_race',
    'employment_ST_sex',
    'coverage_ST_race',
    'coverage_ST_sex'
]

# For single state experiments
HIER_TYPES = ['rsa', 'ras', 'asr', 'ers', 'esr', 'res', 'ser']
#HIER_STATES = ['MA', 'CT', 'NY', 'PA', 'IL', 'OH', 'MO', 'MN', 'FL', 'GA',
#                 'TN', 'AL', 'TX', 'LA', 'AZ', 'CO', 'CA', 'WA']
HIER_STATES = ['NY', 'IL', 'FL', 'CA']
HIER_TASKS = ['income', 'coverage', 'employment']

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
        if args.errs:
            print("\n=== Errors on dataset: {} ===".format(dataset_name))
            errs_results = run_errors(args, dataset, models)
            save_result(models, errs_results, dataset_name)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_models', action='store_true', default=False)
    parser.add_argument('--group_params', action='store_true', default=False)
    parser.add_argument('--bootstraps', default=1000, type=int)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--model', default=-1)
    parser.add_argument('--random_state', default=0, type=int)
    parser.add_argument('--n_cpus', default=2, type=int)
    parser.add_argument('--hier', action='store_true', default=False)
    parser.add_argument('--states', action='store_true', default=False)

    args = parser.parse_args()
    args.agree = False
    args.errs = True
    if args.model == -1: # Run all models
        models = MODELS
    else:
        models = [MODELS[int(args.model)]]

    # Hierarchical clustering experiments
    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    elif args.hier:
        # Hierarchical datasets for task/state/type
        data_combos = itertools.product(HIER_TASKS, HIER_STATES, HIER_TYPES)
        for task, state, hier in data_combos:
            datasets.append(task + "_" + state + "_" + hier)
    elif args.states:
        datasets = STATE_DATASETS
    else:
        raise ValueError("Use the --dataset, --hier, or --states flag!")
    results = main(args, datasets, models)
    