from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from joblib import Parallel, delayed
import numpy as np
import time

'''
Function for training a model ERM on some KFold splits and then evaluating on
each group for each of the splits.
'''
def split_and_eval(X, y, model, transformer, group_memberships, num_groups, train_size):
    splits = train_test_split(*tuple([X, y] + group_memberships), train_size=train_size)
    X_train = splits[0]
    X_test = splits[1]
    y_train = splits[2]
    y_test = splits[3]

    group_train = splits[4::2]
    group_test = splits[5::2]
    model.fit(transformer.transform(X_train), y_train)

    # Evaluate on all groups
    yhats = {}
    test_errs = {}
    for g in range(num_groups):
        yhats[g] = model.predict(transformer.transform(X_test))
        test_errs[g] = np.mean(y_test[group_test[g]] != yhats[g][group_test[g]])        
    
    return yhats, test_errs

def paralell_split_eval(X, y, model, transformer, group_memberships, num_groups, train_size, n_splits, n_jobs=-1):
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
        delayed(split_and_eval)(X, y, model, transformer, group_memberships, 
                                num_groups, train_size
        )
        for split in range(n_splits)
    )

    mean_test_errs = {}
    std_errs = {}
    for g in range(num_groups):
        test_errs_g = [test_errs[g] for _, test_errs in results]
        mean_test_errs[g] = np.mean(test_errs_g)
        std_errs[g] = np.sqrt( np.var(test_errs_g) ) / n_splits

    return mean_test_errs, std_errs

def cross_validate(X, y, transformer, group_memberships, num_groups, model, grid,
                   train_size=0.8, n_splits=5, n_jobs=5, n_estimators=128):
    best_avg_err = np.Inf
    best_worstgp_err = np.Inf

    param_grid = list(ParameterGrid(grid))
    print("Fitting {} models...".format(len(param_grid)))
    for params in param_grid:
        params['n_estimators'] = n_estimators
        param_model = model(**params)
        start = time.time()
        mean_test_errs, _ = paralell_split_eval(X, y, param_model, transformer, group_memberships, num_groups, 
                                                train_size=train_size, n_splits=n_splits, n_jobs=n_jobs)
        end = time.time()
        print("Cross-validated {} ({}) in {} seconds.".format(model, params, end - start))

        avg_err = mean_test_errs[0]
        worstgp_err = max(mean_test_errs.values())
        print("\tAverage Error: {}".format(avg_err))
        print("\tWorst-group Error: {}".format(worstgp_err))

        if avg_err < best_avg_err:
            best_avg_err = avg_err
            best_params_avg = params.copy()
        if worstgp_err < best_worstgp_err:
            best_worstgp_err = worstgp_err
            best_params_worstgp = params.copy()

    return best_params_avg, best_params_worstgp

'''
For a fixed model class, find the best hyperparameters for the model trained on
ONLY samples from each group. This gives |G| different sets of hyperparameters.
'''
def cross_val_group_helper(X, y, transformer, model, params, n_splits):
    clf = model(**params)
    scores = cross_val_score(clf, transformer.transform(X), y, cv=n_splits)
    return (scores.mean(), params)

def cross_validate_pergroup(X, y, transformer, group_memberships, num_groups,
                            model, grid, n_splits=3, n_jobs=8):
    param_grid = list(ParameterGrid(grid))
    best_params_pergroup = {}
    for g in range(num_groups):
        print("Fitting {} models to group {}...".format(len(param_grid), g))
        start = time.time()
        X_g = X[group_memberships[g]]
        y_g = y[group_memberships[g]]
        parallel = Parallel(n_jobs=-1)
        # results will have len(param_grid) tuples of (score, params)
        results = parallel(
            delayed(cross_val_group_helper)(X_g, y_g, transformer, 
                                            model, params, n_splits)
            for params in param_grid
        )
        end = time.time()
        print("Took {} seconds for group {}.".format(end - start, g))

        # find best parameter for group g
        scores = np.array([score for score, _ in results])
        best_params_pergroup[g] = results[np.argmax(scores)][1]
        print("best params for G{}: {}".format(g, best_params_pergroup[g]))

    return best_params_pergroup