from sklearn.model_selection import train_test_split, ParameterGrid
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