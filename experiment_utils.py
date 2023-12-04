from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from joblib import Parallel, delayed
import numpy as np

def std_err(n:int, e:float):
    """Return the lower and upper bound on error rate when test set size is n and empirical error rate is e"""
    assert e >= 0. and e <= 1 and n >= 0, f'Invalid input: n={n}, e={e}'
    a = 4.+n
    b = 2.+n*e
    c = n*e**2
    d = 2.*np.sqrt(1.+n*e*(1.-e))
    return ((b-d)/a, (b+d)/a)

def get_errors(X, y, group_memberships, 
               dt_params, rf_params, gb_params, xgb_params, 
               light_gb=False, bootstrap=True):
    # Bootstrap from full dataset
    num_groups = len(group_memberships)
    if bootstrap:
        splits = resample(*tuple([X, y] + group_memberships), replace=True,
                        n_samples=X.shape[0])
        X_boot = splits[0]
        y_boot = splits[1]
        groups_boot = splits[2:]
    else:
        X_boot = X
        y_boot = y
        groups_boot = group_memberships

    # Train-test split
    splits = train_test_split(*tuple([X_boot, y_boot] + groups_boot),
                            test_size=0.2, random_state=0)
    X_train = splits[0]
    X_test = splits[1]
    y_train = splits[2]
    y_test = splits[3]

    # group_train
    group_train = splits[4::2]
    group_test = splits[5::2]

    # Fit models
    dt_models = {}
    dt_preds = {}
    dt_errs = {}
    dt_erm_preds = {}
    dt_erm_errs = {}
    #rf_models = {}
    rf_model = RandomForestClassifier(**rf_params[0]).fit(X_train, y_train)
    rf_preds = {}
    rf_errs = {}
    #gb_models = {}
    if light_gb:
        print("Using LightGBM!")
        gb_model = HistGradientBoostingClassifier(**gb_params[0]).fit(X_train.toarray(), y_train)
    else:
        gb_model = GradientBoostingClassifier(**gb_params[0]).fit(X_train, y_train)
    gb_preds = {}
    gb_errs = {}
    #xgb_models = {}
    xgb_model = XGBClassifier(**xgb_params[0]).fit(X_train, y_train)
    xgb_preds = {}
    xgb_errs = {}
    print("Done fitting all the ensemble classifiers!")

    for g in range(num_groups):
        print("Fitting DT for group {}...".format(g))
        # Decision Trees
        dt_models[g] = DecisionTreeClassifier(**dt_params[g])
        dt_models[g].fit(X_train[group_train[g]], y_train[group_train[g]])
        dt_preds[g] = dt_models[g].predict(X_test[group_test[g]])
        dt_errs[g] = np.mean(y_test[group_test[g]] != dt_preds[g])

        dt_erm_preds[g] = dt_models[0].predict(X_test[group_test[g]])
        dt_erm_errs[g] = np.mean(y_test[group_test[g]] != dt_erm_preds[g])
        
        # Random Forest
        #rf_models[g] = RandomForestClassifier(**rf_params[g])
        #rf_models[g].fit(X_train[group_train[g]], y_train[group_train[g]])
        rf_preds[g] = rf_model.predict(X_test[group_test[g]])
        rf_errs[g] = np.mean(y_test[group_test[g]] != rf_preds[g])

        # Gradient Boosting
        #gb_models[g] = GradientBoostingClassifier(**gb_params[g])
        #gb_models[g].fit(X_train[group_train[g]], y_train[group_train[g]])
        if light_gb:
            gb_preds[g] = gb_model.predict(X_test[group_test[g]].toarray())
        else:
            gb_preds[g] = gb_model.predict(X_test[group_test[g]])
        gb_errs[g] = np.mean(y_test[group_test[g]] != gb_preds[g])

        # XGBoost
        #xgb_models[g] = XGBClassifier(**xgb_params[g])
        #xgb_models[g].fit(X_train[group_train[g]], y_train[group_train[g]])
        xgb_preds[g] = xgb_model.predict(X_test[group_test[g]])
        xgb_errs[g] = np.mean(y_test[group_test[g]] != xgb_preds[g])

    return dt_errs, dt_erm_errs, rf_errs, gb_errs, xgb_errs

def error_trials(n_bootstraps, X, y, group_memberships, 
                       dt_params, rf_params, gb_params, xgb_params, light_gb=False, bootstrap=True):
    num_groups = len(group_memberships)
    if bootstrap:
        results = Parallel(n_jobs=-1)(delayed(get_errors)(
                                X, y, group_memberships, 
                                dt_params, rf_params, gb_params, xgb_params, light_gb) 
                                for _ in range(n_bootstraps))

        # Compile all bootstrap results
        dt_errs_all = [[] for _ in range(num_groups)]
        dt_erm_errs_all = [[] for _ in range(num_groups)]
        rf_errs_all = [[] for _ in range(num_groups)]
        gb_errs_all = [[] for _ in range(num_groups)]
        xgb_errs_all = [[] for _ in range(num_groups)]
        for i in range(n_bootstraps):
            dt_errs, dt_erm_errs, rf_errs, gb_errs, xgb_errs = results[i]
            for j in range(num_groups):
                dt_errs_all[j].append(dt_errs[j])
                dt_erm_errs_all[j].append(dt_erm_errs[j])
                rf_errs_all[j].append(rf_errs[j])
                gb_errs_all[j].append(gb_errs[j])
                xgb_errs_all[j].append(xgb_errs[j])

        # Compute mean and standard errors
        dt_errs_stats = []
        dt_erm_errs_stats = []
        rf_errs_stats = []
        gb_errs_stats = []
        xgb_errs_stats = []
        for group in range(num_groups):
            dt_errs_stats.append((np.mean(dt_errs_all[group]), 
                                np.std(dt_errs_all[group])))
            dt_erm_errs_stats.append((np.mean(dt_erm_errs_all[group]),
                                    np.std(dt_erm_errs_all[group])))
            rf_errs_stats.append((np.mean(rf_errs_all[group]), 
                                np.std(rf_errs_all[group])))
            gb_errs_stats.append((np.mean(gb_errs_all[group]), 
                                np.std(gb_errs_all[group])))
            xgb_errs_stats.append((np.mean(xgb_errs_all[group]), 
                                np.std(xgb_errs_all[group])))
    else:
        results = get_errors(X, y, group_memberships, dt_params, rf_params,
                             gb_params, xgb_params, light_gb=True, bootstrap=False)
        dt_errs, dt_erm_errs, rf_errs, gb_errs, xgb_errs = results

        dt_errs_stats = []
        dt_erm_errs_stats = []
        rf_errs_stats = []
        gb_errs_stats = []
        xgb_errs_stats = []
        for g in range(num_groups):
            n_g = len(X.toarray()[group_memberships[g]])
            dt_errs_stats.append((dt_errs[g], 
                                 std_err(n_g, dt_errs[g])))
            dt_erm_errs_stats.append((dt_erm_errs[g],
                                      std_err(n_g, dt_erm_errs[g])))
            rf_errs_stats.append((rf_errs[g],
                                 std_err(n_g, rf_errs[g])))
            gb_errs_stats.append((gb_errs[g],
                                 std_err(n_g, gb_errs[g])))
            xgb_errs_stats.append((xgb_errs[g],
                                  std_err(n_g, xgb_errs[g])))

    return dt_errs_stats, dt_erm_errs_stats, rf_errs_stats, gb_errs_stats, xgb_errs_stats