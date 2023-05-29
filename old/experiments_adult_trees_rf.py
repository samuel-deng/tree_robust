# %% [markdown]
# # Subgroup Robustness Grows on Trees (Trees - Adult)
# In this notebook, we (empirically) investigate the different trees produced by three tree ensemble methods and evaluate
# their *individual* performance on groups in the data. In particular, we investigate the trees constructed by:
# 1. Random Forests
# 2. Gradient-boosted Trees (`sklearn`)
# 3. Gradient-boosted Trees (`XGBoost`)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle
import time

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# %% [markdown]
# ## Datasets Overview
# We'll run experiments on four datasets typically used in the fairness and robustness literature:
# 
# 1. ACS Income (*large*). $n = 499,350$ examples, $20$ features, sensitive groups: *Race* and *Sex*.
# 2. ACS Employment (*large*). $n = 376,035$ examples, $17$ features, sensitive groups: *Race* and *Sex*. 
# 3. Adult (*medium*) $n = 48,845$ examples, $14$ features, sensitive groups: *Race* and *Sex*.
# 4. Communities and Crime (*small*) $n = 1,994$ examples, $113$ features, sensitive groups: *Income Level* and *Race.*
# 
# We also plan to do experiments with a fifth synthetic dataset that has overlapping group structure.

# %% [markdown]
# # Adult Dataset (Overlapping Groups)
# The Adult Dataset is a benchmark derived from 1994 US Census data. The task is to predict whether an individual's income
# exceeds $50,000 (binary classification). Sensitive attributes are *Race* and *Sex*.
# 
# - *Race:* $\{\text{White}, \text{Asian-Pac-Islander}, \text{Amer-Indian-Eskimo}, \text{Other}, \text{Black} \}$
# - *Sex:* $\{\text{Male}, \text{Female}\}$
# 
# The standard in the literature has been to simplify the groups for *Race* to $\text{White}$ and $\text{NotWhite}$.

# %% [markdown]
# ## Preprocess and Clean Adult
# The Adult dataset is a binary classification task with *categorical* and *numerical* features. We binarize the categorical
# features with `OneHotEncoder` and normalize the numerical features with `MinMaxScaler`. We also drop all the rows with missing
# features, leaving us with $n = 45,222$ examples total.

# %%
# Adult Dataset
adult_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country", 
                "income"]
adult_data = pd.read_csv("./datasets/adult/adult.data", header=None, names=adult_names, na_values=' ?')
adult_data = adult_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
adult_test = pd.read_csv("./datasets/adult/adult.test", header=None, names=adult_names, na_values=' ?')
adult_test = adult_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
dfs = [adult_data, adult_test]
adult_df = pd.concat(dfs)
adult_df = adult_df.dropna()
adult_df = adult_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
print("Adult Shape: {}".format(adult_df.shape))

# last column in adult has some textual discrepancy
adult_df = adult_df.replace(">50K.", ">50K")
adult_df = adult_df.replace("<=50K.", "<=50K")

# Split into X and y
X, y = adult_df.drop("income", axis=1), adult_df["income"]

# Select categorical and numerical features
cat_idx = X.select_dtypes(include=["object", "bool"]).columns
num_idx = X.select_dtypes(include=['int64', 'float64']).columns
steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', MinMaxScaler(), num_idx)]
col_transf = ColumnTransformer(steps)

# label encoder to target variable so we have classes 0 and 1
assert(len(np.unique(y)) == 2)
y = LabelEncoder().fit_transform(y)
print("% examples >=50k (y=1): {}".format(100 * len(np.where(y == 1)[0])/len(y)))
print("% examples <50k (y=0): {}".format(100 * len(np.where(y == 0)[0])/len(y)))

# %% [markdown]
# Traditionally, there are $|\mathcal{G}| = 4$ groups in Adult: *(White, Male), (NotWhite, Male), (White, Female),* and *(NotWhite, Female)*.
# 
# In this notebook, we will instead do the following group structure for Adult:
# - Overlapping groups (make *White* and *NotWhite* their own groups, make *Male* and *Female* their own groups).
# 

# %%
def adult_gp_indices(df, race_val, sex_val):
    if race_val == "NotWhite":
        return np.where((df['race'] != 'White') & (df['sex'] == sex_val))
    else:
        return np.where((df['race'] == race_val) & (df['sex'] == sex_val))

group_names = ["ALL", "W,M", "W,F", "nW,M", "nW,F", "W", "nW", "M", "F"]
group_memberships = []
group_memberships.append([True] * y.shape[0])
race_gps_coarse = ["White", "NotWhite"]
sex_gps = ["Male", "Female"]

# Traditional disjoint groups
for race in race_gps_coarse:
    for sex in sex_gps:
        indices = adult_gp_indices(X, race, sex)[0]
        membership = np.zeros(y.shape[0], dtype=bool)
        membership[indices] = True
        group_memberships.append(membership)

# Add 4 overlapping groups
w_indices = np.where(X['race'] == 'White')
w_membership = np.zeros(y.shape[0], dtype=bool)
w_membership[w_indices] = True
group_memberships.append(w_membership)

nw_indices = np.where(X['race'] != 'White')
nw_membership = np.zeros(y.shape[0], dtype=bool)
nw_membership[nw_indices] = True
group_memberships.append(nw_membership)

m_indices = np.where(X['sex'] == 'Male')
m_membership = np.zeros(y.shape[0], dtype=bool)
m_membership[m_indices] = True
group_memberships.append(m_membership)

f_indices = np.where(X['sex'] == 'Female')
f_membership = np.zeros(y.shape[0], dtype=bool)
f_membership[f_indices] = True
group_memberships.append(f_membership)

num_groups = len(group_memberships)
print('num_groups = {0}'.format(num_groups))

# %%
# Fit the ColumnTransformer to X
X_transf = col_transf.fit_transform(X)
print("Column-transformed X has shape: {}".format(X_transf.shape))

# Train-test split
splits = train_test_split(*tuple([X, y] + group_memberships), test_size=0.2, random_state=0)
X_train = splits[0]
X_test = splits[1]
y_train = splits[2]
y_test = splits[3]

# group_train
group_train = splits[4::2]
group_test = splits[5::2]

# %%
# group_train and group_test have the indices in X_train, X_test (respectively)
# for each group, as a binary mask.
num_group_train = {}
num_group_test = {}

# TODO: fix column alignment issue :(
print('Group\t\t\ttrain\ttest')
for g in range(num_groups):
    num_group_train[g] = np.sum(group_train[g])
    num_group_test[g] = np.sum(group_test[g])
    print('{0} ({3})\t\t\t{1}\t{2}'.format(g, num_group_train[g], num_group_test[g], group_names[g]))

# %%
for i in range(num_groups):
    print('P(Y=1 | group {0}) = {1}'.format(i, np.mean(y_test[group_test[i]])))

# %% [markdown]
# # Utility Code for Parameter Searching
# Just some code to make things easier.

# %%
# Utilities for fitting  and evaluating models
from sklearn.model_selection import cross_val_predict, GridSearchCV
from joblib import Parallel, delayed

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

# %% [markdown]
# # Fit Random Forests
# First, we fit `sklearn` Random Forests to the dataset, over a hyperparameter sweep on following parameters:
# 1. `n_estimators`: the number of trees total in the forest.
# 2. `max_depth`: the maximum depth of the tree, an `int`. 
# 3. `min_samples_split`: the minimum number of samples required to split at an internal node. if the number of samples at a node is *less than* `min_samples_split`, we
# that node becomes a leaf node.
# 4. `ccp_alpha`: the complexity parameter used in Minimal Cost-Complexity Pruning. for nonzero values, pruning is performed.
# 
# The parameter sweep is linear (unparallelized); the cross-validation is parallelized.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

# %%
# Hyperparameter sweep for RF
n_estimators_params = [8, 16, 32, 64, 128, 256]
grid = {
    'max_depth': [2, 4, 8, 16, None],
    'ccp_alpha': [0., 0.001, 0.01, 0.1],
    'random_state': [0]
}

param_grid = list(ParameterGrid(grid))
print("Fitting {} models total...".format(len(param_grid) * len(n_estimators_params)))
best_rf_params_avg = {}
best_rf_params_worstgp = {}
for n_estimators in n_estimators_params:
    best_avg_err = np.Inf
    best_worstgp_err = np.Inf
    for params in param_grid:
        params['n_estimators'] = n_estimators
        rf_model = RandomForestClassifier(**params)
        start = time.time()
        mean_test_errs, _ = paralell_split_eval(X, y, rf_model, col_transf, group_memberships, num_groups, train_size=0.8, n_splits=5, n_jobs=5)
        end = time.time()
        print("Cross-validated RandomForestClassifier ({}) in {} seconds.".format(params, end - start))

        avg_err = mean_test_errs[0]
        worstgp_err = max(mean_test_errs.values())
        print("\tAverage Error: {}".format(avg_err))
        print("\tWorst-group Error: {}".format(worstgp_err))

        if avg_err < best_avg_err:
            best_avg_err = avg_err
            best_rf_params_avg[n_estimators] = params.copy()
        if worstgp_err < best_worstgp_err:
            best_worstgp_err = worstgp_err
            best_rf_params_worstgp[n_estimators] = params.copy()

# %%
# Save the best parameters to not have to cross-validate again
print(best_rf_params_avg)
print(best_rf_params_worstgp)

# Save to pickle files
with open('adult_trees_params/best_rf_params_avg.pkl', 'wb') as handle:
    pickle.dump(best_rf_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('adult_trees_params/best_rf_params_worstgp.pkl', 'wb') as handle:
    pickle.dump(best_rf_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)