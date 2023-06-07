import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from train_utils import cross_validate

# Adult Dataset: Preprocessing
def adult_gp_indices(df, race_val, sex_val):
    if race_val == "NotWhite":
        return np.where((df['race'] != 'White') & (df['sex'] == sex_val))
    else:
        return np.where((df['race'] == race_val) & (df['sex'] == sex_val))

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

# group_train and group_test have the indices in X_train, X_test (respectively)
# for each group, as a binary mask.
num_group_train = {}
num_group_test = {}

print('Group\t\t\ttrain\ttest')
for g in range(num_groups):
    num_group_train[g] = np.sum(group_train[g])
    num_group_test[g] = np.sum(group_test[g])
    print('{0} ({3})\t\t\t{1}\t{2}'.format(g, num_group_train[g], num_group_test[g], group_names[g]))

for i in range(num_groups):
    print('P(Y=1 | group {0}) = {1}'.format(i, np.mean(y_test[group_test[i]])))

# Cross-validation for GradientBoostingClassifier and XGB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Hyperparameter sweep for gradient-boosted trees
gb_grid = {
    'max_depth': [2, 4, 8, 16],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [0]
}

best_gb_params_avg, best_gb_params_worstgp = cross_validate(X, y, col_transf, group_memberships, num_groups, 
                                                            GradientBoostingClassifier, gb_grid)
print(best_gb_params_avg)
print(best_gb_params_worstgp)

# Save to pickle files
with open('adult_trees_params/best_gb_params_avg.pkl', 'wb') as handle:
    pickle.dump(best_gb_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('adult_trees_params/best_gb_params_worstgp.pkl', 'wb') as handle:
    pickle.dump(best_gb_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Hyperparameter sweep for XGBoost
xgb_grid = {
    'max_depth': [2, 4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [0]
}

best_xgb_params_avg, best_xgb_params_worstgp = cross_validate(X, y, col_transf, group_memberships, num_groups, 
                                                            XGBClassifier, xgb_grid)
print(best_xgb_params_avg)
print(best_xgb_params_worstgp)

# Save to pickle files
with open('adult_trees_params/best_xgb_params_avg.pkl', 'wb') as handle:
    pickle.dump(best_xgb_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('adult_trees_params/best_xgb_params_worstgp.pkl', 'wb') as handle:
    pickle.dump(best_xgb_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Hyperparameter sweep for AdaBoost
ada_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [0]
}

best_ada_params_avg, best_ada_params_worstgp = cross_validate(X, y, col_transf, group_memberships, num_groups, 
                                                            AdaBoostClassifier, ada_grid)
print(best_ada_params_avg)
print(best_ada_params_worstgp)

# Save to pickle files
with open('adult_trees_params/best_ada_params_avg.pkl', 'wb') as handle:
    pickle.dump(best_ada_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('adult_trees_params/best_ada_params_worstgp.pkl', 'wb') as handle:
    pickle.dump(best_ada_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
n_estimators_params = [8, 16, 32, 64, 128, 256, 512]

param_grid = list(ParameterGrid(grid))
print("Fitting {} models total...".format(len(param_grid) * len(n_estimators_params)))
best_gb_params_avg = {}
best_gb_params_worstgp = {}
for n_estimators in n_estimators_params:
    best_avg_err = np.Inf
    best_worstgp_err = np.Inf
    for params in param_grid:
        params['n_estimators'] = n_estimators
        gb_model = GradientBoostingClassifier(**params)
        start = time.time()
        mean_test_errs, _ = paralell_split_eval(X, y, gb_model, col_transf, group_memberships, num_groups, train_size=0.8, n_splits=5, n_jobs=5)
        end = time.time()
        print("Cross-validated GradientBoostingClassifier ({}) in {} seconds.".format(params, end - start))

        avg_err = mean_test_errs[0]
        worstgp_err = max(mean_test_errs.values())
        print("\tAverage Error: {}".format(avg_err))
        print("\tWorst-group Error: {}".format(worstgp_err))

        if avg_err < best_avg_err:
            best_avg_err = avg_err
            best_gb_params_avg[n_estimators] = params.copy()
        if worstgp_err < best_worstgp_err:
            best_worstgp_err = worstgp_err
            best_gb_params_worstgp[n_estimators] = params.copy()
'''