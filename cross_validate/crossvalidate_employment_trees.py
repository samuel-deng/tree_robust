# %% [markdown]
# # Subgroup Robustness Grows on Trees (Experiments)
# Here are some initial experiments corroborating the empirical observations from *Subgroup Robustness Grows on Trees*.

# %%
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from train_utils import cross_validate

from folktables import ACSDataSource, ACSEmployment

# Preprocess Employment Dataset (folktables)

# Download data and define groups
data_source = ACSDataSource(survey_year='2016', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)
sex = features[:, -2]
old = (features[:,0] > 65)
features.shape

# Define groups 
group_names = []
group_memberships = []
group_memberships.append([True] * label.shape[0])
group_names.append('ALL')
for g in np.unique(group):
    if g == 4 or g == 5: # group is too small
        continue
    group_memberships.append(group == g)
    group_names.append('R{0}'.format(g))
group_memberships.append(sex == 1)
group_names.append('S1')
group_memberships.append(sex == 2)
group_names.append('S2')
group_memberships.append(old == False)
group_names.append('A1')
group_memberships.append(old == True)
group_names.append('A2')
num_groups = len(group_memberships)
print('num_groups = {0}'.format(num_groups))

to_one_hot = set(['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'DREM', 'RAC1P'])
to_leave_alone = set(ACSEmployment.features) - to_one_hot
one_hot_inds = [i for i, x in enumerate(ACSEmployment.features) if x in to_one_hot]
leave_alone_inds = [i for i, x in enumerate(ACSEmployment.features) if x in to_leave_alone]

steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', MinMaxScaler(), leave_alone_inds)]
col_transf = ColumnTransformer(steps)
features_t = col_transf.fit_transform(features).toarray()
print("Column-transformed X has shape: {}".format(features_t.shape))

# Train-test split
splits = train_test_split(*tuple([features, label] + group_memberships), test_size=0.2, random_state=0)
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

print('Group\ttrain\ttest')
for g in range(num_groups):
    num_group_train[g] = np.sum(group_train[g])
    num_group_test[g] = np.sum(group_test[g])
    print('{0} ({3})\t{1}\t{2}'.format(g, num_group_train[g], num_group_test[g], group_names[g]))

for i in range(num_groups):
    print('P(Y=1 | group {0}) = {1}'.format(i, np.mean(y_test[group_test[i]])))

# Cross-validation for GradientBoostingClassifier and XGB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

EMPLOYMENT_PARAMS_PATH = 'employment_trees_params/'

# Hyperparameter sweep for gradient-boosted trees
gb_grid = {
    'max_depth': [2, 4, 8, 16],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [0]
}

best_gb_params_avg, best_gb_params_worstgp = cross_validate(X_train, y_train, col_transf, group_train, num_groups, 
                                                            GradientBoostingClassifier, gb_grid)
print(best_gb_params_avg)
print(best_gb_params_worstgp)

# Save to pickle files
with open(os.path.join(EMPLOYMENT_PARAMS_PATH, 'best_gb_params_avg.pkl'), 'wb') as handle:
    pickle.dump(best_gb_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(EMPLOYMENT_PARAMS_PATH, 'best_gb_params_worstgp.pkl'), 'wb') as handle:
    pickle.dump(best_gb_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Hyperparameter sweep for XGBoost trees
xgb_grid = {
    'max_depth': [2, 4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [0]
}

best_xgb_params_avg, best_xgb_params_worstgp = cross_validate(X_train, y_train, col_transf, group_train, num_groups, 
                                                            XGBClassifier, xgb_grid)
print(best_xgb_params_avg)
print(best_xgb_params_worstgp)

# Save to pickle files
with open(os.path.join(EMPLOYMENT_PARAMS_PATH, 'best_xgb_params_avg.pkl'), 'wb') as handle:
    pickle.dump(best_xgb_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(EMPLOYMENT_PARAMS_PATH, 'best_xgb_params_worstgp.pkl'), 'wb') as handle:
    pickle.dump(best_xgb_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Hyperparameter sweep for AdaBoost
ada_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'random_state': [0]
}

best_ada_params_avg, best_ada_params_worstgp = cross_validate(X_train, y_train, col_transf, group_train, num_groups, 
                                                            AdaBoostClassifier, ada_grid)
print(best_ada_params_avg)
print(best_ada_params_worstgp)

# Save to pickle files
with open(os.path.join(EMPLOYMENT_PARAMS_PATH, 'best_ada_params_avg.pkl'), 'wb') as handle:
    pickle.dump(best_ada_params_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(EMPLOYMENT_PARAMS_PATH, 'best_ada_params_worstgp.pkl'), 'wb') as handle:
    pickle.dump(best_ada_params_worstgp, handle, protocol=pickle.HIGHEST_PROTOCOL)