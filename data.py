import pandas as pd
import numpy as np
import os
import itertools

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from compas_utils import preprocess_compas_df
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage

INCOME_CATGEORICAL = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']
COVERAGE_CATEGORICAL = ['ESP', 'MAR', 'CIT', 'MIG', 'MIL', 
                      'ANC', 'ESR', 'ST', 'RAC1P']
EMPLOYMENT_CATEGORICAL = ['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
                          'DREM', 'RAC1P']

region_names = ['NORTHEAST', 'MIDWEST', 'SOUTH', 'WEST']
region_vals = [1, 2, 3, 4]
division_names = ['NewEngland', 'MidAtlantic', 'EastNorthCentral',
                  'WestNorthCentral', 'SouthAtlantic', 'EastSouthCentral',
                  'WestSouthCentral', 'Mountain', 'Pacific']
division_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
state_names = ['MA', 'CT', 'NY', 'PA', 'IL', 'OH', 'MO', 'MN', 'FL', 'GA',
                 'TN', 'AL', 'TX', 'LA', 'AZ', 'CO', 'CA', 'WA']
state_vals = [25, 9, 36, 42, 17, 39, 29, 27, 12, 13, 47, 1, 48, 22, 4, 8, 6, 53]

class Dataset:
    """
    Dataset object for handling the different datasets.
    Variables:
        X: Features for the dataset (ColumnTransformer applied).
        y: Labels (all binary) for the data.
        groups: Binary-valued arrays for group membership. Only includes original groups, not their intersections.
        intersections: Tuples of indices for group intersections (e.g. (1, 9)).
        group_names: List of all group names for the dataset (e.g. "W" or "Y")
        inter_names: List of all intersection names for the dataset (e.g. "(W, M)").
    """
    def __init__(self, name, X, y, groups, group_names, tree):
        self.name = name
        self.X = X
        self.y = y
        self.groups = groups
        self.group_names = group_names
        self.tree = tree

def adult_gp_indices(df, race_val, sex_val):
    if race_val == "NotWhite":
        return np.where((df['race'] != 'White') & (df['sex'] == sex_val))
    else:
        return np.where((df['race'] == race_val) & (df['sex'] == sex_val))
    
def communities_gp_indices(df, race_val, income_val):
    if race_val == "NotWhite":
        return np.where((df['race'] == 0) & (df['income_level'] == income_val))
    else:
        return np.where((df['race'] == 1) & (df['income_level'] == income_val))

def compas_gp_indices(df, race_val, sex_val):
    if race_val == "NotWhite":
        return np.where((df['race'] != 1) & (df['sex'] == sex_val))
    else:
        return np.where((df['race'] == 1) & (df['sex'] == sex_val))

def german_gp_indices(df, sex_val, age_val):
    if sex_val == "Male":
        return np.where((df['sex'] == 1) & (df['age'] == age_val))
    else:
        return np.where((df['sex'] == 0) & (df['age'] == age_val))

def construct_hier(groups, group_names, skip_empty=False):
    '''
    Helper function for constructing the hierarchical partitioning for the Folktables datasets. 'groups' takes a list of lists of group membership arrays (Boolean np.arrays).
    '''
    tree = []
    hier_groups = []
    hier_group_names = []
    num_levels = len(groups)
    if num_levels < 1:
        raise ValueError("num_levels should be at least 1!")

    for level in range(num_levels):
        tree.append([])
        if level == 0:
            for i, g in enumerate(groups[level]):
                hier_groups.append(g)
                hier_group_names.append(group_names[level][i])
                tree[level].append(len(hier_groups) - 1)
        else:
            for j, g_prev_index in enumerate(tree[level - 1]):
                g_prev = hier_groups[g_prev_index]
                for i, g in enumerate(groups[level]):
                    if skip_empty:
                        overlap = np.sum(g & g_prev)
                        if overlap == 0:
                            continue
                        else:
                            hier_groups.append(g & g_prev)
                            tree[level].append(len(hier_groups) - 1)
                            if hier_group_names[tree[level - 1][j]] == 'ALL':
                                hier_group_names.append(group_names[level][i])
                            else:
                                hier_group_names.append(
                                    hier_group_names[tree[level - 1][j]] + "," 
                                + group_names[level][i])
                    else:
                        hier_groups.append(g & g_prev)
                        tree[level].append(len(hier_groups) - 1)
                        if hier_group_names[tree[level - 1][j]] == 'ALL':
                            hier_group_names.append(group_names[level][i])
                        else:
                            hier_group_names.append(
                                hier_group_names[tree[level - 1][j]] + "," 
                                + group_names[level][i])
    
    return hier_groups, hier_group_names, tree

def preprocess_adult(train_path='datasets/adult/adult.data',
                     test_path='datasets/adult/adult.test'):
    adult_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                    "marital-status", "occupation", "relationship", "race","sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    adult_data = pd.read_csv(train_path, header=None, 
                            names=adult_names, na_values=' ?')
    adult_data = adult_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
    adult_test = pd.read_csv(test_path, header=None, 
                             names=adult_names, na_values=' ?')
    adult_test = adult_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace
    dfs = [adult_data, adult_test]
    adult_df = pd.concat(dfs)
    adult_df = adult_df.dropna()
    adult_df = adult_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespace

    # last column in adult has some textual discrepancy
    adult_df = adult_df.replace(">50K.", ">50K")
    adult_df = adult_df.replace("<=50K.", "<=50K")

    # Split into X and y
    X, y = adult_df.drop("income", axis=1), adult_df["income"]

    # Select categorical and numerical features
    cat_idx = X.select_dtypes(include=["object", "bool"]).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', StandardScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variable so we have classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)

    group_names = ["ALL", "W", "nW", "M", "F"]
    inter_names = ["(W,M)", "(W,F)", "(nW,M)", "(nW,F)"]
    groups = []
    intersections = []
    groups.append([True] * y.shape[0])
    race_gps_coarse = ["White", "NotWhite"]
    sex_gps = ["Male", "Female"]

    # Overlapping groups
    w_indices = np.where(X['race'] == 'White')
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    groups.append(w_membership)

    nw_indices = np.where(X['race'] != 'White')
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    groups.append(nw_membership)

    m_indices = np.where(X['sex'] == 'Male')
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    groups.append(m_membership)

    f_indices = np.where(X['sex'] == 'Female')
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    groups.append(f_membership)

    # Intersections of groups
    race_indices = [1, 2]
    sex_indices = [3, 4]
    intersections = list(itertools.product(race_indices, sex_indices))

    # Construct trees for TREEPEND
    trees = []
    trees.append(construct_tree(race_indices, intersections))
    trees.append(construct_tree(sex_indices, intersections))

    """
    for race in race_gps_coarse:
    for sex in sex_gps:
        indices = adult_gp_indices(X, race, sex)[0]
        membership = np.zeros(y.shape[0], dtype=bool)
        membership[indices] = True
        intersections.append(membership)
    """

    # Fit the ColumnTransformer to X
    X = col_transf.fit_transform(X)
    dataset = Dataset("adult", X, y, groups, intersections, 
                      group_names, inter_names, trees)
    return dataset

def preprocess_communities(train_path='datasets/communities-and-crime/attributes.csv', test_path='datasets/communities-and-crime/communities.data'):
    # Communities & Crime Dataset
    attrib = pd.read_csv(train_path, delim_whitespace=True)
    data = pd.read_csv(test_path, names=attrib['attributes'])

    target_threshold = 0.08
    # remove non predicitive features
    for c in ['state', 'county', 'community', 'communityname', 'fold']:
        data.drop(columns=c, axis=1, inplace=True)

    data = data.replace('?', np.nan).dropna(axis=1)
    data["race"] = data['racePctWhite'].apply(
        lambda x: 1 if x >= 0.85 else 0)
    income_thresh = data["medIncome"].median()
    data["income_level"] = data["medIncome"].apply(
        lambda x: 1 if x > income_thresh else 0)
    data = data.drop(columns=['racePctAsian', 'racePctHisp',
                                'racepctblack', 'whitePerCap',
                                'blackPerCap', 'indianPerCap',
                                'AsianPerCap',  # 'OtherPerCap',
                                'HispPerCap',
                                'racePctWhite', 'medIncome'
                                ], axis=1).rename(
        columns={'ViolentCrimesPerPop': "target"})
    data["target"] = (data["target"] >= target_threshold).astype(int)

    X, y = data.drop("target", axis=1), data["target"]
    cat_idx = X.select_dtypes(include=["object", "bool"]).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', StandardScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variale so we have two classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)

    group_names = ["ALL", "W", "nW", "H", "L"]
    inter_names = ["(W,H)", "(W,L)", "(nW,H)", "(nW,L)"]
    groups = []
    intersections = []
    groups.append([True] * y.shape[0])
    race_gps_coarse = ["White", "NotWhite"]
    income_gps = [1, 0]

    # Groups
    w_indices = np.where(X['race'] == 1)
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    groups.append(w_membership)

    nw_indices = np.where(X['race'] == 0)
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    groups.append(nw_membership)

    m_indices = np.where(X['income_level'] == 1)
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    groups.append(m_membership)

    f_indices = np.where(X['income_level'] == 0)
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    groups.append(f_membership)

    # Group Intersections
    race_indices = [1, 2]
    income_indices = [3, 4]
    intersections = list(itertools.product(race_indices, income_indices))

    # Construct trees for TREEPEND
    trees = []
    trees.append(construct_tree(race_indices, intersections))
    trees.append(construct_tree(income_indices, intersections))

    # Fit the ColumnTransformer to X
    X = col_transf.fit_transform(X)
    dataset = Dataset("communities", X, y, groups, intersections, 
                      group_names, inter_names, trees)
    return dataset

def preprocess_compas(train_path='datasets/compas/compas.csv'):
    # COMPAS Dataset
    compas_data = pd.read_csv(train_path, header=0, na_values='?')
    compas_df = preprocess_compas_df(compas_data)
    compas_df = compas_df.dropna()

    X, y = compas_df.drop("is_recid", axis=1), compas_df["is_recid"]
    cat_idx = ['c_charge_degree', 'sex', 'race', 'screening_year_is_2013']
    num_idx = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'age']
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', StandardScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variale so we have two classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)

    group_names = ["ALL", "W", "nW", "M", "F"]
    inter_names = ["(W,M)", "(W,F)", "(nW,M)", "(nW,F)"]
    groups = []
    groups.append([True] * y.shape[0])
    race_gps_coarse = ["White", "NotWhite"]
    sex_gps = [1, 0]

    # Add 4 overlapping groups
    w_indices = np.where(X['race'] == 1)
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    groups.append(w_membership)

    nw_indices = np.where(X['race'] != 1)
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    groups.append(nw_membership)

    m_indices = np.where(X['sex'] == 1)
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    groups.append(m_membership)

    f_indices = np.where(X['sex'] == 0)
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    groups.append(f_membership)

    # Group Intersections
    race_indices = [1, 2]
    sex_indices = [3, 4]
    intersections = list(itertools.product(race_indices, sex_indices))

    # Construct trees for TREEPEND
    trees = []
    trees.append(construct_tree(race_indices, intersections))
    trees.append(construct_tree(sex_indices, intersections))

    # Fit the ColumnTransformer to X
    X = col_transf.fit_transform(X)
    dataset = Dataset("compas", X, y, groups, intersections, 
                      group_names, inter_names, trees)
    return dataset

def preprocess_german(train_path='datasets/german/german.data'):
    df = pd.read_csv(train_path, sep=" ", header=None)
    df.columns = ["status", "duration", "credit_history",
                "purpose", "credit_amt", "savings_acct_bonds",
                "present_unemployed_since", "installment_rate",
                "per_status_sex", "other_debtors", "pres_res_since",
                "property", "age", "other_installment", "housing",
                "num_exist_credits", "job", "num_ppl", "has_phone",
                "foreign_worker", "target"]

    # Code labels as in tfds; see
    # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/structured/german_credit_numeric.py
    df["target"] = 2 - df["target"]

    # convert per_status_sex into separate columns.
    # Sens is 1 if male; else 0.
    df["sex"] = df["per_status_sex"].apply(
        lambda x: 1 if x not in ["A92", "A95"] else 0)

    # Age sens is 1 if above median age, else 0.
    median_age = df["age"].median()
    df["age"] = df["age"].apply(lambda x: 1 if x > median_age else 0)
    df["single"] = df["per_status_sex"].apply(
        lambda x: 1 if x in ["A93", "A95"] else 0)
    df.drop(columns="per_status_sex", inplace=True)

    # features 15-23 are categorical/indicators
    categorical_columns = [
        "status", "credit_history",
        "purpose", "savings_acct_bonds",
        "present_unemployed_since", "single",
        "other_debtors",
        "property", "other_installment", "housing",
        "job", "has_phone",
        "foreign_worker"]

    for colname in categorical_columns:
        df[colname] = df[colname].astype('category')

    X, y = df.drop("target", axis=1), df["target"]
    cat_idx = X.select_dtypes(include=["object", "bool", "category"]).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', StandardScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variale so we have two classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)

    group_names = ["ALL", "M", "F", "O", "Y"]
    inter_names = ["(M,O)", "(M,Y)", "(F,O)", "(F,Y)"]
    groups = []
    intersections = []
    groups.append([True] * y.shape[0])
    sex_gps = ["Male", "Female"]
    age_gps = [1, 0]

    # Groups
    w_indices = np.where(X['sex'] == 1)
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    groups.append(w_membership)

    nw_indices = np.where(X['sex'] == 0)
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    groups.append(nw_membership)

    m_indices = np.where(X['age'] == 1)
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    groups.append(m_membership)

    f_indices = np.where(X['age'] == 0)
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    groups.append(f_membership)

    # Group Intersections
    sex_indices = [1, 2]
    age_indices = [3, 4]
    intersections = list(itertools.product(sex_indices, age_indices))

    # Construct trees for TREEPEND
    trees = []
    trees.append(construct_tree(sex_indices, intersections))
    trees.append(construct_tree(age_indices, intersections))

    # Fit the ColumnTransformer to X
    X = col_transf.fit_transform(X)
    dataset = Dataset("german", X, y, groups, intersections, 
                      group_names, inter_names, trees)
    return dataset

def preprocess_folktables(task, state, hier):
    """
    Returns a Dataset object given a folktables task, state, and hierarchy of groups.
    """
    data_source = ACSDataSource(survey_year="2016", horizon="1-Year",
                                survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    if task == 'income':
        X, y, group = ACSIncome.df_to_numpy(acs_data)
        feature_names = ACSIncome.features
        to_one_hot = set(INCOME_CATGEORICAL)
    elif task == 'coverage':
        X, y, group = ACSPublicCoverage.df_to_numpy(acs_data)
        feature_names = ACSPublicCoverage.features
        to_one_hot = set(COVERAGE_CATEGORICAL)
    elif task == 'employment':
        X, y, group = ACSEmployment.df_to_numpy(acs_data)
        feature_names = ACSEmployment.features
        to_one_hot = set(EMPLOYMENT_CATEGORICAL)

    else:
        raise ValueError("Task for folktables invalid!")
    
    # get correct indices for each feature
    sex_idx = feature_names.index('SEX')
    sch_idx = feature_names.index('SCHL')
    age_idx = feature_names.index('AGEP')
    
    # Construct groups
    sex = X[:, sex_idx]
    young = (X[:,age_idx] < 35)
    mid = np.logical_and(X[:,age_idx] >= 35, X[:,age_idx] < 60)
    old = (X[:,age_idx] >= 60)

    sch_lhs = X[:,sch_idx] <= 15
    sch_hs = (X[:,sch_idx] > 15) & (X[:,sch_idx] < 20)
    sch_col = (X[:,sch_idx] >= 20) & (X[:,sch_idx] < 22)
    sch_adv = (X[:,sch_idx] >= 22)

    # 12 groups (including ALL)
    ALL = [True] * y.shape[0]
    race_groups = []
    # Get race groups, combining (R3, R4, R5) and (R6, R7)
    for g in np.unique(group):
        if g == 3:  # R3 (American-Indian)
            R34 = np.logical_or(group == 3, group == 4)
            race_groups.append(np.logical_or(R34, group == 5))
        elif g == 6: # R6 (Asian)
            race_groups.append(np.logical_or(group == 6, group == 7))
        elif g == 4 or g == 5 or g == 7: # group is too small
            continue
        else:
            race_groups.append(group == g)
    race_group_names = ["R1", "R2", "R3+", "R6+", "R7", "R8", "R9"]
    sex_groups = [sex == 1, sex == 2]
    sex_group_names = ["M", "F"]
    age_groups = [young, mid, old]
    age_group_names = ["Ya", "Ma", "Oa"]
    school_groups = [sch_lhs, sch_hs, sch_col, sch_adv]
    school_group_names = ["HS-", "HS", "COL", "COL+"]

    if hier == 'rsa':
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    sex_groups, age_groups],[["ALL"], race_group_names, sex_group_names,age_group_names])
    elif hier == 'ras':       
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    age_groups, sex_groups],[["ALL"], race_group_names, age_group_names, sex_group_names])
    elif hier == 'asr':
        groups, group_names, tree = construct_hier([[ALL], age_groups,
                                                    sex_groups, race_groups],[["ALL"], age_group_names, sex_group_names, race_group_names])
    elif hier == "ers":
        groups, group_names, tree = construct_hier([[ALL], school_groups,
                                                    race_groups, sex_groups],[["ALL"], school_group_names, race_group_names, sex_group_names])
    elif hier == "esr":
        groups, group_names, tree = construct_hier([[ALL], school_groups,
                                                    sex_groups, race_groups],[["ALL"], school_group_names, sex_group_names, race_group_names])
    elif hier == "res":
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    school_groups, sex_groups],[["ALL"], race_group_names, school_group_names, sex_group_names])
    elif hier == "ser":
        groups, group_names, tree = construct_hier([[ALL], sex_groups,
                                                    school_groups, race_groups],[["ALL"], sex_group_names, school_group_names, race_group_names])
    else:
        raise ValueError('hier must be specified!')
    
    to_leave_alone = set(feature_names) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(feature_names) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(feature_names) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', StandardScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X = col_transf.fit_transform(X)
    dataset = Dataset(task + state, X, y, groups, group_names, tree)
    return dataset

def preprocess_folkstates(task, sens, path='data/'):
    df = pd.read_csv(os.path.join(path, 'states.csv'))
    if task == 'income':
        ACSIncomeNew = ACSIncome
        ACSIncomeNew.features.extend(['DIV', 'REG', 'ST'])
        X, y, group = ACSIncomeNew.df_to_numpy(df)
        feature_names = ACSIncomeNew.features
        to_one_hot = set(INCOME_CATGEORICAL + ['DIV', 'REG', 'ST'])
    elif task == 'coverage':
        ACSPublicCoverageNew = ACSPublicCoverage
        ACSPublicCoverageNew.features.extend(['DIV', 'REG', 'ST'])
        X, y, group = ACSPublicCoverageNew.df_to_numpy(df)
        feature_names = ACSPublicCoverageNew.features
        to_one_hot = set(COVERAGE_CATEGORICAL + ['DIV', 'REG', 'ST'])
    elif task == 'employment':
        ACSEmploymentNew = ACSEmployment
        ACSEmploymentNew.features.extend(['DIV', 'REG', 'ST'])
        X, y, group = ACSEmploymentNew.df_to_numpy(df)
        feature_names = ACSEmploymentNew.features
        to_one_hot = set(EMPLOYMENT_CATEGORICAL + ['DIV', 'REG', 'ST'])

    sex_idx = feature_names.index('SEX')
    st_idx = feature_names.index('ST')
    reg_idx = feature_names.index('REG')
    div_idx = feature_names.index('DIV')

    sex = X[:, sex_idx]
    state = X[:, st_idx]
    region = X[:, reg_idx]
    div = X[:, div_idx]

    region_groups = []
    for val in region_vals:
        region_groups.append(region == val)

    div_groups = []
    for val in division_vals:
        div_groups.append(div == val)

    state_groups = []
    for val in state_vals:
        state_groups.append(state == val)

    ALL = [True] * y.shape[0]
    race_groups = []
    # Get race groups, combining (R3, R4, R5) and (R6, R7)
    for g in np.unique(group):
        if g == 3:  # R3 (American-Indian)
            R34 = np.logical_or(group == 3, group == 4)
            race_groups.append(np.logical_or(R34, group == 5))
        elif g == 6: # R6 (Asian)
            race_groups.append(np.logical_or(group == 6, group == 7))
        elif g == 4 or g == 5 or g == 7: # group is too small
            continue
        else:
            race_groups.append(group == g)
    race_group_names = ["R1", "R2", "R3+", "R6+", "R7", "R8", "R9"]
    sex_groups = [sex == 1, sex == 2]
    sex_group_names = ["M", "F"]

    if sens == 'sex':
        groups, group_names, tree = construct_hier([[ALL], region_groups,
                                                    div_groups, state_groups, sex_groups],
                                                    [["ALL"], region_names, division_names, state_names, sex_group_names], skip_empty=True)
    elif sens == 'race':
        groups, group_names, tree = construct_hier([[ALL], region_groups,
                                                    div_groups, state_groups, race_groups],
                                                    [["ALL"], region_names, division_names, state_names, race_group_names], skip_empty=True)
    else:
        raise ValueError("Invalid sens attribute={}".format(sens))

    
    to_leave_alone = set(feature_names) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(feature_names) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(feature_names) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', StandardScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X = col_transf.fit_transform(X)
    dataset = Dataset(task + "ST", X, y, groups, group_names, tree)
    return dataset

def name_to_dataset(dataset):
    """
    Takes a dataset name and outputs the preprocessed dataset as a Dataset object with X, y, groups, intersections, group_names, and inter_names.
    """
    splits = dataset.split("_")

    if dataset == 'adult':
        dataset = preprocess_adult()
    elif dataset == 'compas':
        dataset = preprocess_compas()
    elif dataset == 'communities':
        dataset = preprocess_communities()
    elif dataset == 'german':
        dataset = preprocess_german()
    elif len(splits) == 3 and splits[1] == 'ST':
        task = splits[0]
        sens = splits[2]
        dataset = preprocess_folkstates(task, sens)
    elif len(splits) == 3 and splits[1] != 'ST':
        task = splits[0]
        state = splits[1]
        hier = splits[2]
        dataset = preprocess_folktables(task, state, hier)
    else:
        raise ValueError("Dataset: {} is not valid!".format(dataset))
    return dataset