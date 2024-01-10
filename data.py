import pandas as pd
import numpy as np
import os
import itertools

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from compas_utils import preprocess_compas_df
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSTravelTime

INCOME_CATGEORICAL = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']
COVERAGE_CATEGORICAL = ['ESP', 'MAR', 'CIT', 'MIG', 'MIL', 
                      'ANC', 'ESR', 'ST', 'RAC1P']
EMPLOYMENT_CATEGORICAL = ['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
                          'DREM', 'RAC1P']
TRAVEL_CATEGORICAL = ['MAR', 'ESP', 'MIG', 'RELP', 'RAC1P', 'PUMA', 'ST',
                      'CIT', 'OCCP', 'JWTR', 'POWPUMA']

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

def preprocess_adult(hier, path='datasets/adult/adult_reconstruction.csv'):
    df = pd.read_csv(path, header=0, na_values='?')
    pd.to_numeric(df['income'])
    df['label'] = df['income'] >= 50000
    X, y = df.drop(['income', 'label'], axis=1), df['label']

    # Get categorical and numerical features
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', StandardScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # Group logic
    ALL = [True] * y.shape[0]
    EDU_PREHS = ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
                 'Preschool']
    EDU_HS = ['HS-grad', 'Assoc-acdm', 'Assoc-voc', 'Prof-school'] 
    EDU_COL= ['Bachelors', 'Some-college']
    EDU_MASTERS = ['Masters']
    EDU_PHD = ['Doctorate']

    # Marriage
    mar = np.array(X['marital-status'] == 'Married-civ-spouse')
    nmar = np.array(X['marital-status'] != 'Married-civ-spouse')
    marriage_group_names = ['Mar', 'nMar']
    marriage_groups = [mar, nmar]

    # Age
    young = np.array(X['age'] <= 35)
    mid = np.array((X['age'] > 35) & (X['age'] <= 50))
    old = np.array(X['age'] > 50)
    age_group_names = ['Ya', 'Ma', 'Oa']
    age_groups = [young, mid, old]

    # Education
    edu_prehs = np.array(X['education'].isin(EDU_PREHS))
    edu_hs = np.array(X['education'].isin(EDU_HS))
    edu_col = np.array(X['education'].isin(EDU_COL))
    edu_masters = np.array(X['education'].isin(EDU_MASTERS))
    edu_phd = np.array(X['education'].isin(EDU_PHD))

    edu_groups = [edu_prehs, edu_hs, edu_col, edu_masters, edu_phd]
    edu_group_names = ['PreHS', 'HS', 'COL', 'MAS', 'DOC']

    # Race groups
    race_aie = np.array(X['race'] == 'Amer-Indian-Eskimo')
    race_api = np.array(X['race'] == 'Asian-Pac-Islander')
    race_b = np.array(X['race'] == 'Black')
    race_o = np.array(X['race'] == 'Other')
    race_w = np.array(X['race'] == 'White')
    race_groups = [race_aie, race_api, race_b, race_o, race_w]
    race_group_names = ["AIE", "API", "B", "O", "W"]

    # Sex groups
    sex_groups = [np.array(X['gender'] == 'Male'), 
                np.array(X['gender'] == 'Female')]
    sex_group_names = ['M', 'F']

    groups, group_names, tree = construct_hier([[ALL], sex_groups, race_groups],
                                            [["ALL"], sex_group_names, race_group_names])
    
    if hier == 'se':
        groups, group_names, tree = construct_hier([[ALL], sex_groups,
                                                    edu_groups],
                                                    [["ALL"], sex_group_names, edu_group_names])
    elif hier == 'sa':
        groups, group_names, tree = construct_hier([[ALL], sex_groups,
                                                    age_groups],
                                                    [["ALL"], sex_group_names, age_group_names])
    elif hier == 'ae':
        groups, group_names, tree = construct_hier([[ALL], age_groups,
                                                    edu_groups],
                                                    [["ALL"], age_group_names, edu_group_names])
    elif hier == 're':
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                            edu_groups],
                                            [["ALL"], race_group_names, edu_group_names])
    elif hier == 'me':
        groups, group_names, tree = construct_hier([[ALL], marriage_groups,
                                            edu_groups],
                                            [["ALL"], marriage_group_names, edu_group_names])
    else:
        raise ValueError("hier must be specified!")

    # label encoder to target variable so we have classes 0 and 1
    assert(len(np.unique(y)) == 2)
    X = col_transf.fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    dataset = Dataset("adult_" + hier, X, y, groups, group_names, tree)
    return dataset

def preprocess_insurance(hier, path='datasets/insurance/insurance.csv'):
    # Load data and add label
    df = pd.read_csv(path, header=0)
    pd.to_numeric(df['charges'])
    median = df['charges'].median()
    df['label'] = df['charges'] >  median
    X, y = df.drop(["charges", "label"], axis=1), df["label"]

    # Get categorical and numerical features
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', StandardScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # Get groups: AGE, SMOKER, SEX, BMI
    ALL = [True] * y.shape[0]

    young = np.array(X['age'] <= 35)
    mid = np.array((X['age'] > 35) & (X['age'] <= 50))
    old = np.array(X['age'] > 50)
    age_group_names = ['Ya', 'Ma', 'Oa']
    age_groups = [young, mid, old]

    smoker = np.array(X['smoker'] == 'yes')
    nonsmoker = np.array(X['smoker'] == 'no')
    smoker_groups = [smoker, nonsmoker]
    smoker_group_names = ['SMK', 'nSMK']

    bmi1 = np.array(X['bmi'] < 18.5)
    bmi2 = np.array((X['bmi'] >= 18.5) & (X['bmi'] < 25))
    bmi3 = np.array((X['bmi']) >= 25 & (X['bmi'] < 30))
    bmi4 = np.array(X['bmi'] >= 30)
    bmi_groups = [bmi1, bmi2, bmi3, bmi4]
    bmi_group_names = ['BMI1', 'BMI2', 'BMI3', 'BMI4']

    sex_groups = [np.array(X['sex'] == 'male'), np.array(X['sex'] == 'female')]
    sex_group_names = ['M', 'F']

    if hier == 'sm':
        groups, group_names, tree = construct_hier([[ALL], sex_groups,
                                                    smoker_groups],
                                                    [["ALL"], sex_group_names, smoker_group_names])
    elif hier == 'sa':
        groups, group_names, tree = construct_hier([[ALL], sex_groups,
                                                    age_groups],
                                                    [["ALL"], sex_group_names, age_group_names])
    elif hier == 'sb':
        groups, group_names, tree = construct_hier([[ALL], sex_groups,
                                                    bmi_groups],
                                                    [["ALL"], sex_group_names, bmi_group_names])
    elif hier == 'ab':
        groups, group_names, tree = construct_hier([[ALL], age_groups,
                                            bmi_groups],
                                            [["ALL"], age_group_names, bmi_group_names])
    else:
        raise ValueError("hier must be specified!")
    
    # label encoder to target variable so we have classes 0 and 1
    assert(len(np.unique(y)) == 2)
    X = col_transf.fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    dataset = Dataset("insurance_" + hier, X, y, groups, group_names, tree)
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

def preprocess_folktables(task, state, hier, verbose=False):
    """
    Returns a Dataset object given a folktables task, state, and hierarchy of groups.
    """
    data_source = ACSDataSource(survey_year="2016", horizon="1-Year",
                                survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    OLD_AGE = 60
    if task == 'income':
        X, y, group = ACSIncome.df_to_numpy(acs_data)
        feature_names = ACSIncome.features
        to_one_hot = set(INCOME_CATGEORICAL)
    elif task == 'coverage':
        X, y, group = ACSPublicCoverage.df_to_numpy(acs_data)
        feature_names = ACSPublicCoverage.features
        to_one_hot = set(COVERAGE_CATEGORICAL)
        OLD_AGE = 50
    elif task == 'employment':
        X, y, group = ACSEmployment.df_to_numpy(acs_data)
        feature_names = ACSEmployment.features
        to_one_hot = set(EMPLOYMENT_CATEGORICAL)
    elif task == 'travel':
        X, y, group = ACSTravelTime.df_to_numpy(acs_data)
        feature_names = ACSTravelTime.features
        to_one_hot = set(TRAVEL_CATEGORICAL)
    else:
        raise ValueError("Task for folktables invalid!")
    
    # get correct indices for each feature
    sex_idx = feature_names.index('SEX')
    sch_idx = feature_names.index('SCHL')
    age_idx = feature_names.index('AGEP')
    
    # Construct groups
    sex = X[:, sex_idx]
    young = (X[:, age_idx] < 35)
    mid = np.logical_and(X[:,age_idx] >= 35, X[:,age_idx] < OLD_AGE)
    old = (X[:,age_idx] >= OLD_AGE)

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

    # Get stats for each of the desired attributes
    if verbose:
        print("Total n={}".format(X.shape[0]))
        print("=== Stats for Race Groups ===")
        for i, group in enumerate(race_groups):
            print("Group {}={}".format(race_group_names[i], np.sum(group)))
        print("=== Stats for Sex Groups ===")
        for i, group in enumerate(sex_groups):
            print("Group {}={}".format(sex_group_names[i], np.sum(group)))
        print("=== Stats for Age Groups ===")
        for i, group in enumerate(age_groups):
            print("Group {}={}".format(age_group_names[i], np.sum(group)))
        print("=== Stats for Edu Groups ===")
        for i, group in enumerate(school_groups):
            print("Group {}={}".format(school_group_names[i], np.sum(group)))

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
    elif hier == 'rse':
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    sex_groups, school_groups],[["ALL"], race_group_names, sex_group_names, school_group_names])
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
    OLD_AGE = 60 # different for coverage
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
        OLD_AGE = 50
        to_one_hot = set(COVERAGE_CATEGORICAL + ['DIV', 'REG', 'ST'])
    elif task == 'employment':
        ACSEmploymentNew = ACSEmployment
        ACSEmploymentNew.features.extend(['DIV', 'REG', 'ST'])
        X, y, group = ACSEmploymentNew.df_to_numpy(df)
        feature_names = ACSEmploymentNew.features
        to_one_hot = set(EMPLOYMENT_CATEGORICAL + ['DIV', 'REG', 'ST'])

    sex_idx = feature_names.index('SEX')
    age_idx = feature_names.index('AGEP')
    st_idx = feature_names.index('ST')
    reg_idx = feature_names.index('REG')
    div_idx = feature_names.index('DIV')

    sex = X[:, sex_idx]
    age = X[:, age_idx]
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

    young = age < 35
    mid = np.logical_and(age >= 35, age < OLD_AGE)
    old = age >= OLD_AGE

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

    # So we don't have to repeat code
    sens_dict = {
        'sex': [sex_groups, sex_group_names],
        'sexST': [sex_groups, sex_group_names],
        'race': [race_groups, race_group_names],
        'raceST': [race_groups, race_group_names],
        'age': [age_groups, age_group_names],
        'ageST': [age_groups, age_group_names]
    }

    if sens == 'sex' or sens == 'race' or sens == 'age':
        groups, group_names, tree = construct_hier([[ALL], region_groups,
                                                    div_groups, state_groups, sens_dict[sens][0]],
                                                    [["ALL"], region_names, division_names, state_names, sens_dict[sens][1]], skip_empty=True)
    elif sens == 'sexST' or sens == 'raceST' or sens == 'ageST':
        groups, group_names, tree = construct_hier([[ALL], state_groups,
                                                    sens_dict[sens][0]],
                                                    [["ALL"], state_names, sens_dict[sens][1]], skip_empty=True)
    elif sens == 'srs':
        groups, group_names, tree = construct_hier([[ALL], state_groups,
                                                    race_groups, sex_groups],
                                                    [["ALL"], state_names, race_group_names, sex_group_names], skip_empty=True)
    elif sens == 'sra':
        groups, group_names, tree = construct_hier([[ALL], state_groups,
                                                    race_groups, age_groups],
                                                    [["ALL"], state_names, race_group_names, age_group_names], skip_empty=True)
    else:
        raise ValueError("Invalid sens attribute={}".format(sens))
        
    '''
    if sens == 'sex':
        groups, group_names, tree = construct_hier([[ALL], region_groups,
                                                    div_groups, state_groups, sex_groups],
                                                    [["ALL"], region_names, division_names, state_names, sex_group_names], skip_empty=True)
    elif sens == 'race':
        groups, group_names, tree = construct_hier([[ALL], region_groups,
                                                    div_groups, state_groups, race_groups],
                                                    [["ALL"], region_names, division_names, state_names, race_group_names], skip_empty=True)
    elif sens == 'age':
        groups, group_names, tree = construct_hier([[ALL], region_groups,
                                                    div_groups, state_groups, age_groups],
                                                    [["ALL"], region_names, division_names, state_names, age_group_names], skip_empty=True)
    elif sens == 'sexST':
        groups, group_names, tree = construct_hier([[ALL], state_groups,
                                                    sex_groups],
                                                    [["ALL"], state_names, sex_group_names], skip_empty=True)
    elif sens == 'raceST':
        groups, group_names, tree = construct_hier([[ALL], state_groups,
                                                    race_groups],
                                                    [["ALL"], state_names, race_group_names], skip_empty=True)
    '''

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

    if len(splits) == 2 and splits[0] == 'insurance':
        hier = splits[1]
        dataset = preprocess_insurance(hier)
    elif len(splits) == 2 and splits[0] == 'adult':
        hier = splits[1]
        dataset = preprocess_adult(hier)
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