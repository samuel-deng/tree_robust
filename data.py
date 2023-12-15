import pandas as pd
import numpy as np
import itertools

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from compas_utils import preprocess_compas_df
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage
from models import construct_tree

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
    def __init__(self, name, X, y, groups, intersections, 
                 group_names, inter_names, tree):
        self.name = name
        self.X = X
        self.y = y
        self.groups = groups
        self.intersections = intersections
        self.group_names = group_names
        self.inter_names = inter_names
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

def gen_group_pairs(num_groups, folktables=False):
    erm_group_pairs = list()
    group_pairs = list()
    group_pairs_cond = list()

    if num_groups == 9:
        # append G0 (ALL) paired with all other groups
        for i in range(1, num_groups):
            erm_group_pairs.append((0, i))

        # append pairs intersecting at G1 (W,M)
        group_pairs.append((1,5))
        group_pairs_cond.append(1)
        group_pairs.append((5,7))
        group_pairs_cond.append(1)
        group_pairs.append((1,7))
        group_pairs_cond.append(1)

        # append pairs intersecting at G2 (W,F)
        group_pairs.append((2,5))
        group_pairs_cond.append(2)
        group_pairs.append((5,8))
        group_pairs_cond.append(2)
        group_pairs.append((2,8))
        group_pairs_cond.append(2)

        # append pairs intersecting at G3 (nW,M)
        group_pairs.append((3,6))
        group_pairs_cond.append(3)
        group_pairs.append((6,7))
        group_pairs_cond.append(3)
        group_pairs.append((3,7))
        group_pairs_cond.append(3)

        # append pairs intersecting at G4 (nW,F)
        group_pairs.append((4,6))
        group_pairs_cond.append(4)
        group_pairs.append((6,8))
        group_pairs_cond.append(4)
        group_pairs.append((4,8))
        group_pairs_cond.append(4)

    # For Folktables datasets
    elif folktables:
        # append G0 (ALL) paired with all other groups
        for i in range(1, num_groups):
            erm_group_pairs.append((0, i))

        # Race-Sex Group Intersections
        race_sex_pairs = list()
        race_sex_cond = list()
        for r in range(1, 8):
            race_sex_pairs.append((r, 8)) # (race, S1)
            race_sex_cond.append(11 + (2 * r - 1))
            race_sex_pairs.append((r, 9)) # (race, S2)
            race_sex_cond.append(12 + (2 * r - 1))

        # Race-Age Group Intersections
        race_age_pairs = list()
        race_age_cond = list()
        for r in range(1, 8):
            race_age_pairs.append((r, 10)) # (race, A1)
            race_age_cond.append(25 + (2 * r - 1))
            race_age_pairs.append((r, 11)) # (race, A2)
            race_age_cond.append(26 + (2 * r - 1))

        # Sex-Age Group Intersections
        sex_age_pairs = list()
        sex_age_cond = list()
        sex_age_pairs.append((8, 10))
        sex_age_cond.append(40)
        sex_age_pairs.append((8, 11))
        sex_age_cond.append(41)
        sex_age_pairs.append((9, 10))
        sex_age_cond.append(42)
        sex_age_pairs.append((9, 11))
        sex_age_cond.append(43)

        group_pairs = (race_sex_pairs, race_age_pairs, sex_age_pairs)
        group_pairs_cond = (race_sex_cond, race_age_cond, sex_age_cond)

    return erm_group_pairs, group_pairs, group_pairs_cond

def construct_hier(groups, group_names):
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

def preprocess_employment(year='2016', horizon='1-Year', states=["CA"],
                          hier=None):
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    X, y, group = ACSEmployment.df_to_numpy(acs_data)
    sex = X[:, -2]
    old = (X[:,0] > 65)

    # 12 groups (including ALL)
    ALL = [True] * y.shape[0]
    race_groups = []
    for g in np.unique(group):
        if g == 4 or g == 5: # group is too small
            continue
        race_groups.append(group == g)
    race_group_names = ["R1", "R2", "R3", "R6", "R7", "R8", "R9"]
    sex_groups = [sex == 1, sex == 2]
    sex_group_names = ["M", "F"]
    age_groups = [old == False, old == True]
    age_group_names = ["Y", "O"]

    if hier == 'RACE-SEX':
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    sex_groups, age_groups],[["ALL"], race_group_names, sex_group_names,age_group_names])
    elif hier == 'RACE-AGE':       
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    age_groups, sex_groups],[["ALL"], race_group_names, age_group_names, sex_group_names])
    elif hier == 'AGE-SEX':
        groups, group_names, tree = construct_hier([[ALL], age_groups,
                                                    sex_groups, race_groups],[["ALL"], age_group_names, sex_group_names, race_group_names])
    elif hier == None:
        groups = []
        group_names = ["ALL", "R1", "R2", "R3", "R6", "R7", "R8", "R9", 
                "M", "F", "Y", "O"]
        for g in np.unique(group):
                if g == 4 or g == 5: # group is too small
                    continue
                groups.append(group == g)
        groups.append(sex == 1) # index: 8
        groups.append(sex == 2) # index: 9
        groups.append(old == False) # index: 10
        groups.append(old == True) # index: 11
    else:
        raise ValueError('hier must be RACE-SEX, RACE-AGE, or AGE-SEX!')

    # Additional groups for pair-wise intersections
    intersections = []
    race_sex_intersections = []
    race_age_intersections = []
    sex_age_intersections = []
    inter_names = []

    for r in range(1, 8):
        inter_names.append("({},{})".format(group_names[r], "M"))
        intersections.append((r, 8))
        race_sex_intersections.append((r,8))
        inter_names.append("({},{})".format(group_names[r], "F"))
        intersections.append((r, 9))
        race_sex_intersections.append((r,9))

    for r in range(1, 8):
        inter_names.append("({},{})".format(group_names[r], "Y"))
        intersections.append((r, 10))
        race_age_intersections.append((r,10))
        inter_names.append("({},{})".format(group_names[r], "O"))
        intersections.append((r, 11))
        race_age_intersections.append((r,11))

    for s in range(8, 10):
        for a in range(10, 12):
            inter_names.append("({},{})".format(
                group_names[s], group_names[a]))
            intersections.append((s, a))
            sex_age_intersections.append((s,a))

    to_one_hot = set(['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'DREM', 'RAC1P'])
    to_leave_alone = set(ACSEmployment.features) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(ACSEmployment.features) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(ACSEmployment.features) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', StandardScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X = col_transf.fit_transform(X)
    name = "employment{}".format(states[0])
    dataset = Dataset(name, X, y, groups, intersections, 
                      group_names, inter_names, tree)
    return dataset

def preprocess_income(year='2016', horizon='1-Year', states=["CA"], hier=False):
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    X, y, group = ACSIncome.df_to_numpy(acs_data)
    sex = X[:, -2]
    old = (X[:,0] > 65)

    # 12 groups (including ALL)
    ALL = [True] * y.shape[0]
    race_groups = []
    for g in np.unique(group):
        if g == 4 or g == 5: # group is too small
            continue
        race_groups.append(group == g)
    race_group_names = ["R1", "R2", "R3", "R6", "R7", "R8", "R9"]
    sex_groups = [sex == 1, sex == 2]
    sex_group_names = ["M", "F"]
    age_groups = [old == False, old == True]
    age_group_names = ["Y", "O"]

    if hier == 'RACE-SEX':
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    sex_groups, age_groups],[["ALL"], race_group_names, sex_group_names,age_group_names])
    elif hier == 'RACE-AGE':       
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    age_groups, sex_groups],[["ALL"], race_group_names, age_group_names, sex_group_names])
    elif hier == 'AGE-SEX':
        groups, group_names, tree = construct_hier([[ALL], age_groups,
                                                    sex_groups, race_groups],[["ALL"], age_group_names, sex_group_names, race_group_names])
    elif hier == None:
        groups = []
        group_names = ["ALL", "R1", "R2", "R3", "R6", "R7", "R8", "R9", 
                "M", "F", "Y", "O"]
        for g in np.unique(group):
                if g == 4 or g == 5: # group is too small
                    continue
                groups.append(group == g)
        groups.append(sex == 1) # index: 8
        groups.append(sex == 2) # index: 9
        groups.append(old == False) # index: 10
        groups.append(old == True) # index: 11
    else:
        raise ValueError('hier must be RACE-SEX, RACE-AGE, or AGE-SEX!')

    # Additional groups for pair-wise intersections
    intersections = []
    race_sex_intersections = []
    race_age_intersections = []
    sex_age_intersections = []
    inter_names = []

    for r in range(1, 8):
        inter_names.append("({},{})".format(group_names[r], "M"))
        intersections.append((r, 8))
        race_sex_intersections.append((r,8))
        inter_names.append("({},{})".format(group_names[r], "F"))
        intersections.append((r, 9))
        race_sex_intersections.append((r,9))

    for r in range(1, 8):
        inter_names.append("({},{})".format(group_names[r], "Y"))
        intersections.append((r, 10))
        race_age_intersections.append((r,10))
        inter_names.append("({},{})".format(group_names[r], "O"))
        intersections.append((r, 11))
        race_age_intersections.append((r,11))

    for s in range(8, 10):
        for a in range(10, 12):
            inter_names.append("({},{})".format(
                group_names[s], group_names[a]))
            intersections.append((s, a))
            sex_age_intersections.append((s,a))

    to_one_hot = set(['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P'])
    to_leave_alone = set(ACSIncome.features) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(ACSIncome.features) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(ACSIncome.features) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', StandardScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X = col_transf.fit_transform(X)
    name = "income{}".format(states[0])
    dataset = Dataset(name, X, y, groups, intersections, 
                      group_names, inter_names, tree)
    return dataset

def preprocess_coverage(year='2016', horizon='1-Year', states=['CA'],
                        hier=None):
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    X, y, group = ACSPublicCoverage.df_to_numpy(acs_data)
    old = (X[:,0] > 32)
    sex = X[:, 3]

    ALL = [True] * y.shape[0]
    race_groups = []
    for g in np.unique(group):
        if g == 4 or g == 5: # group is too small
            continue
        race_groups.append(group == g)
    race_group_names = ["R1", "R2", "R3", "R6", "R7", "R8", "R9"]
    sex_groups = [sex == 1, sex == 2]
    sex_group_names = ["M", "F"]
    age_groups = [old == False, old == True]
    age_group_names = ["Y", "O"]

    if hier == 'RACE-SEX':
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    sex_groups, age_groups],[["ALL"], race_group_names, sex_group_names,age_group_names])
    elif hier == 'RACE-AGE':       
        groups, group_names, tree = construct_hier([[ALL], race_groups,
                                                    age_groups, sex_groups],[["ALL"], race_group_names, age_group_names, sex_group_names])
    elif hier == 'AGE-SEX':
        groups, group_names, tree = construct_hier([[ALL], age_groups,
                                                    sex_groups, race_groups],[["ALL"], age_group_names, sex_group_names, race_group_names])
    elif hier == None:
        groups = []
        group_names = ["ALL", "R1", "R2", "R3", "R6", "R7", "R8", "R9", 
                "M", "F", "Y", "O"]
        for g in np.unique(group):
                if g == 4 or g == 5: # group is too small
                    continue
                groups.append(group == g)
        groups.append(sex == 1) # index: 8
        groups.append(sex == 2) # index: 9
        groups.append(old == False) # index: 10
        groups.append(old == True) # index: 11
    else:
        raise ValueError('hier must be RACE-SEX, RACE-AGE, or AGE-SEX!')

    # Pair-wise intersections
    intersections = []
    race_sex_intersections = []
    race_age_intersections = []
    sex_age_intersections = []
    inter_names = []

    for r in range(1, 8):
        inter_names.append("({},{})".format(group_names[r], "M"))
        intersections.append((r, 8))
        race_sex_intersections.append((r,8))
        inter_names.append("({},{})".format(group_names[r], "F"))
        intersections.append((r, 9))
        race_sex_intersections.append((r,9))

    for r in range(1, 8):
        inter_names.append("({},{})".format(group_names[r], "Y"))
        intersections.append((r, 10))
        race_age_intersections.append((r,10))
        inter_names.append("({},{})".format(group_names[r], "O"))
        intersections.append((r, 11))
        race_age_intersections.append((r,11))

    for s in range(8, 10):
        for a in range(10, 12):
            inter_names.append("({},{})".format(
                group_names[s], group_names[a]))
            intersections.append((s, a))
            sex_age_intersections.append((s,a))

    to_one_hot = set(['ESP', 'MAR', 'CIT', 'MIG', 'MIL', 
                    'ANC', 'ESR', 'ST', 'RAC1P'])
    to_leave_alone = set(ACSPublicCoverage.features) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(ACSPublicCoverage.features) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(ACSPublicCoverage.features) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', StandardScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X = col_transf.fit_transform(X)
    name = "coverage{}".format(states[0])
    dataset = Dataset(name, X, y, groups, intersections, 
                        group_names, inter_names, tree)
    return dataset

def name_to_dataset(dataset):
    """
    Takes a dataset name and outputs the preprocessed dataset as a Dataset object with X, y, groups, intersections, group_names, and inter_names.
    """
    if dataset == 'adult':
        dataset = preprocess_adult()
    elif dataset == 'compas':
        dataset = preprocess_compas()
    elif dataset == 'communities':
        dataset = preprocess_communities()
    elif dataset == 'german':
        dataset = preprocess_german()

    elif dataset == 'employmentCA':
        dataset = preprocess_employment()
    elif dataset == 'employmentCA_rsa':
        dataset = preprocess_employment(hier='RACE-SEX')
    elif dataset == 'employmentCA_ras':
        dataset = preprocess_employment(hier='RACE-AGE')
    elif dataset == 'employmentCA_asr':
        dataset = preprocess_employment(hier='AGE-SEX')

    elif dataset == 'employmentNY':
        dataset = preprocess_employment(states=['NY'])
    elif dataset == 'employmentNY_rsa':
        dataset = preprocess_employment(states=['NY'], hier='RACE-SEX')
    elif dataset == 'employmentNY_ras':
        dataset = preprocess_employment(states=['NY'], hier='RACE-AGE')
    elif dataset == 'employmentNY_asr':
        dataset = preprocess_employment(states=['NY'], hier='AGE-SEX')

    elif dataset == 'employmentTX':
        dataset = preprocess_employment(states=['TX'])
    elif dataset == 'employmentTX_rsa':
        dataset = preprocess_employment(states=['TX'], hier='RACE-SEX')
    elif dataset == 'employmentTX_ras':
        dataset = preprocess_employment(states=['TX'], hier='RACE-AGE')
    elif dataset == 'employmentTX_asr':
        dataset = preprocess_employment(states=['TX'], hier='AGE-SEX')

    elif dataset == 'incomeCA':
        dataset = preprocess_income()
    elif dataset == 'incomeCA_rsa':
        dataset = preprocess_income(hier='RACE-SEX')
    elif dataset == 'incomeCA_ras':
        dataset = preprocess_income(hier='RACE-AGE')
    elif dataset == 'incomeCA_asr':
        dataset = preprocess_income(hier='AGE-SEX')

    elif dataset == 'incomeNY':
        dataset = preprocess_income(states=['NY'])
    elif dataset == 'incomeNY_rsa':
        dataset = preprocess_income(states=['NY'],hier='RACE-SEX')
    elif dataset == 'incomeNY_ras':
        dataset = preprocess_income(states=['NY'],hier='RACE-AGE')
    elif dataset == 'incomeNY_asr':
        dataset = preprocess_income(states=['NY'],hier='AGE-SEX')
    
    elif dataset == 'incomeTX':
        dataset = preprocess_income(states=['TX'])
    elif dataset == 'incomeTX_rsa':
        dataset = preprocess_income(states=['TX'],hier='RACE-SEX')
    elif dataset == 'incomeTX_ras':
        dataset = preprocess_income(states=['TX'],hier='RACE-AGE')
    elif dataset == 'incomeTX_asr':
        dataset = preprocess_income(states=['TX'],hier='AGE-SEX')

    elif dataset == 'coverageCA':
        dataset = preprocess_coverage()
    elif dataset == 'coverageCA_rsa':
        dataset = preprocess_coverage(hier='RACE-SEX')
    elif dataset == 'coverageCA_ras':
        dataset = preprocess_coverage(hier='RACE-AGE')
    elif dataset == 'coverageCA_asr':
        dataset = preprocess_coverage(hier='AGE-SEX')

    elif dataset == 'coverageNY':
        dataset = preprocess_coverage(states=['NY'])
    elif dataset == 'coverageNY_rsa':
        dataset = preprocess_coverage(states=['NY'],hier='RACE-SEX')
    elif dataset == 'coverageNY_ras':
        dataset = preprocess_coverage(states=['NY'],hier='RACE-AGE')
    elif dataset == 'coverageNY_asr':
        dataset = preprocess_coverage(states=['NY'],hier='AGE-SEX')

    elif dataset == 'coverageTX':
        dataset = preprocess_coverage(states=['TX'])
    elif dataset == 'coverageTX_rsa':
        dataset = preprocess_coverage(states=['TX'],hier='RACE-SEX')
    elif dataset == 'coverageTX_ras':
        dataset = preprocess_coverage(states=['TX'],hier='RACE-AGE')
    elif dataset == 'coverageTX_asr':
        dataset = preprocess_coverage(states=['TX'],hier='AGE-SEX')
    else:
        raise ValueError("Dataset: {} is not valid!".format(dataset))
    return dataset



    # 12 groups (including ALL)
    '''
    # Race Groups
    group_names = ["ALL", "R1", "R2", "R3", "R6", "R7", "R8", "R9"]
    tree.append([])
    for g in np.unique(group): # indices: 1 -> 7
        if g == 4 or g == 5: # group is too small
            continue
        groups.append(group == g)
        tree[1].append(len(groups) - 1)

    # Race-Sex Intersections
    for i, r in enumerate(groups[1:]):
        groups.append(r & A1)
        group_names.append("({},{})".format(group_names[i + 1], "Y"))
        tree[2].append(len(groups) - 1)
        groups.append(r & A2)
        group_names.append("({},{})".format(group_names[i + 1], "O"))
        tree[2].append(len(groups) - 1)

    # Race-Sex-Age Intersections
    tree.append([])
    for i in tree[2]:
        groups.append(S1 & groups[i])
        group_names.append("({},{})".format(group_names[i], "M"))
        tree[3].append(len(groups) - 1)
        groups.append(S2 & groups[i])
        group_names.append("({},{})".format(group_names[i], "F"))
        tree[3].append(len(groups) - 1)
    '''