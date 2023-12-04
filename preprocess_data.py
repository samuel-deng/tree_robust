import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from compas_utils import preprocess_compas_df
from folktables import ACSDataSource, ACSEmployment, ACSIncome

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

def preprocess_adult(train_path='datasets/adult/adult.data',
                     test_path='datasets/adult/adult.test'):
    adult_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                    "marital-status", "occupation", "relationship", "race", "sex",
                    "capital-gain", "capital-loss", "hours-per-week", "native-country", 
                    "income"]
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

    # Fit the ColumnTransformer to X
    X_transf = col_transf.fit_transform(X)
    return X, y, col_transf, group_memberships

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
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', MinMaxScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variale so we have two classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)
    print("% examples (y=1): {}".format(100 * len(np.where(y == 1)[0])/len(y)))
    print("% examples (y=0): {}".format(100 * len(np.where(y == 0)[0])/len(y)))

    print("Communities and Crime Shape: {}".format(data.shape))

    group_names = ["ALL", "W,H", "W,L", "nW,H", "nW,L", "W", "nW", "H", "L"]
    group_memberships = []
    group_memberships.append([True] * y.shape[0])
    race_gps_coarse = ["White", "NotWhite"]
    income_gps = [1, 0]

    # Traditional disjoint groups
    for race in race_gps_coarse:
        for income in income_gps:
            indices = communities_gp_indices(X, race, income)[0]
            membership = np.zeros(y.shape[0], dtype=bool)
            membership[indices] = True
            group_memberships.append(membership)

    # Add 4 overlapping groups
    w_indices = np.where(X['race'] == 1)
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    group_memberships.append(w_membership)

    nw_indices = np.where(X['race'] == 0)
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    group_memberships.append(nw_membership)

    m_indices = np.where(X['income_level'] == 1)
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    group_memberships.append(m_membership)

    f_indices = np.where(X['income_level'] == 0)
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    group_memberships.append(f_membership)

    num_groups = len(group_memberships)
    print('num_groups = {0}'.format(num_groups))

    # Fit the ColumnTransformer to X
    X_transf = col_transf.fit_transform(X)
    print("Column-transformed X has shape: {}".format(X_transf.shape))

    return X, y, col_transf, group_memberships

def preprocess_compas(train_path='datasets/compas/compas.csv'):
    # COMPAS Dataset
    compas_data = pd.read_csv(train_path, header=0, na_values='?')
    compas_df = preprocess_compas_df(compas_data)
    compas_df = compas_df.dropna()
    print("COMPAS Shape: {}".format(compas_df.shape))

    X, y = compas_df.drop("is_recid", axis=1), compas_df["is_recid"]
    cat_idx = ['c_charge_degree', 'sex', 'race', 'screening_year_is_2013']
    num_idx = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'age']
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', MinMaxScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variale so we have two classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)
    print("% examples recidivate (y=1): {}".format(100 * len(np.where(y == 1)[0])/len(y)))
    print("% examples NOT recidivate (y=0): {}".format(100 * len(np.where(y == 0)[0])/len(y)))

    group_names = ["ALL", "W,M", "W,F", "nW,M", "nW,F", "W", "nW", "M", "F"]
    group_memberships = []
    group_memberships.append([True] * y.shape[0])
    race_gps_coarse = ["White", "NotWhite"]
    sex_gps = [1, 0]

    # Traditional disjoint groups
    for race in race_gps_coarse:
        for sex in sex_gps:
            indices = compas_gp_indices(X, race, sex)[0]
            membership = np.zeros(y.shape[0], dtype=bool)
            membership[indices] = True
            group_memberships.append(membership)

    # Add 4 overlapping groups
    w_indices = np.where(X['race'] == 1)
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    group_memberships.append(w_membership)

    nw_indices = np.where(X['race'] != 1)
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    group_memberships.append(nw_membership)

    m_indices = np.where(X['sex'] == 1)
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    group_memberships.append(m_membership)

    f_indices = np.where(X['sex'] == 0)
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    group_memberships.append(f_membership)

    num_groups = len(group_memberships)
    print('num_groups = {0}'.format(num_groups))

    # Fit the ColumnTransformer to X
    X_transf = col_transf.fit_transform(X)
    return X, y, col_transf, group_memberships

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
    steps = [('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx), ('num', MinMaxScaler(), num_idx)]
    col_transf = ColumnTransformer(steps)

    # label encoder to target variale so we have two classes 0 and 1
    assert(len(np.unique(y)) == 2)
    y = LabelEncoder().fit_transform(y)
    print("% examples (y=1): {}".format(100 * len(np.where(y == 1)[0])/len(y)))
    print("% examples (y=0): {}".format(100 * len(np.where(y == 0)[0])/len(y)))

    print("German Credit Shape: {}".format(df.shape))

    group_names = ["ALL", "M,O", "F,Y", "M,O", "F,Y", "M", "F", "O", "Y"]
    group_memberships = []
    group_memberships.append([True] * y.shape[0])
    race_gps_coarse = ["Male", "Female"]
    age_gps = [1, 0]

    # Traditional disjoint groups
    for race in race_gps_coarse:
        for age in age_gps:
            indices = german_gp_indices(X, race, age)[0]
            membership = np.zeros(y.shape[0], dtype=bool)
            membership[indices] = True
            group_memberships.append(membership)

    # Add 4 overlapping groups
    w_indices = np.where(X['sex'] == 1)
    w_membership = np.zeros(y.shape[0], dtype=bool)
    w_membership[w_indices] = True
    group_memberships.append(w_membership)

    nw_indices = np.where(X['sex'] == 0)
    nw_membership = np.zeros(y.shape[0], dtype=bool)
    nw_membership[nw_indices] = True
    group_memberships.append(nw_membership)

    m_indices = np.where(X['age'] == 1)
    m_membership = np.zeros(y.shape[0], dtype=bool)
    m_membership[m_indices] = True
    group_memberships.append(m_membership)

    f_indices = np.where(X['age'] == 0)
    f_membership = np.zeros(y.shape[0], dtype=bool)
    f_membership[f_indices] = True
    group_memberships.append(f_membership)

    num_groups = len(group_memberships)
    print('num_groups = {0}'.format(num_groups))

    # Fit the ColumnTransformer to X
    X_transf = col_transf.fit_transform(X)
    print("Column-transformed X has shape: {}".format(X_transf.shape))
    return X, y, col_transf, group_memberships

def preprocess_employment(year='2016', horizon='1-Year', states=["CA"]):
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    X, y, group = ACSEmployment.df_to_numpy(acs_data)
    sex = X[:, -2]
    old = (X[:,0] > 65)

    # 12 groups (including ALL)
    group_memberships = []
    group_memberships.append([True] * y.shape[0]) # index: 0
    for g in np.unique(group): # indices: 1 -> 7
        if g == 4 or g == 5: # group is too small
            continue
        group_memberships.append(group == g)
    group_memberships.append(sex == 1) # index: 8
    group_memberships.append(sex == 2) # index: 9
    group_memberships.append(old == False) # index: 10
    group_memberships.append(old == True) # index: 11
    num_groups = len(group_memberships)
    print('num_groups = {0}'.format(num_groups))

    # Additional groups for pair-wise intersections
    group_members_all = group_memberships.copy()

    # Race-Sex Intersections
    # indices: 12 -> 25
    for r in range(1, 8):
        group_members_all.append(group_memberships[r] & group_memberships[8])
        group_members_all.append(group_memberships[r] & group_memberships[9])
    
    # Race-Age Intersections
    # indices: 26 -> 39
    for r in range(1, 8):
        group_members_all.append(group_memberships[r] & group_memberships[10])
        group_members_all.append(group_memberships[r] & group_memberships[11])

    # Sex-Age Intersections
    group_members_all.append(group_memberships[8] & group_memberships[10]) # 40
    group_members_all.append(group_memberships[8] & group_memberships[11]) # 41
    group_members_all.append(group_memberships[9] & group_memberships[10]) # 42
    group_members_all.append(group_memberships[9] & group_memberships[11]) # 43

    # Sex-Age-Race Intersections
    for r in range(1, 8):
        group_members_all.append(group_memberships[r] & group_members_all[40])
        group_members_all.append(group_memberships[r] & group_members_all[41])
        group_members_all.append(group_memberships[r] & group_members_all[42])
        group_members_all.append(group_memberships[r] & group_members_all[43])

    to_one_hot = set(['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'DREM', 'RAC1P'])
    to_leave_alone = set(ACSEmployment.features) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(ACSEmployment.features) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(ACSEmployment.features) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', MinMaxScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X_transf = col_transf.fit_transform(X).toarray()
    print("Column-transformed X has shape: {}".format(X_transf.shape))
    return X, y, col_transf, group_memberships, group_members_all

def preprocess_income(year='2016', horizon='1-Year', states=["CA"]):
    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    X, y, group = ACSIncome.df_to_numpy(acs_data)
    sex = X[:, -2]
    old = (X[:,0] > 65)
    print("ACS Income Features: {}".format(ACSIncome.features))
    print("ACS Income Shape {}".format(X.shape))

    group_names = []
    group_memberships = []
    group_memberships.append([True] * y.shape[0])
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

    # Additional groups for pair-wise intersections
    group_members_all = group_memberships.copy()

    # Race-Sex Intersections
    # indices: 12 -> 25
    for r in range(1, 8):
        group_members_all.append(group_memberships[r] & group_memberships[8])
        group_members_all.append(group_memberships[r] & group_memberships[9])
    
    # Race-Age Intersections
    # indices: 26 -> 39
    for r in range(1, 8):
        group_members_all.append(group_memberships[r] & group_memberships[10])
        group_members_all.append(group_memberships[r] & group_memberships[11])

    # Sex-Age Intersections
    group_members_all.append(group_memberships[8] & group_memberships[10]) # 40
    group_members_all.append(group_memberships[8] & group_memberships[11]) # 41
    group_members_all.append(group_memberships[9] & group_memberships[10]) # 42
    group_members_all.append(group_memberships[9] & group_memberships[11]) # 43

    # Sex-Age-Race Intersections
    for r in range(1, 8):
        group_members_all.append(group_memberships[r] & group_members_all[40])
        group_members_all.append(group_memberships[r] & group_members_all[41])
        group_members_all.append(group_memberships[r] & group_members_all[42])
        group_members_all.append(group_memberships[r] & group_members_all[43])

    to_one_hot = set(['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P'])
    to_leave_alone = set(ACSIncome.features) - to_one_hot
    one_hot_inds = [i for i, x in enumerate(ACSIncome.features) if x in to_one_hot]
    leave_alone_inds = [i for i, x in enumerate(ACSIncome.features) if x in to_leave_alone]

    steps = [('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_inds), ('num', MinMaxScaler(), leave_alone_inds)]
    col_transf = ColumnTransformer(steps)
    X_transf = col_transf.fit_transform(X).toarray()
    print("Column-transformed X has shape: {}".format(X_transf.shape))
    return X, y, col_transf, group_memberships, group_members_all