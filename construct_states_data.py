import pandas as pd
import numpy as np
import itertools
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage

state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

region_names = ['NORTHEAST', 'MIDWEST', 'SOUTH', 'WEST']
region_vals = [1, 2, 3, 4]
division_names = ['NewEngland', 'MidAtlantic', 'EastNorthCentral',
                  'WestNorthCentral', 'SouthAtlantic', 'EastSouthCentral',
                  'WestSouthCentral', 'Mountain', 'Pacific']
division_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
state_names = ['MA', 'CT', 'NY', 'PA', 'IL', 'OH', 'MO', 'MN', 'FL', 'GA',
                 'TN', 'AL', 'TX', 'LA', 'AZ', 'CO', 'CA', 'WA']
state_vals = [25, 9, 36, 42, 17, 39, 29, 27, 12, 13, 47, 1, 48, 22, 4, 8, 6, 53]

REGIONS = {
    'NORTHEAST': ['MA', 'CT', 'NY', 'PA'],
    'MIDWEST': ['IL', 'OH', 'MO', 'MN'],
    'SOUTH': ['FL', 'GA', 'TN', 'AL', 'TX', 'LA'],
    'WEST': ['AZ', 'CO', 'CA', 'WA']
}

REGION_CODES = {
    'NORTHEAST': 1,
    'MIDWEST': 2,
    'SOUTH': 3,
    'WEST': 4
}

DIVISIONS = {
    'NewEngland': ['MA', 'CT'],
    'MidAtlantic': ['NY', 'PA'],
    'EastNorthCentral': ['IL', 'OH'],
    'WestNorthCentral': ['MO', 'MN'],
    'SouthAtlantic': ['FL', 'GA'],
    'EastSouthCentral': ['TN', 'AL'],
    'WestSouthCentral': ['TX', 'LA'],
    'Mountain': ['AZ', 'CO'],
    'Pacific': ['CA', 'WA']
}

DIVISION_CODES = {
    'NewEngland': 1,
    'MidAtlantic': 2,
    'EastNorthCentral': 3,
    'WestNorthCentral': 4,
    'SouthAtlantic': 5,
    'EastSouthCentral': 6,
    'WestSouthCentral': 7,
    'Mountain': 8,
    'Pacific': 9
}

_STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
                'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
                'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
                'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
                'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
                'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
                'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
                'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
                'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
                'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
                'PR': '72'}

_STATE_CODES_INV = {
    '1': 'AL',
    '2': 'AK',
    '4': 'AZ',
    '5': 'AR',
    '6': 'CA',
    '8': 'CO',
    '9': 'CT',
    '10': 'DE',
    '12': 'FL',
    '13': 'GA',
    '15': 'HI',
    '16': 'ID',
    '17': 'IL',
    '18': 'IN',
    '19': 'IA',
    '20': 'KS',
    '21': 'KY',
    '22': 'LA',
    '23': 'ME',
    '24': 'MD',
    '25': 'MA',
    '26': 'MI',
    '27': 'MN',
    '28': 'MS',
    '29': 'MO',
    '30': 'MT',
    '31': 'NE',
    '32': 'NV',
    '33': 'NH',
    '34': 'NJ',
    '35': 'NM',
    '36': 'NY',
    '37': 'NC',
    '38': 'ND',
    '39': 'OH',
    '40': 'OK',
    '41': 'OR',
    '42': 'PA',
    '44': 'RI',
    '45': 'SC',
    '46': 'SD',
    '47': 'TN',
    '48': 'TX',
    '49': 'UT',
    '50': 'VT',
    '51': 'VA',
    '53': 'WA',
    '54': 'WV',
    '55': 'WI',
    '56': 'WY',
    '72': 'PR',
}

if __name__ == "__main__":
    data_source = ACSDataSource(survey_year="2016", horizon="1-Year",
                                survey='person')
    acs_data = data_source.get_data(states=state_names, download=True)

    # Choose a random sample of 0.4 * rows for each state
    state_dfs = {}
    for state in state_names:
        state_df = acs_data.loc[acs_data['ST'] == int(_STATE_CODES[state])]
        sample = state_df.sample(n=int(0.4 * len(state_df)), random_state=0)
        state_dfs[state] = sample

    # Add division and region features
    for division in DIVISIONS:
        for state in DIVISIONS[division]:
            div_code = DIVISION_CODES[division]
            state_dfs[state]['DIV'] = [div_code] * len(state_dfs[state])

    for region in REGIONS:
        for state in REGIONS[region]:
            region_code = REGION_CODES[region]
            state_dfs[state]['REG'] = [region_code] * len(state_dfs[state])

    # Concatenate into a big DF with all states and regions
    DF_PATH = 'data/'       # should exist if we loaded Folktables data
    all_df = pd.concat(state_dfs.values())
    all_df = all_df.sample(frac=1) # shuffle the rows
    print("Writing DF to {}".format(os.path.join(DF_PATH, 'states.csv')))
    all_df.to_csv(os.path.join(DF_PATH, 'states.csv'), index=False)