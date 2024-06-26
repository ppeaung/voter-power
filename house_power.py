import os
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import t
from scipy.special import erf


# ======================================================================
# GLOBAL VARIABLES 
# ======================================================================

YEAR = 2024
DAYS_UNTIL_ELECTION = (datetime(2024,11,5) - datetime.today()).days
SIGMA = 8
HOUSE_DISTRICTS = 'AZ-1,AZ-4,AZ-6,NC-1,OR-5,MT-1,WI-1,WI-3,OH-1,OH-9,OH-13 '

## TODO: Figure out what the polling format is for that pandas data frame. 

## Toy data for some Congressional Districts

## Will highlight the ones that Sam has on ``PEC information 2024'' see this link: https://docs.google.com/spreadsheets/d/1zox_09R8m3ELD_3H99q56HtcGBLPvcPAe5Zou-ZGV6o/edit?gid=0#gid=0

# ======================================================================
# LOAD POLL DATA
# ======================================================================

## TODO: Figure out polling data structure for the House and load the appropriate margins for each congressional district

# Each state takes up 5 spaces
house_csv_file_path = f'./outputs/{YEAR}.house.polls.median.csv'
house_df = pd.read_csv(house_csv_file_path)

#num_states = len(SENATE_STATES) // 5
#if senate_df.shape[0] % num_states !=0: 
#    raise Exception(f"Warning: {YEAR}.Senate.polls.median.csv is not a multiple of num_states lines long") 

#senate_df = senate_df.iloc[:num_states]
#AZ-4 has no polling data yet
HOUSE_DISTRICTS = 'AZ-1,AZ-4,AZ-6,NC-1,OR-5,MT-1,WI-1,WI-3,OH-1,OH-9,OH-13 '

# as of June 26th 
house_margins = np.array([-3.0, 6, -4, 3, -4, -4, -1, 0.3, 2, 0.2, 4])

# ======================================================================
# METAMARGIN CALCULATION
# ======================================================================

## TODO: Implement Metamargin calculation

metamargin = 0

# ======================================================================
# LOAD VOTER TURNOUT DATA
# ======================================================================

# Assuming that each CD is population balanced, so the population should be approximately the same. 
# TODO: If we want to get more precise, we can get CD population data too. 

HOUSE_DISTRICTS = 'AZ-1,AZ-4,AZ-6,NC-1,OR-5,MT-1,WI-1,WI-3,OH-1,OH-9,OH-13 '

num_congressional_districts = np.array([9, 14, 6, 2, 8, 15])
state_population = np.array([7.359, 10.7, 4.24, 1.123, 5.893, 11.76]) # in millions
population_per_dc = state_population / num_congressional_districts
vote_turnout = np.array([population_per_dc[0], population_per_dc[0], population_per_dc[0], 
                               population_per_dc[1],
                               population_per_dc[2],
                               population_per_dc[3],
                               population_per_dc[4],population_per_dc[4],
                               population_per_dc[5],population_per_dc[5],population_per_dc[5]])

# TODO: Change csv file titles to align with variables (Unnamed -> ...)

# TODO: Read voter_turnout csv so that we only select the states of interest for the Senate instead of all 51 States

# vote_turnout_csv_file_path = './data/voter_turnout_data2022.csv'
# df = pd.read_csv(vote_turnout_csv_file_path)

# Extract state and turnout numbers
# vote_turnout_df = df[['State','Unnamed: 1']]
# vote_turnout_df =  vote_turnout_df.iloc[2:-1]

# Remove commas 
# vote_turnout_df['Unnamed: 1'] = vote_turnout_df['Unnamed: 1'].apply(lambda x: pd.to_numeric(x.replace(',', ''), errors='coerce'))
# vote_turnout_df['Unnamed: 1'] = pd.to_numeric(vote_turnout_df['Unnamed: 1'])

# Remove * 
# vote_turnout_df['State'] = vote_turnout_df['State'].apply(lambda x: x.replace('*', ''))

# For now, let's just look at the numbers 
# vote_turnout_dict = vote_turnout_df.to_dict(orient = "index")
# vote_turnout = vote_turnout_df['Unnamed: 1'].to_numpy()

# ======================================================================
# VOTER POWER CALCULATION
# ======================================================================

# t-pdf setup 
df = 3 
t_dist = t(df)

# For sanity checking
metamargin = 0 

Z = (house_margins - metamargin) / SIGMA
num = t_dist.pdf(Z)
den = vote_turnout

voter_power = np.divide(num, den)
voter_power = ((voter_power - np.min(voter_power)) / (np.max(voter_power) - np.min(voter_power)))*100
voter_power = np.round(voter_power)

print(f"The congressional districts are {HOUSE_DISTRICTS}")
print(f"The voter power for the congressional districts of interest are {voter_power}")

# TODO: Make the saving process more automated

FIPS_Code = np.array([4, 4, 4, 37, 41, 30, 55, 55, 39, 39, 39])
State_List = np.array(['AZ-1','AZ-4','AZ-6','NC-1','OR-5','MT-1','WI-1','WI-3','OH-1','OH-9','OH-13'])
HOUSE_DISTRICTS = 'AZ-1,AZ-4,AZ-6,NC-1,OR-5,MT-1,WI-1,WI-3,OH-1,OH-9,OH-13 '

congressional_districts = []

for i in range(len(State_List)):
    congressional_districts.append("Congressional District "+ State_List[i][3])
    State_List[i] = State_List[i][0:2]


# TODO: Add shapefile geometry as a dataset 

df = pd.DataFrame({
    'STATEFP' : FIPS_Code, 
    'States' : State_List,
    'NAMELSAD' : congressional_districts, 
    'Margins': house_margins, 
    'Voter Power': voter_power
})

dir_path = os.path.dirname(os.path.realpath(__file__))

path = os.path.join(dir_path, f'outputs/{YEAR}.house.VoterPower.csv')
df.to_csv(path, index=False, float_format='%.2f')