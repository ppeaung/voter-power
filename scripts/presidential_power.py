import os
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import t


# ======================================================================
# GLOBAL VARIABLES 
# ======================================================================

YEAR = 2024
DAYS_UNTIL_ELECTION = (datetime(2024,11,5) - datetime.today()).days
SIGMA = 5
EV_PER_STATE = [9,3,11,6,54,10,7,3,3,30,16,4,4,19,11,6,6,8,8,2,10,11,15,10,6,10,4,2,6,4,14,5,28,16,3,17,7,8,19,4,9,3,11,40,6,3,13,12,4,10,3,1,1,1,1,1]
EV_STATES = 'AL,AK,AZ,AR,CA,CO,CT,DC,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY,M1,M2,N1,N2,N3 '

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def Senate_median(polls, bias_pct, num_states, dem_safe):
    """
    Args:
        - polls: Pandas dataframe containing scraped poll data
        - biaspct: Bias term
        - num_states: Number of states 
        - Demsafe: Number of confidently safe Senate Democratic seats 
        
    Returns: 
        various calculated values that correspond to their MATLAB counterparts
        Note that the tcdf function needs to be replaced with the appropriate Python function

    scipy.stats.t.cdf
    computes the cumulative distribution function (CDF) of the t-distribution
    
    x = np.array([-1, 0, 3, 4])  # Values at which to evaluate the CDF
    df = 10  # Degrees of freedom
    p = t.cdf(x, df)  # Compute the CDF

    print(p)
    # Output: [0.19146012 0.5        0.99330715 0.99864852]

    """

    # Working with pandas.DataFrame
    
    num_polls = polls['num_polls'] 
    # julian_date = polls[:, 1]
    # date_most_recent_poll = polls[:, 2]
    # median_margin = polls[:, 3]
    # est_std_dev = polls[:, 4]
    # state_num = polls[:, 5]

    # Calculate z-score and convert to probability, assuming normal distribution.
    polls['z'] = (polls['median_margin'] + bias_pct) / polls['median_std_dev']

    polls['prob_Dem_win'] = t.cdf(polls['z'], df=2)
    polls['prob_GOP_win'] = 1 - polls['prob_Dem_win']

    # TODO: Ask Sam about Estimated Standard Deviation method (probably using the MLE for std)

    # using ESD, not SEM
    polls_znov = (polls['median_margin'] + bias_pct) / np.sqrt(polls['SEM'] * polls['SEM'] + 5 * 5)
    # SEM = ESD / sqrt (N) where N = num of polls
    # polls_znov = (polls['median_margin'] + bias_pct) / np.sqrt(polls['est_std_dev'] * polls['est_std_dev'] + 5 * 5)
    polls['prob_Dem_November'] = t.cdf(polls_znov, 2)

    state_probs = np.round(polls['prob_Dem_win'] * 100)
    state_nov_probs = np.round(polls['prob_Dem_November'] * 100)

    # print(state_nov_probs)

    # The meta-magic: store the Electoral Votes (EV) distribution,
    # the exact probability distribution of all possible outcomes
    # Initialize a list to store the EV distribution
    EV_distribution = [1 - polls['prob_Dem_win'].iloc[0], polls['prob_Dem_win'].iloc[0]]
    for i in range(1, num_states):
        nextEV = [1 - polls['prob_Dem_win'].iloc[i], polls['prob_Dem_win'].iloc[i]]
        EV_distribution = np.convolve(EV_distribution, nextEV)
    
    print("Printing")
    print(EV_distribution)
    print("Done")

    # Cumulative histogram of all possibilities
    histogram = EV_distribution[1:num_states + 1]
    cumulative_prob = np.cumsum(histogram)

    print(histogram)

    # Range of Senate seats Democratics are projected to win
    senate_seats = range(dem_safe + 1, dem_safe + num_states + 1)
    print(senate_seats)
    R_Senate_control_probability = cumulative_prob[max(50 - dem_safe, 0)]
    D_Senate_control_probability = 1 - R_Senate_control_probability

    # Calculate Senate seat that represents the median of distribution of outcomes
    median_index = next(i for i, prob in enumerate(cumulative_prob) if prob >= 0.5)
    median_seats = senate_seats[median_index]

    # Calculate the mean number of expected Senate seats 
    weighted_seats = [histogram[i] * senate_seats[i] for i in range(len(histogram))]
    mean_seats = round(sum(weighted_seats), 2)

    print(mean_seats)

    return state_probs, state_nov_probs, histogram, cumulative_prob, R_Senate_control_probability, D_Senate_control_probability, median_seats, mean_seats

# ======================================================================
# LOAD POLL DATA
# ======================================================================

# Each state takes up 3 spaces
senate_csv_file_path = f'../outputs/{YEAR}.EV.polls.median.csv'
senate_df = pd.read_csv(senate_csv_file_path)

num_states = len(EV_STATES) // 3
if senate_df.shape[0] % num_states !=0: 
    raise Exception(f"Warning: {YEAR}.EV.polls.median.csv is not a multiple of num_states lines long") 

senate_df = senate_df.iloc[:num_states]

# TODO: Ask Sam about the implementation of analysisdate/search for it. As I see right now, there's no need
# to consider analysisdate (so do the loop where we only consider the first 1:numstates from the scraped polls)
# See the MATLAB script below

#% find the desired data within the file
#if analysisdate>0 && numlines>num_states
#    foo=find(polldata(:,2)==analysisdate,1,'first'); % find the start of the entry matching analysisdate
# %   ind=min([size(polldata,1)-num_states+1 foo']);
#    foo2=find(polldata(:,2)==max(polldata(:,5)),1,'first'); % find the start of the freshest entry
#    ind=max([foo2 foo]); %assume reverse time order, take whichever of the two was done earlier, also protect against no data for analysisdate
#    polldata=polldata(ind:ind+num_states-1,:);
#    clear foo2 foo ind
#elseif numlines>num_states
#%    polldata = polldata(numlines-num_states+1:numlines,:); % end of file
#    polldata = polldata(1:num_states,:); % top of file
#end

# Convert polls to statistics

# NOTE: May not need to convert to numpy array if we deal with Pandas df 

senate_margins = senate_df['median_margin'].to_numpy()

min_uncertainty = np.full(num_states, 3) # At least 2% uncertainty
senate_df['SEM'] = np.maximum(senate_df['median_std_dev'], min_uncertainty)
senate_sem = senate_df['SEM'].to_numpy()

total_polls_used = (senate_df['num_polls'].to_numpy()).sum()

# ======================================================================
# METAMARGIN CALCULATION
# ======================================================================

bias_pct = 0 # Where would this change? 
dem_safe = 42 # Number of safe Democratic seats for 2024

# Calculate needed statistics associated with Senate mean

(state_probs, state_nov_probs, histogram, cumulative_prob,
 R_Senate_control_probability, D_Senate_control_probability,
 median_seats, mean_seats) = Senate_median(senate_df, bias_pct, num_states, dem_safe)

reality = 1 - D_Senate_control_probability

meta_calc = 1

if meta_calc == 0: 
    metamargin = -999
else: 
    foo = bias_pct
    bias_pct = -7
    (state_probs, state_nov_probs, histogram, cumulative_prob,
    R_Senate_control_probability, D_Senate_control_probability,
    median_seats, mean_seats) = Senate_median(senate_df, bias_pct, num_states, dem_safe)
    while median_seats < 50: 
        bias_pct = bias_pct + 0.02
        (state_probs, state_nov_probs, histogram, cumulative_prob,
        R_Senate_control_probability, D_Senate_control_probability,
        median_seats, mean_seats) = Senate_median(senate_df, bias_pct, num_states, dem_safe)
    metamargin = -bias_pct
    bias_pct = foo
    del foo 

# ======================================================================
# LOAD VOTER TURNOUT DATA
# ======================================================================

# Old one
# vote_turnout = np.array([2592, 7797, 2031, 4500, 468, 1024, 4201, 5410, 8152, 2673, 495]) 

voters = np.array([1424087, 267047, 2592313, 914227, 11146610, 2540666, 1297811, 325632, 205774, 
                   7796916, 3964926, 423443, 595350, 4144125, 1880755, 1230417, 1008998, 1502550, 
                   1410466, 680909, 2031635, 2511461, 4500400, 2526646, 709100, 2304250, 468326, 
                   682716, 1023617, 626931, 2645539, 714754, 5962278, 3790202, 242566, 4201368, 
                   1153284, 1997689, 5410022, 361449, 1718626, 354670, 1756397, 8151590, 1084634, 
                   291955, 3021956, 3067686, 494753, 2673154, 198198])

# Maine and Nebraska
voters = np.append(voters, [voters[19]/2, voters[19]/2, voters[27]/3, voters[27]/3, voters[27]/3])


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

Z = (senate_margins - metamargin) / SIGMA
num = t_dist.pdf(Z)*EV_PER_STATE
den = vote_turnout

voter_power = np.divide(num, den)
voter_power = ((voter_power - np.min(voter_power)) / (np.max(voter_power) - np.min(voter_power)))*100

# Hard-coded values for sanity check
# kvoters = np.array([2592, 7797, 2031, 4500, 468, 1024, 4201, 5410, 8152, 2673, 495]) 
# margins = np.array([3.0, -8.0, 5.0, 3.0, 0.0, 11.0, 8.0, 7.0, -11.0, 8.0, -33.0])

print(f"The metamargin is {metamargin}")
print(f"The voter power is {voter_power}")

# ======================================================================
# SAVE DATA 
# ======================================================================

# TODO: Get codes from geocodes csv file rather than hardcoding 
FIPS_Code = np.array([4,12,24,26,30,32,39,42,48,55,54])
Senate_Seats_List = np.array(['AZ','FL','MD','MI','MT','NV','OH','PA','TX','WI','WV'])

df = pd.DataFrame({
    'FIPS_CODE' : FIPS_Code, 
    'States' : Senate_Seats_List, 
    'Margins': senate_margins, 
    'Voter Power': voter_power
})

dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, f'../outputs/{YEAR}.EV.VoterPower.csv')
df.to_csv(path, index=False, float_format='%.2f')