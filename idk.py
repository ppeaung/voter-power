import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt

csv_file_path = './data/voter_turnout_data2022.csv'
df = pd.read_csv(csv_file_path)

# Extract state and turnout numbers
vote_turnout_df = df[['State','Unnamed: 1']]
vote_turnout_df =  vote_turnout_df.iloc[1:-1]

# Remove commas 
vote_turnout_df['Unnamed: 1'] = vote_turnout_df['Unnamed: 1'].apply(lambda x: pd.to_numeric(x.replace(',', ''), errors='coerce'))
vote_turnout_df['Unnamed: 1'] = pd.to_numeric(vote_turnout_df['Unnamed: 1'])

# Remove * 
vote_turnout_df['State'] = vote_turnout_df['State'].apply(lambda x: x.replace('*', ''))

# For now, let's just look at the numbers 
# vote_turnout_dict = vote_turnout_df.to_dict(orient = "index")
vote_turnout = vote_turnout_df['Unnamed: 1'].to_numpy()

# meta_margin = 
sigma = 13 

Alabama_pop = vote_turnout[1]
# Alabama_margin = 



## Scrape the polling data and get the margins 

## Calculate the meta margin 

## Look at the pdf and calculate 

## Voter Turnout 

df = 3
t_dist = t(df)
x = np.linspace(-5, 5, 100)
y = t_dist.pdf(x)

plt.plot(x, y)
plt.show()
## Vote Contribution (EV/Senate)