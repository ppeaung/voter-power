import os
import csv 
import maup
import geopandas as gpd
import matplotlib.pyplot as plt
from gerrychain import Graph
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import t

# ======================================================================
# Constants
# ======================================================================

YEAR = 2024
SIGMA = 5

state_fips = {
    'AL': 1, 'AK': 2, 'AZ': 4, 'AR': 5, 'CA': 6,
    'CO': 8, 'CT': 9, 'DE': 10, 'FL': 12, 'GA': 13,
    'HI': 15, 'ID': 16, 'IL': 17, 'IN': 18, 'IA': 19,
    'KS': 20, 'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24,
    'MA': 25, 'MI': 26, 'MN': 27, 'MS': 28, 'MO': 29,
    'MT': 30, 'NE': 31, 'NV': 32, 'NH': 33, 'NJ': 34,
    'NM': 35, 'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39,
    'OK': 40, 'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45,
    'SD': 46, 'TN': 47, 'TX': 48, 'UT': 49, 'VT': 50,
    'VA': 51, 'WA': 53, 'WV': 54, 'WI': 55, 'WY': 56,
    'DC': 11
}

# ======================================================================
# Helper Functions
# ======================================================================

def round_to_nearest_tens(x):
    return round(x / 10) * 10

# ======================================================================
# Meta-Margin Calculation 
# ======================================================================

# TODO: Implement

# ======================================================================
# Reading Shapefiles
# ======================================================================

# Initialize first one



for state_abbreviation in state_fips:
    
