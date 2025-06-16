"""
Here are a few exapmples of how one might use this library
for analysis. Physicists are mostly interested in: cross sections,
vaarious cross-section asymmetries, CFFs, and GPDs.
"""

# External Library | NumPy
import numpy as np

# External Library | Pandas
import pandas as pd

# (1): Specify the name (and location) of the data file:
DATA_FILE = './data/dvcs_CLAS_2009_tab1.csv'

# (2): Use Pandas to read the .csv:
pandas_dataframe = pd.read_csv(
    filepath_or_buffer = DATA_FILE,
    delimiter = ",")

# (3): Verify the data columns:
print(f"> Found columns:\n> {pandas_dataframe.columns}")