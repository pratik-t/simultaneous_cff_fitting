"""

"""

import os
import glob
import re

# 3rd Party Libraries | Pandas:
import pandas as pd

# 3rd Party Libraries | Numpy:
import numpy as np

def generate_replica_data(pandas_dataframe: pd.DataFrame):
    """
    ## Description:
    Generates a replica dataset by sampling a given observable 
    within a Normal Distribution within its standard deviation.
    """

    # (1): We first copy the original DF:
    pseudodata_dataframe = pandas_dataframe.copy()

    # (2): We extract the *names* of the columns to determine what observables are in them:
    all_columns = pseudodata_dataframe.columns

    # exclude empty, error and min/max columns

    names_of_columns = []
    exclude_pattern = re.compile(r"(sig_BSA|del_ALU|_sys_plus|_sys_minus|_stat_plus|_stat_minus|_min|_max|link)$")
    for col in all_columns:
        if exclude_pattern.search(col):
            continue
        series = pseudodata_dataframe[col].dropna().astype(str)
        if series.ne('').any():
            names_of_columns.append(col)

    # (): We now need to figure out what kinematics and observables are *in* the actual DF:

    # Create a dictionary of observables
    # UNSURE OF THE MEANING OF VARIABLES
    kinematics = {
        "q_squared": "> Detected q^2 kinematic.",
        "x_b": "> Detected Bjorken variable kinematic.", 
        "w": "> Detected COM energy (W) kinematic.",
        "k": "> Detected incoming lepton energy (k) kinematic.",
        "t": "> Detected momentum transfer (t) kinematic.",
        "phi": "> Detected angle (phi) kinematic."
    }

    # Create a dictionary of observables
    # UNSURE OF THE MEANING OF VARIABLES
    observables = {
        "sigma [nb]":
            "> Detected total cross-section observable.",
        "dsigma/dt [nb/GeV^2]":
            "> Detected d_sigma/d_t observable.",
        "D2_sigma_d_omega (nb/sr)":
            "> Detected d^2_sigma/d_omega observable.",
        "D4_sigma (nb/Gev^4)":
            "> Detected d^4_sigma observable.",
        "D5_sigma (fb/(MeV sr2))":
            "> Detected d^5_sigma observable.",
        "Helc_diff_D^4_sigma (nb/Gev^4)":
            "> Detected Helicity difference observable.",
        "1/2 Helc_diff_d4_sigma (nb/GeV^4)":
            "> Detected Helicity difference observable.",
        "1/2 Helc_sum_d4_sigma (nb/GeV^4)":
            "> Detected Helicity sum observable.",
        "ALU":
            "> Detected beam spin asymmetry observable.",
        "BSA": 
            "> Detected beam spin asymmetry observable.",
        "ALU_sin_PHI":
            "> Detected beam spin asymmetry observable.",
        "ALU_sin_2PHI":
            "> Detected beam spin asymmetry observable.",
        "TSA":
            "> Detected target spin asymmetry observable.",
        "DSA":
            "> Detected double spin asymmetry observable.",
        "BCA":
            "> Detected beam charge asymmetry observable.",
    }

    remaining_columns = []

    for col_name in names_of_columns:
        if col_name in kinematics.keys():
            print(kinematics.get(col_name))
        elif col_name in observables.keys():
            print(observables.get(col_name))
        else:
            remaining_columns.append(col_name)

    if np.size(remaining_columns)>1:
        print(f'\nRemaining Columns: {remaining_columns}')

    return pseudodata_dataframe


# Check if code is working as intended:

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.abspath(os.path.join(script_dir, '..', 'data'))

for filepath in glob.glob(os.path.join(folder_path, '*.csv')):
    filename = os.path.basename(filepath)

    print(f'\n========================{filename}========================\n')

    df = pd.read_csv(filepath)
    generate_replica_data(df)
