"""
This script generates replica data using the pseudodata sampling method.
"""

# Native Library | os
import os

# Native Library | glob
import glob

# Native Library | re
import re

# 3rd Party Libraries | Pandas:
import pandas as pd

# 3rd Party Libraries | Numpy:
import numpy as np

# 3rd Party Libraries | Matplotlib:
import matplotlib.pyplot as plt

SETTING_VERBOSE = True
SETTING_DEBUG = True

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

    # (3): Initialize an empty array to hold names of columns corresponding to observables:
    names_of_columns = []

    # (4): Use RegEx to initialize a list of column names to match:
    exclude_pattern = re.compile(r"(sig_BSA|del_ALU|_min|_max|link)$")

    # (5); We now iterate over all the columns in a given dataframe:
    for column in all_columns:

        # (5.1): Here, we compare the name of the given column to the RegEx pattern:
        if exclude_pattern.search(column):

            # (5.1.1): If it matches, then we *exclude* the column without "pure" observables:
            continue
        
        # (5.2): Otherwise, we first turn the corresponding column into a Series type with no NaNs:
        series = pseudodata_dataframe[column].dropna().astype(str)

        # (5.3): We check if the created Series has any (non!)-empty strings:
        # | This is done to *include* columns containing *some* experimental data
        # | rather than keeping columns with *no* experimental data.
        series_of_nonempty_entries = series.ne('')

        # (5.4): And, if we detect *any* element in the Series that is the empty string...
        # | (Notice this will *not* pass in the case that *all* Series elements are False!)
        if series_of_nonempty_entries.any():

            # (5.4.1): ... then we add the column to the list of column names --- should *not* be observables!
            names_of_columns.append(column)

    # (): We now need to figure out what kinematics and observables are *in* the actual DF:

    # Create a dictionary of observables
    # UNSURE OF THE MEANING OF VARIABLES
    # kinematics = {
    #     "q_squared": "> Detected Q^2 kinematic.",
    #     "x_b": "> Detected Bjorken variable kinematic.",
    #     "w": "> Detected COM energy (W) kinematic.",
    #     "k": "> Detected incoming lepton energy (k) kinematic.",
    #     "t": "> Detected momentum transfer (t) kinematic.",
    #     "phi": "> Detected angle (phi) kinematic."
    # }

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

    # (X): Iterate over the collected observables from the last *for* loop:
    for column_name in names_of_columns:

        if SETTING_DEBUG:
            print(f"> [DEBUG] Now analyzing column: {column_name}")

        # (X): If the column name corresponds to an observable...
        if column_name in observables:

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Found column corresponding to observable! {column_name}")

            if any(suffix in column_name for suffix in ['_stat_plus', '_stat_minus', '_sys_plus', '_sys_minus']):
                continue

            # (): Rely on RegEx to eliminate "unit brackets" around names of observables,
            # | then strip any whitespace:
            column_base_name = re.sub(r'\s*\[.*?\]', '', column_name).strip()

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Stripped column name to: {column_base_name}")

            # (): Construct the corresponding string (which will become a key) for statistical 
            # | uncertainty for this observable. Notce that we are taking the PLUS value to 
            # | represent the "width":
            observable_statistical_uncertainty = f"{column_base_name}_stat_plus"

            # (): Same as above except the systematic uncertainty:
            observable_statistical_uncertainty = f"{column_base_name}_sys_plus"

            # (X): Obtain a Series consisting of *the untouched*, raw, experimental observable values:
            mean_values = pandas_dataframe[column_name]

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Successfully queried corresponding Series values:\n{mean_values}")

            # (X): Obtain a Series of the above's corresponding uncertainty:
            standard_deviations = pandas_dataframe[observable_statistical_uncertainty]

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Successfully queried observable's corresponding errors:\n{standard_deviations}")

            # (X): Perform element-wise Normal Distribution sampling to construct a *new* Series
            # | column --- this is the "pseudodata representation" of the original observable:
            pseudodata_dataframe[column_name] = np.random.normal(
                loc = mean_values,
                scale = standard_deviations)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Randomly sampled U(mean_values, standard_deviations) to obtain new Series:\n{pseudodata_dataframe[column_name] }")

            # figure_instance_pseudodata = plt.figure(figsize = (10, 5.5))

            # axis_instance_pseudodata = figure_instance_pseudodata.add_subplot(1, 1, 1)

            # axis_instance_pseudodata.set_title("Data from .csv")
            # axis_instance_pseudodata.set_xlabel(r"$\phi$")
            # axis_instance_pseudodata.set_ylabel("Observable")

            # axis_instance_pseudodata.errorbar(
            #     x = pandas_dataframe['phi'],
            #     y = pandas_dataframe[column_name],
            #     yerr = pandas_dataframe[observable_statistical_uncertainty],
            #     marker = 'o',
            #     linestyle = '',
            #     markersize = 3.0,
            #     ecolor = 'black',
            #     elinewidth = 0.5,
            #     capsize = 1,
            #     color = 'black',
            #     label = "Raw Data")
            
            # axis_instance_pseudodata.errorbar(
            #     x = pseudodata_dataframe['phi'],
            #     y = pseudodata_dataframe[column_name],
            #     yerr = pandas_dataframe[observable_statistical_uncertainty],
            #     marker = 'o',
            #     linestyle = '',
            #     markersize = 3.0,
            #     ecolor = 'black',
            #     elinewidth = 0.5,
            #     capsize = 1,
            #     color = 'orange',
            #     label = "Generated Pseudodata")
            
            # plt.legend()

            # plt.show()
            # plt.close()

    pseudodata_dataframe.to_csv("test1.csv")

    return pseudodata_dataframe


script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.abspath(os.path.join(script_dir, '..', 'data'))

for filepath in glob.glob(os.path.join(folder_path, '*.csv')):
    filename = os.path.basename(filepath)

    print(f'\n========================{filename}========================\n')

    df = pd.read_csv(filepath)
    generate_replica_data(df)
