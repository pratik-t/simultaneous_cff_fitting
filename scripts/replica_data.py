"""

"""

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
    names_of_columns = pseudodata_dataframe.columns

    print(names_of_columns)

    # (): We now need to figure out what observables are *in* the actual DF:
    if "sigma [nb]" in names_of_columns:
        print("> Detected total cross-section observable.")

    if "ALU" in names_of_columns:
        print("> Detected beam spin asymmetry observable.")

    # Please provide the rest of them:


    return pseudodata_dataframe
