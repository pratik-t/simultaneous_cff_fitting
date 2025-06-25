"""
This script will run the replica method with N replicas to produce
a local fit of a given observable.
"""

# Native Library | argparse
import argparse

# Native Library | datetime
import datetime

# Native Library | os
import os

# 3rd Party Library | Pandas:
import pandas as pd

# static_strings > argparse > description:
from statics.static_strings import _ARGPARSE_DESCRIPTION
from statics.static_strings import _ARGPARSE_ARGUMENT_INPUT_DATAFILE
from statics.static_strings import _ARGPARSE_ARGUMENT_NUMBER_REPLICAS
from statics.static_strings import _ARGPARSE_ARGUMENT_VERBOSE
from statics.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_INPUT_DATAFILE
from statics.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_NUMBER_REPLICAS
from statics.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_VERBOSE

# static_strings > "k"
from statics.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM

# static_strings > "x_b"
from statics.static_strings import _COLUMN_NAME_X_BJORKEN

# static_strings > "q_squared"
from statics.static_strings import _COLUMN_NAME_Q_SQUARED

# static_strings > "t"
from statics.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE

# static_strings > "phi"
from statics.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI

from scripts.replica_data import generate_replica_data

SETTING_VERBOSE = True
SETTING_DEBUG = True

def main(
        kinematics_dataframe_name: str,
        number_of_replicas: int,
        verbose: bool = False):
    """
    Main entry point to the local fitting procedure.
    """
    
    # (1): Begin iteratng over the replicas:
    for replica_index in range(number_of_replicas):

        # (1.1): Obtain the replica number by adding 1 to the index:
        replica_number = replica_index + 1

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed replica number to be: {replica_number}")

        # (1.2): Propose a replica name:
        current_replica_name = f"replica_{replica_number}"

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed replica name to be: {current_replica_name}")

        # (1.3): Immediately construct the filetype for the replica:
        model_file_name = f"{current_replica_name}.h5"

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed corresponding replica file name to be: {model_file_name}")

        # (1.4): Create the directory for the replica:
        did_we_create_replica_directory = None

        # (X): Rely on Pandas to correctly read the just-generated .csv file:
        kinematics_dataframe_path = os.path.join("data", kinematics_dataframe_name)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed path to .csv file: {kinematics_dataframe_path}")

        # (X): Use Pandas' `.read_csv()` method to generate a corresponding DF:
        this_replica_data_set = pd.read_csv(kinematics_dataframe_path)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Now printing the Pandas DF head using df.head():\n {this_replica_data_set.head()}")

        # (X): We now compute a *given* replica's DF --- it will *not* be the same as the original DF!
        generated_replica_data = generate_replica_data(pandas_dataframe = this_replica_data_set)

        # (X): Identify the "x values" for our model:
        raw_x_data = generated_replica_data[[_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_LEPTON_MOMENTUM]]

        # (X): Identify the "y values" for our model:
        # raw_y_data = generated_replica_data[_COLUMN_NAME_CROSS_SECTION]
        # yraw__data_error = generated_replica_data[_COLUMN_NAME_CROSS_SECTION_ERROR]

        # (X): Begin timing the replica time:
        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        if verbose:
            print(f"> Replica #{replica_index + 1} now running...")


if __name__ == "__main__":

    # (1): Create an instance of the ArgumentParser
    parser = argparse.ArgumentParser(description = _ARGPARSE_DESCRIPTION)

    # (2): Enforce the path to the datafile:
    parser.add_argument(
        '-d',
        _ARGPARSE_ARGUMENT_INPUT_DATAFILE,
        type = str,
        required = True,
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_INPUT_DATAFILE)
    
    # (3): Enforce the path to the datafile:
    # parser.add_argument(
    #     '-kin',
    #     _ARGPARSE_ARGUMENT_KINEMATIC_SET_NUMBER,
    #     type = int,
    #     required = True,
    #     help = _ARGPARSE_ARGUMENT_DESCRIPTION_KINEMATIC_SET_NUMBER)

    # (4): Enforce the number of replicas:
    parser.add_argument(
        '-nr',
        _ARGPARSE_ARGUMENT_NUMBER_REPLICAS,
        type = int,
        required = True,
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_NUMBER_REPLICAS)

    # (5): Ask, but don't enforce debugging verbosity:
    parser.add_argument(
        '-v',
        _ARGPARSE_ARGUMENT_VERBOSE,
        required = False,
        action = 'store_false',
        help = _ARGPARSE_ARGUMENT_DESCRIPTION_VERBOSE)
    
    arguments = parser.parse_args()

    main(
        kinematics_dataframe_name = arguments.input_datafile,
        number_of_replicas = arguments.number_of_replicas,
        verbose = arguments.verbose)