"""
Later!
"""
# Native Library | gc
import gc

# Native Library | argparse
import argparse

# Native Library | datetime
import datetime

# Native Library | os
import os

# 3rd Party Library | NumPy:
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# 3rd Party Library | Matplotlib:
import matplotlib.colors as mcolors

# 3rd Party Library | Pandas:
import pandas as pd

# 3rd Party Library | TensorFlow:
import tensorflow as tf

# 3rd Party Library | Keras:
import keras

# 3rd Party Library | Keras
import tensorflow.keras.backend as K

# 3rd Party Library | sklearn:
from sklearn.model_selection import train_test_split

# 3rd Party Library | SciPy:
from scipy.stats import norm

# 3rd Party Library | tqdm:
from tqdm import tqdm

# (X): Function | model > architecture > build_simultaneous_model
from models.architecture import build_simultaneous_model

# (X): In order to correctly deserialize a TF model, you need to define
# | custom objects when loading it. And so that requires that we
# | actually import the damn custom layers we made:
from models.architecture import CrossSectionLayer, BSALayer

# (X): Function | scripts > replica_data > generate_replica_data
from scripts.replica_data import generate_replica_data

from scripts.train_local_fit import run_replica

from scripts.train_local_fit import create_subdirectories, create_run_directory

# static_strings > argparse > description:
from statics.static_strings import _ARGPARSE_DESCRIPTION

# static_strings > argparse > -d:
from statics.static_strings import _ARGPARSE_ARGUMENT_INPUT_DATAFILE

# static_strings > argparse > -nr:
from statics.static_strings import _ARGPARSE_ARGUMENT_NUMBER_REPLICAS

# static_strings > argparse > verbose:
from statics.static_strings import _ARGPARSE_ARGUMENT_VERBOSE

# static_strings > argparse > description for -d:
from statics.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_INPUT_DATAFILE

# static_strings > argparse > description for -nr
from statics.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_NUMBER_REPLICAS

# static_strings > argparse > description for verbose:
from statics.static_strings import _ARGPARSE_ARGUMENT_DESCRIPTION_VERBOSE

# static_strings > "set"
from statics.static_strings import _COLUMN_NAME_KINEMATIC_BIN

# static_strings > "k"
from statics.static_strings import _COLUMN_NAME_LEPTON_MOMENTUM

# static_strings > r"$x_B$"
from statics.static_strings import _COLUMN_NAME_X_BJORKEN

# static_strings > "q_squared"
from statics.static_strings import _COLUMN_NAME_Q_SQUARED

# static_strings > "t"
from statics.static_strings import _COLUMN_NAME_T_MOMENTUM_CHANGE

# static_strings > "phi"
from statics.static_strings import _COLUMN_NAME_AZIMUTHAL_PHI

# static_strings > "F"
from statics.static_strings import _COLUMN_NAME_CROSS_SECTION

# static_strings > "F_err"
from statics.static_strings import _COLUMN_NAME_CROSS_SECTION_ERROR

# static_strings > /replicas
from statics.static_strings import _DIRECTORY_REPLICAS

# static_strings > /data
from statics.static_strings import _DIRECTORY_DATA

# static_strings > /data/raw
from statics.static_strings import _DIRECTORY_DATA_RAW

# static_strings > /data/replicas
from statics.static_strings import _DIRECTORY_DATA_REPLICAS

# static_strings > /replicas/losses
from statics.static_strings import _DIRECTORY_REPLICAS_LOSSES

# static_strings > /replicas/fits
from statics.static_strings import _DIRECTORY_REPLICAS_FITS

# static_strings > /replicas/performance
from statics.static_strings import _DIRECTORY_REPLICAS_PERFORMANCE

# static_strings > .keras
from statics.static_strings import _TF_FORMAT_KERAS

# static_strings > .eps
from statics.static_strings import _FIGURE_FORMAT_EPS

# static_strings > .svg
from statics.static_strings import _FIGURE_FORMAT_SVG

# static_strings > .png
from statics.static_strings import _FIGURE_FORMAT_PNG

# static_strings > array of subdirectories required
from statics.static_strings import REQUIRED_SUBDIRECTORIES_LIST

# static_strings > number of epochs
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_EPOCHS

# static_strings > batch size for training
from statics.static_strings import _HYPERPARAMETER_BATCH_SIZE

# static_strings > learning rate patience parameter
from statics.static_strings import _HYPERPARAMETER_LR_PATIENCE

# static_strings > learning rate factor
from statics.static_strings import _HYPERPARAMETER_LR_FACTOR

# static_strings > earlystop callback parameter
from statics.static_strings import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER

# static_strings > verbose output in TF
from statics.static_strings import _DNN_VERBOSE_SETTING

# static_strings > train/test split percentage
from statics.static_strings import _DNN_TRAIN_TEST_SPLIT_PERCENTAGE

# (X): We tell rcParams to use LaTeX. Note: this will *crash* your
# | version of the code if you do not have TeX distribution installed!
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# (X): rcParams for the x-axis tick direction:
plt.rcParams['xtick.direction'] = 'in'

# (X): rcParams for the "major" (larger) x-axis vertical size:
plt.rcParams['xtick.major.size'] = 8.5

# (X): rcParams for the "major" (larger) x-axis horizonal width:
plt.rcParams['xtick.major.width'] = 0.5

# (X): rcParams for the "minor" (smaller) x-axis vertical size:
plt.rcParams['xtick.minor.size'] = 2.5

# (X): rcParams for the "minor" (smaller) x-axis horizonal width:
plt.rcParams['xtick.minor.width'] = 0.5

# (X): rcParams for the minor ticks to be *shown* versus invisible:
plt.rcParams['xtick.minor.visible'] = True

# (X): rcParams dictating that we want ticks along the x-axis on *top* (opposite side) of the bounding box:
plt.rcParams['xtick.top'] = True

# (X): rcParams for the y-axis tick direction:
plt.rcParams['ytick.direction'] = 'in'

# (X): rcParams for the "major" (larger) y-axis vertical size:
plt.rcParams['ytick.major.size'] = 8.5

# (X): rcParams for the "major" (larger) y-axis horizonal width:
plt.rcParams['ytick.major.width'] = 0.5

# (X): rcParams for the "minor" (smaller) y-axis vertical size:
plt.rcParams['ytick.minor.size'] = 2.5

# (X): rcParams for the "minor" (smaller) y-axis horizonal width:
plt.rcParams['ytick.minor.width'] = 0.5

# (X): rcParams for the minor ticks to be *shown* versus invisible:
plt.rcParams['ytick.minor.visible'] = True

# (X): rcParams dictating that we want ticks along the y-axis on the *left* of the bounding box:
plt.rcParams['ytick.right'] = True

SETTING_VERBOSE = True
SETTING_DEBUG = True

def plot_kinematic_distributions(kinematic_dataframe: pd.DataFrame):
    """
    ## Description:
    Later!
    """
    # if not all(col in kinematic_dataframe.columns for col in [[_COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_T_MOMENTUM_CHANGE]]):
    #     raise ValueError("DataFrame must include columns: 'x_B', 'Q²', and 't'.")

    x_bjorken = kinematic_dataframe[_COLUMN_NAME_X_BJORKEN]
    q_squared = kinematic_dataframe[_COLUMN_NAME_Q_SQUARED]
    t = kinematic_dataframe[_COLUMN_NAME_T_MOMENTUM_CHANGE]
    minus_t = -t

    # (X): Set up the  -t vs. x_{B} figure:
    t_versus_x_bjorken_figure = plt.figure(figsize = (10, 6))

    # (X): Set up the Q^{2} vs. x_{B} figure:
    q_squared_versus_x_bjorken_figure = plt.figure(figsize = (10, 6))

    # (X): Set up the -t vs. Q^{2} figure:
    t_versus_q_squared_figure = plt.figure(figsize = (10, 6))

    # (X): Set up the kinematic domain figure:
    input_space_figure = plt.figure(figsize = (10, 6))

    # (X): Add an Axis object to the -t vs. x_{B} figure:
    t_versus_x_bjorken_axis = t_versus_x_bjorken_figure.add_subplot(1, 1, 1)

    # (X): Add an Axis object to the Q^{2} vs. x_{B} figure:
    q_squared_versus_x_bjorken_axis = q_squared_versus_x_bjorken_figure.add_subplot(1, 1, 1)

    # (X): Add an Axis object to the -t vs. Q^{2} figure:
    t_versus_q_squared_axis = t_versus_q_squared_figure.add_subplot(1, 1, 1)

    # (X): Add an Axis object to the kinematic domain figure:
    input_space_axis = input_space_figure.add_subplot(1, 1, 1, projection = "3d")

    t_versus_x_bjorken_axis.scatter(x_bjorken, minus_t, alpha = 0.8, s = 10.4)
    t_versus_x_bjorken_axis.set_xlabel(r"$x_B$")
    t_versus_x_bjorken_axis.set_ylabel(r"$-t$")
    t_versus_x_bjorken_axis.set_title(r"$-t$ vs. $x_B$")

    q_squared_versus_x_bjorken_axis.scatter(x_bjorken, q_squared, alpha = 0.8, s = 10.4)
    q_squared_versus_x_bjorken_axis.set_xlabel(r"$x_B$")
    q_squared_versus_x_bjorken_axis.set_ylabel(r"$Q^2$")
    q_squared_versus_x_bjorken_axis.set_title(r"$Q^2$ vs. $x_B$")

    t_versus_q_squared_axis.scatter(q_squared, minus_t, alpha = 0.8, s = 10.4)
    t_versus_q_squared_axis.set_xlabel(r"$Q^2$")
    t_versus_q_squared_axis.set_ylabel(r"$-t$")
    t_versus_q_squared_axis.set_title(r"$-t$ vs. $Q^2$")

    input_space_axis.scatter(x_bjorken, q_squared, minus_t, alpha = 0.8, s = 6.4)
    input_space_axis.set_xlabel(r"$x_B$")
    input_space_axis.set_ylabel(r"$Q^2$")
    input_space_axis.set_zlabel(r"$-t$")
    input_space_axis.set_title("Kinematic Settings Space")
    
    t_versus_x_bjorken_figure.savefig(fname = "test_v1.png", dpi = 300)
    q_squared_versus_x_bjorken_figure.savefig(fname = "test2_v2.png", dpi = 300)
    t_versus_q_squared_figure.savefig(fname = "test3_v2.png", dpi = 300)
    input_space_figure.savefig(fname = "test4_v2.png", dpi = 300)

    plt.close(t_versus_x_bjorken_figure)
    plt.close(q_squared_versus_x_bjorken_figure)
    plt.close(t_versus_q_squared_figure)
    plt.close(input_space_figure)


def get_unique_kinematic_sets(kinematics_dataframe):
    """
    ## Description:
    In local fitting across several kinematic regions, we need to be able to 
    determine the total number of kinematic sets in the file.
    """
    
    if _COLUMN_NAME_KINEMATIC_BIN not in kinematics_dataframe.columns:
        raise ValueError(f'> [ERROR]: The column "bin" is missing from {kinematics_dataframe}. Cannot proceed.')

    unique_sets = sorted(kinematics_dataframe[_COLUMN_NAME_KINEMATIC_BIN].unique())
    return unique_sets, len(unique_sets)

def main(
        kinematics_dataframe_name: str,
        number_of_replicas: int,
        verbose: bool = False):
    """
    ## Description:
    Main entry point to the local fitting procedure.

    ## Detailed Description:
    A "local fit" is a DNN fit to observables in an attempt to back-extract the CFFs at a
    fixed kinematic setting. In this function, we run several local fits at different kinematic
    settings. The objective is to extract the CFFs at these various settings so that we can see
    how the CFFs look at these several different points.
    """
    # (X): Rely on Pandas to correctly read the just-generated .csv file:
    kinematics_dataframe_path = os.path.join(_DIRECTORY_DATA, kinematics_dataframe_name)

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed path to data .csv file: {kinematics_dataframe_path}")

    # (X): Use Pandas' `.read_csv()` method to generate a corresponding DF:
    this_replica_data_set = pd.read_csv(kinematics_dataframe_path)

    unique_sets, number_of_unique_sets = get_unique_kinematic_sets(this_replica_data_set)

    if SETTING_VERBOSE:
        print(f"> [VERBOSE]: {number_of_unique_sets} unique kinematic settings.")

    plot_kinematic_distributions(this_replica_data_set)
    
    # (X): Determine devices' GPUs:
    gpus = tf.config.list_physical_devices('GPU')

    # (X): If there exist available GPUs...
    if gpus:

        # (X): ... begin iteration over each one and...
        for gpu in gpus:

            # (X): ... enforce growth of memory rather than using all of it:
            tf.config.experimental.set_memory_growth(gpu, True)

    entire_run_directory, creation_timestamp = create_run_directory(verbose = verbose)

    for kinematic_set_number in unique_sets:

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Now running local fitting routine on kinematic set #{kinematic_set_number}")

        # (1): Enforce creation of required directory structure:
        current_replica_run_directory = create_subdirectories(
            current_run_folder = entire_run_directory,
            data_file_name = kinematics_dataframe_name,
            kinematic_set = kinematic_set_number,
            number_of_replicas = number_of_replicas,
            timestamp = creation_timestamp,
            verbose = verbose)
        
         # (1): Begin iteratng over the replicas:
        for replica_index in range(number_of_replicas):

            # (1.1): Obtain the replica number by adding 1 to the index:
            replica_number = replica_index + 1

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed replica number to be: {replica_number}")

                run_replica(
                    kinematics_dataframe_name = kinematics_dataframe_name,
                    kinematic_set_number = kinematic_set_number,
                    replica_number = replica_number,
                    job_run_directory = current_replica_run_directory)
        
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