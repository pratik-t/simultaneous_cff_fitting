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

# 3rd Party Library | NumPy:
import numpy as np

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# 3rd Party Library | Pandas:
import pandas as pd

# 3rd Party Library | Pandas:
import tensorflow as tf

# 3rd Party Library | sklearn:
from sklearn.model_selection import train_test_split

from models.architecture import build_simultaneous_model

from scripts.replica_data import generate_replica_data

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

# (X):
from statics.static_strings import _COLUMN_NAME_CROSS_SECTION

# (X):
from statics.static_strings import _COLUMN_NAME_CROSS_SECTION_ERROR

from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_EPOCHS

from statics.static_strings import _HYPERPARAMETER_BATCH_SIZE

from statics.static_strings import _HYPERPARAMETER_LR_PATIENCE

from statics.static_strings import _HYPERPARAMETER_LR_FACTOR

from statics.static_strings import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER

from statics.static_strings import _DNN_VERBOSE_SETTING

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True

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
        raw_kinematics = generated_replica_data[[
            _COLUMN_NAME_Q_SQUARED, 
            _COLUMN_NAME_X_BJORKEN, 
            _COLUMN_NAME_T_MOMENTUM_CHANGE,
            _COLUMN_NAME_LEPTON_MOMENTUM,
            _COLUMN_NAME_AZIMUTHAL_PHI]]

        # (X): Obtain the cross section data from the replica dataframe:
        raw_cross_section = generated_replica_data[_COLUMN_NAME_CROSS_SECTION]

        # (X): Obtain the associated cross section error from the replica dataframe:
        raw_cross_section_error = generated_replica_data[_COLUMN_NAME_CROSS_SECTION_ERROR]

        # (X): Use sklearn's traing/validation split function to split into training and testing data:
        x_training, x_validation, y_training, y_validation = train_test_split(
            raw_kinematics,
            raw_cross_section,
            test_size = 0.2,
            random_state = 42)

        # (X): Begin timing the replica time:
        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        if SETTING_DEBUG or SETTING_VERBOSE:
            print(f"> Replica #{replica_index + 1} now running...")

        # (X): Initialize the model:
        dnn_model = build_simultaneous_model()
        
        # (X): Here, we run the fitting procedure:
        neural_network_training_history = dnn_model.fit(

            # (X): Insert the training input-data here (independent variables):
            x_training,

            # (X): Insert the training output-data here (dependent variables):
            y_training,

            # (X): Insert a tuple of validation data according to (input, output):
            validation_data = (x_validation, y_validation),

            # (X): Hyperparameter: Epoch number:
            epochs = _HYPERPARAMETER_NUMBER_OF_EPOCHS,

            # (X): Hyperparameters: Batch size:
            batch_size = _HYPERPARAMETER_BATCH_SIZE,

            # (X): A list of TF callbacks:
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor = 'loss',
                    factor = _HYPERPARAMETER_LR_FACTOR,
                    patience = _HYPERPARAMETER_LR_PATIENCE,
                    mode = 'auto'),
                tf.keras.callbacks.EarlyStopping(
                    monitor = 'loss',
                    patience = _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER)
            ],

            # (X): TF verbose setting:
            verbose = _DNN_VERBOSE_SETTING)
        
        if SETTING_DEBUG or SETTING_VERBOSE:
            print(f"> Replica #{replica_index + 1} finished running!")

        # (X): Extract the 'loss' key from TF's history object. It has 
        # | loss vs. epoch data on it:
        training_loss_data = neural_network_training_history.history['loss']

        # (X): Extract the 'val_loss' (validation loss) from the TF history object:
        validation_loss_history_array = neural_network_training_history.history['val_loss']
            
        evaluation_figure = plt.figure(
            figsize = (11, 8))
        
        evaluation_axis = evaluation_figure.add_subplot(1, 1, 1)
        
        evaluation_axis.plot(
            np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
            np.array([np.max(training_loss_data) for number in training_loss_data]),
            color = "red",
            label = "Initial MSE Loss")
        
        evaluation_axis.plot(
            np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
            np.zeros(shape = _HYPERPARAMETER_NUMBER_OF_EPOCHS),
            color = "green",
            label = r"MSE $=0$")
        
        evaluation_axis.plot(
            np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
            training_loss_data,
            color = "blue",
            label = "MSE Loss")
        
        evaluation_axis.plot(
            np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
            validation_loss_history_array,
            color = "purple",
            label = "Validation Loss")
        
        evaluation_axis.set_xlabel('Epoch Number')
        evaluation_axis.set_ylabel('Scalar Loss')
        evaluation_axis.legend(fontsize = 20, shadow = True)
        evaluation_figure.savefig(f"loss_analytics_replica_{replica_index + 1}_v1.png")
        plt.close()


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