"""
This script will run the replica method with N replicas to produce
a *single* local fit of a given observable or set of observables.

## Notes:
(1): This will only run a *single* local fit, which translates to
fitting at a *single* kinematic setting.
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

def extract_kinematics(input_data):
    """
    ## Description:
    We need to extract the kinematics for the purpose of plotting.
    """
    _DECIMAL_ROUNDING_PARAMETER = 3

    q_squared = round(
        number = np.mean(input_data[_COLUMN_NAME_Q_SQUARED]),
        ndigits = _DECIMAL_ROUNDING_PARAMETER)
    
    x_bjorken = round(
        number = np.mean(input_data[_COLUMN_NAME_X_BJORKEN]),
        ndigits = _DECIMAL_ROUNDING_PARAMETER)
    
    t  = round(
        number = np.mean(input_data[_COLUMN_NAME_T_MOMENTUM_CHANGE]),
        ndigits = _DECIMAL_ROUNDING_PARAMETER)
    
    k  = round(
        number = np.mean(input_data[_COLUMN_NAME_LEPTON_MOMENTUM]),
        ndigits = _DECIMAL_ROUNDING_PARAMETER)
    
    return q_squared, x_bjorken, t, k

def plot_hyperplane_separations(
        current_replica_run_directory,
        replica_number,
        x_training,
        y_training,
        dnn_model):
    """
    ## Description:
    We construct a scatterplot that shows how aligned the model's predictions
    are with the data it was trained on. (This is mostly about the cross-section.)
    """

    # (X): Evaluate the network:
    y_predictions = dnn_model.predict(x_training).flatten()

    # (X): Compute residuals:
    residuals = np.abs(y_training - y_predictions)

    # (X): Normalize residuals for colormap scaling:
    residuals_normalized = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-8)

    # (X): Instantiate a figure object for the scatterplot:
    separation_figure = plt.figure(figsize = (8, 6))

    # (X): Add an Axes object to the figure:
    separation_axis = separation_figure.add_subplot(1, 1, 1)

    # (X): Add the scatter plot and return it:
    separation_scatterplot = separation_axis.scatter(
        y_training,
        y_predictions,
        c = residuals_normalized,
        cmap = "RdYlGn_r",
        alpha = 0.7)
    
    # (X): Add a colorbar to the *figure*:
    colorbar = separation_figure.colorbar(separation_scatterplot, ax = separation_axis)

    # (X): Annotate the colorbar:
    colorbar.set_label("Normalized Residual", fontsize = 16)

    # (X): Set the labels
    separation_axis.set_xlabel("True Cross Section", rotation = 0, fontsize = 18)
    separation_axis.set_ylabel("Predicted Cross Section", fontsize = 18)
    separation_axis.set_title("Model Fit: Prediction vs. Ground Truth", rotation = 0, fontsize = 20)

    # (X): Add a grid:
    separation_axis.grid(True)

    # (X): Compute the path to the directory:
    figure_savepath = f"{current_replica_run_directory}/{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_PERFORMANCE}"

    # (X): Save the figure:
    separation_figure.savefig(f"{figure_savepath}/distribution_of_predictions_replica_{replica_number}_v1.{_FIGURE_FORMAT_PNG}")

    # (X): Close the figure:
    plt.close(separation_figure)

def plot_cross_section_with_residuals_and_interpolation(
        current_replica_run_directory,
        replica_number,
        x_training,
        phi_values,
        true_values,
        dnn_model,
        fixed_kinematics_except_phi):
    """
    ## Description:
    Constructs a plot that shows the true cross-section values compared against the
    DNN's prediction of them, and then connects the two predictions vertically with a
    colored line, where the color of the line depends on how big the residual value and
    if it's positive/negative (blue/red).

    We also plot true vs. predicted cross sections with residual lines,
    plus a smooth interpolated DNN prediction curve.
    
    ## Notes:
    `fixed_kinematics_except_phi` should be shape (4,) matching 
    [q_squared, x_bjorken, t, k], everything except phi.
    """

    # (X): First, we need to predict the cross-section values:
    predicted_values = dnn_model.predict(x_training).flatten()

    # (X): Obtain the kinematic settings:
    q_squared_value, x_bjorken_value, t_value, k_value = extract_kinematics(x_training)
    
    # (X): Standard thing in regression: compute residuals:
    residuals = predicted_values - true_values

    # (X): Compute the biggest residual in the fit. We will
    # | use this later to vary the intensity of a "color value" in
    # | the histogram.
    max_abs_residual = np.max(np.abs(residuals))

    # (X): Perform normalization:
    normalized_resiudals = plt.Normalize(-max_abs_residual, max_abs_residual)

    # (X): Define a custom red/white/blue colormap:
    custom_colormap = mcolors.LinearSegmentedColormap.from_list(
        name = "residual_colormap",
        colors = [
            (0, 'blue'), (0.5, 'white'), (1, 'red')
            ])

    # (X): Set up the "residuals-only" figure:
    residuals_figure = plt.figure(figsize = (10, 6))

    # (X): Set up the interpolation figure:
    interpolation_figure = plt.figure(figsize = (10, 6))

    # (X): Add an Axis object to the residuals figure:
    residuals_axis = residuals_figure.add_subplot(1, 1, 1)

    # (X): Add an Axis object to the residuals figure:
    interplation_axis = interpolation_figure.add_subplot(1, 1, 1)

    # (X): To the residuals figure, add the true cross-section values:
    residuals_axis.scatter(
        x = phi_values,
        y = true_values,
        color = 'black',
        label = 'True Cross Section',
        zorder = 3,
        s = 4)
    
    # (X): Now, also add the predicted values:
    residuals_axis.scatter(
        x = phi_values,
        y = predicted_values,
        color = 'red',
        label = 'Predicted Cross Section',
        zorder = 3,
        s = 4)
    
    # (X): We add the true cross-section values to the interpolation figure, too!
    interplation_axis.scatter(
        x = phi_values,
        y = true_values,
        color = 'black',
        label = 'True Cross Section',
        zorder = 3,
        s = 4)
    
    # (X): Add the predicted values as well to the interpolation figure!
    interplation_axis.scatter(
        x = phi_values,
        y = predicted_values,
        color = 'red',
        label = 'Predicted Cross Section',
        zorder = 3,
        s = 4)

    # (X): Begin iteration over cross section values and residuals:
    for phi, true_cross_section, predicted_cross_section, residual in zip(phi_values, true_values, predicted_values, residuals):

        # (X): Map from the original interval of R to the 0-1 interval of R:
        color_value = normalized_resiudals(residual)

        # (X): Now, utilize cmap's ability to assign colors:
        color = custom_colormap(color_value)

        # (X): Plot a vertical line (understand why!) to connect the predicted and true cross-sections:
        residuals_axis.plot(
            [phi, phi],
            [true_cross_section, predicted_cross_section],
            color = color,
            linewidth = 1)
        
        # (X): Do the same for the interpolation figure:
        interplation_axis.plot(
            [phi, phi],
            [true_cross_section, predicted_cross_section],
            color = color,
            linewidth = 1)
        
    # (X): Construct a "densely-packed" array of azimuthal phi values for interpolation:
    phi_dense = np.linspace(0, 360, 500)

    # (X): Now, attach the phi values to the rest of the kinematic values:
    dense_inputs = np.column_stack([
        np.tile(fixed_kinematics_except_phi, (phi_dense.shape[0], 1)),
        phi_dense.reshape(-1, 1)
    ]).astype(np.float32)

    # (X): Run the inputs through the DNN model for predictions:
    dnn_predictions_dense = dnn_model.predict(dense_inputs, verbose = 0).flatten()

    # (X): Now actually *add* the interpolation:
    interplation_axis.plot(
        phi_dense,
        dnn_predictions_dense,
        color = "purple",
        linewidth = 2,
        label = "Interpolated DNN Prediction")
    
    # (X): Compute the title of the residuals plot:
    kinematic_settings_string = rf"$Q^2 = {q_squared_value:.2f}\ \mathrm{{GeV}}^2,\ x_{{\mathrm{{B}}}} = {x_bjorken_value:.3f},\ -t = {t_value:.3f}\ \mathrm{{GeV}}^2$"

    # (X): Set the labels for the residuals plots:
    residuals_axis.set_xlabel(r"Azimuthal Angle $\phi$ [degrees]", fontsize = 16)
    residuals_axis.set_ylabel("Cross Section", fontsize = 16)
    residuals_axis.set_title(f"Cross-Section Fitting Residuals for Replica {replica_number} with {kinematic_settings_string}", fontsize = 16)
    
    # (X): Set the labels for the interpolation plots:
    interplation_axis.set_xlabel(r"Azimuthal Angle $\phi$ [degrees]", fontsize = 16)
    interplation_axis.set_ylabel("Cross Section", fontsize = 16)
    interplation_axis.set_title(f"DNN Interpolation for Replica {replica_number} with {kinematic_settings_string}", fontsize = 16)
    
    # (X): We want the legend for both:
    residuals_axis.legend(shadow = True)
    interplation_axis.legend(shadow = True)

    # (X): We also want a grid for both:
    residuals_axis.grid(True)
    interplation_axis.grid(True)

    # (X): Let's compute the path to save it:
    figure_savepath = f"{current_replica_run_directory}/{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_PERFORMANCE}"
    
    # (X): Save the residuals figure:
    residuals_figure.savefig(f"{figure_savepath}/predicted_vs_true_x_section_{replica_number}_v1.{_FIGURE_FORMAT_PNG}")
    
    # (X): Save the interpolation figure:
    interpolation_figure.savefig(f"{figure_savepath}/dnn_interpolation_x_section_fit_{replica_number}_v1.{_FIGURE_FORMAT_PNG}")

    # (X): Close the plots:
    plt.close(interpolation_figure)
    plt.close(residuals_figure)


def create_relevant_directories(
        data_file_name: str,
        number_of_replicas: int,
        verbose: bool = False):
    """
    ## Description:
    A function that automates the construction of the several relevant folders
    used for the analysis of ML output.
    """

    # (1): We create a *unique* timestamp to name the analysis folder:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed timestamp to be: {timestamp}")

    # (2): We now use an f-string to compute the folder name:
    current_run_name = f"replica_run_{timestamp}"

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed current replica run to be: {current_run_name}")

    # (3): Use os.path to construct a path...
    current_run_folder = os.path.join(f"{os.getcwd()}/analysis", current_run_name)

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Determined run folder to be: {current_run_folder}")

    # (4): Make a bunch of subdirectories required for analysis:
    for subdirectory in REQUIRED_SUBDIRECTORIES_LIST:

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Now iterating over subdirectory: {subdirectory}")

        # (4.1): Join paths:
        full_path = os.path.join(current_run_folder, subdirectory)

        # (4.2): Use `makedirs` to enforce construction of folders:
        os.makedirs(full_path, exist_ok = True)

    # (5): Compute the path for the README file that contains infor about the *data*:
    data_readme_file_path_and_name = os.path.join(current_run_folder, "data/raw/README.md")

    # (6): Compute the path for the README file that will contain info about the *replicas*
    # | and other ANN stuff (e.g. hyperparameters):
    replicas_readme_file_path_and_name = os.path.join(current_run_folder, "data/replicas/README.md")

    # (7): Open the data README file to prepare to write:
    with open(
        file = data_readme_file_path_and_name,
        mode = "w",
        encoding = "utf-8") as new_data_readme:
        new_data_readme.write(f"# Raw Data for Replica Run on {timestamp}\n")
        new_data_readme.write("This folder contains the original data used to generate pseudodata for replicas.\n")
        new_data_readme.write(f"\n- Source data file: `{data_file_name}`\n")
        new_data_readme.close()

    # (8): Open the replica README file to prepare to write:
    with open(
        file = replicas_readme_file_path_and_name,
        mode = "w",
        encoding = "utf-8") as new_replica_readme:
        new_replica_readme.write(f"# Replicas for Replica Run on {timestamp} \n")
        new_replica_readme.write("This folder contains the .csv files and .keras model files for each replica.\n")
        new_replica_readme.write("The hyperparameters that characterize this replica DNN are:\n")
        new_replica_readme.write(f"- Number of replicas: {number_of_replicas}\n")
        new_replica_readme.write(f"- Number of epochs per replica: {_HYPERPARAMETER_NUMBER_OF_EPOCHS}\n")
        new_replica_readme.write(f"- Batch size: {_HYPERPARAMETER_BATCH_SIZE}\n")
        new_replica_readme.write(f"- Learning rate patience: {_HYPERPARAMETER_LR_PATIENCE}\n")
        new_replica_readme.write(f"- Learning rate factor: {_HYPERPARAMETER_LR_FACTOR}\n")
        new_replica_readme.write(f"- EarlyStop patience: {_HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER}\n")
        new_replica_readme.close()

    if SETTING_VERBOSE:
        print(f"> [VERBOSE]: Created replica analysis directory at: {current_run_folder}")

    return current_run_folder

def run_replica(
        kinematics_dataframe_name,
        replica_number,
        job_run_directory):
    """
    ## Description:
    Later!
    """

    # (1.2): Propose a replica name:
    current_replica_name = f"replica_{replica_number}"

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed replica name to be: {current_replica_name}")

    # (1.3): Immediately construct the filetype for the replica:
    model_file_name = f"{current_replica_name}.h5"

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed corresponding replica file name to be: {model_file_name}")

    # (X): Rely on Pandas to correctly read the just-generated .csv file:
    kinematics_dataframe_path = os.path.join('data', kinematics_dataframe_name)

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed path to data .csv file: {kinematics_dataframe_path}")

    # (X): Use Pandas' `.read_csv()` method to generate a corresponding DF:
    this_replica_data_set = pd.read_csv(kinematics_dataframe_path)

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Now printing the Pandas DF head using df.head():\n {this_replica_data_set.head()}")

    # (X): We now compute a *given* replica's DF --- it will *not* be the same as the original DF!
    generated_replica_data = generate_replica_data(pandas_dataframe = this_replica_data_set)

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Successfully generated replica data. Now displaying using df.head():\n {generated_replica_data.head()}")

    # (X): Use an f-string to compute the name *and location* of the file!
    computed_path_and_name_of_replica_data = f"{job_run_directory}/{_DIRECTORY_DATA}/{_DIRECTORY_DATA_RAW}/pseudodata_replica_{replica_number}_data.csv"

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed path and file name of current replica data: {computed_path_and_name_of_replica_data}")

    # (X): We also store the pseudodata/replica data for reproducability purposes:
    generated_replica_data.to_csv(
        path_or_buf = computed_path_and_name_of_replica_data,
        index_label = None)
    
    if SETTING_DEBUG:
        print("> [DEBUG]: Saved replica data!")

    # (X): Identify the "x values" for our model:
    raw_kinematics = generated_replica_data[[
        _COLUMN_NAME_Q_SQUARED,
        _COLUMN_NAME_X_BJORKEN,
        _COLUMN_NAME_T_MOMENTUM_CHANGE,
        _COLUMN_NAME_LEPTON_MOMENTUM,
        _COLUMN_NAME_AZIMUTHAL_PHI]]

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Obtained kinematic settings columns --- using .head() to display:\n{raw_kinematics.head()}")

    # (X): Obtain the cross section data from the replica dataframe:
    raw_cross_section = generated_replica_data[_COLUMN_NAME_CROSS_SECTION]

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Obtained cross-section column --- using .head() to display:\n{raw_cross_section.head()}")

    # (X): Obtain the associated cross section error from the replica dataframe:
    # raw_cross_section_error = generated_replica_data[_COLUMN_NAME_CROSS_SECTION_ERROR]

    raw_cross_section_error = this_replica_data_set[_COLUMN_NAME_CROSS_SECTION_ERROR]

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Obtained cross-section error column --- using .head() to display:\n{raw_cross_section_error.head()}")

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Example of numerical values of experimental kinematics: {raw_kinematics.iloc[0]}")

    if SETTING_DEBUG:
        print(f"> [DEBUG] Now showing min/max and big picture of the kinematic values: {raw_kinematics.describe()}")

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Example of numerical values of experimental cross-sections: {raw_cross_section.iloc[0]}")

    if SETTING_DEBUG:
        print(f"> [DEBUG] Now showing min/max and big picture of the cross-section values: {raw_cross_section.describe()}")

    if SETTING_DEBUG:
        print("> [DEBUG]: Sanity check sample rows:")
        for i in range(5):
            print(f"> [DEBUG]: Row {i} — Kinematics: {raw_kinematics.iloc[i].to_dict()} — Cross Section: {raw_cross_section.iloc[i]}")

    # (X): Detect if there are NaN values in the cross-section:
    assert not np.any(np.isnan(raw_cross_section.values)), "NaNs detected in cross section"

    # (X): Detect if there are INFINITIES in the cross-section --- this will break
    # | every TF thing we've ever done:
    assert not np.any(np.isinf(raw_cross_section.values)), "Infs detected in cross section"

    # (X): Use sklearn's traing/validation split function to split into training and testing data:
    x_training, x_validation, y_training, y_validation = train_test_split(
        raw_kinematics,
        raw_cross_section,
        test_size = _DNN_TRAIN_TEST_SPLIT_PERCENTAGE,)
        # random_state = 42)

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Partitioned data into train/test with split percentage of: {_DNN_TRAIN_TEST_SPLIT_PERCENTAGE}")

    # (X): Begin timing the replica time:
    start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
    
    if SETTING_DEBUG or SETTING_VERBOSE:
        print(f"> [VERBOSE]: Replica #{replica_number} started at {start_time_in_milliseconds}...")

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
    
    if SETTING_VERBOSE:
        print(f"> [VERBOSE]: Replica #{replica_number} finished running!")

    # (X): Compute the path that we'll store the replica:
    computed_path_of_replica_model = f"{job_run_directory}/{_DIRECTORY_DATA}/{_DIRECTORY_DATA_REPLICAS}/replica_{replica_number}.{_TF_FORMAT_KERAS}"

    if SETTING_DEBUG:
        print(f"> [DEBYG]: Computed path to replica storage: {computed_path_of_replica_model}")

    # (X): Now, save the replica:
    dnn_model.save(computed_path_of_replica_model)

    if SETTING_VERBOSE:
        print("> [VERBOSE] Saved replica!")

    if SETTING_DEBUG:
        print(f"> [VERBOSE] Saved replica to {computed_path_of_replica_model}")

    plot_hyperplane_separations(
        job_run_directory,
        replica_number,
        x_training,
        y_training,
        dnn_model)
    
    fixed_kinematics_except_phi = x_training.iloc[0][
            [_COLUMN_NAME_Q_SQUARED, _COLUMN_NAME_X_BJORKEN, _COLUMN_NAME_T_MOMENTUM_CHANGE, _COLUMN_NAME_LEPTON_MOMENTUM]
        ].to_numpy()
    
    plot_cross_section_with_residuals_and_interpolation(
        job_run_directory,
        replica_number,
        x_training,
        x_training[_COLUMN_NAME_AZIMUTHAL_PHI],
        y_training,
        dnn_model,
        fixed_kinematics_except_phi)

    # (X): Extract the 'loss' key from TF's history object. It has
    # | loss vs. epoch data on it:
    training_loss_data = neural_network_training_history.history['loss']

    # (X): Extract the 'val_loss' (validation loss) from the TF history object:
    validation_loss_history_array = neural_network_training_history.history['val_loss']
        
    # (X): Define a Figure object for plotting network loss:
    evaluation_figure = plt.figure(
        figsize = (10, 5.5))
    
    # (X): Add the subplot, which returns an Axes:
    evaluation_axis = evaluation_figure.add_subplot(1, 1, 1)
    
    # (X): Add a simple horizonal line that shows the *initial value* of the MSEl
    evaluation_axis.plot(
        np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
        np.array([np.max(training_loss_data) for number in training_loss_data]),
        color = "red",
        label = "Initial MSE Loss")
    
    # (X): Add a simple horizonal line that shows where MSE = 0:
    evaluation_axis.plot(
        np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
        np.zeros(shape = _HYPERPARAMETER_NUMBER_OF_EPOCHS),
        color = "green",
        label = r"MSE $=0$")
    
    # (X): Add a line plot that shows MSE loss vs. epoch:
    evaluation_axis.plot(
        np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
        training_loss_data,
        color = "blue",
        label = "MSE Loss")
    
    # (X): Add a line plot that shows the trend of validation loss vs. epoch:
    evaluation_axis.plot(
        np.arange(0, _HYPERPARAMETER_NUMBER_OF_EPOCHS, 1),
        validation_loss_history_array,
        color = "purple",
        label = "Validation Loss")
    
    # (X): Add a descriptive title:
    evaluation_axis.set_title(rf"Replica ${replica_number}$ Learning Curves")
    
    # (X): Add the x-label:
    evaluation_axis.set_xlabel('Epoch Number', rotation = 0, labelpad = 17.0, fontsize = 18)

    # (X): Add the y-label:
    evaluation_axis.set_ylabel('MSE', rotation = 0, labelpad = 26.0, fontsize = 18)

    # (X): Add the legend for clarity:
    plt.legend(fontsize = 17)

    # (X): Compute the string that will be the filename of the loss plot:
    current_replica_loss_plot_filename = f"{job_run_directory}/{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_LOSSES}/loss_analytics_replica_{replica_number}_v1"

    if SETTING_DEBUG or SETTING_VERBOSE:
        print(f"> Computed replica loss plot file destination:\n> {current_replica_loss_plot_filename}")

    # (X): Save a version of the figure according to .eps format for Overleaf stuff:
    evaluation_figure.savefig(
        fname = f"{current_replica_loss_plot_filename}.{_FIGURE_FORMAT_EPS}",
        format = _FIGURE_FORMAT_EPS)
    
    # (X): Save an immediately-visualizable figure with vector graphics:
    evaluation_figure.savefig(
        fname = f"{current_replica_loss_plot_filename}.{_FIGURE_FORMAT_SVG}",
        format = _FIGURE_FORMAT_SVG)
    
    # (X): Save an immediately-visualizable figure with vector graphics:
    evaluation_figure.savefig(
        fname = f"{current_replica_loss_plot_filename}.{_FIGURE_FORMAT_PNG}",
        format = _FIGURE_FORMAT_PNG)
    
    # (X): Closing figures:
    plt.close(evaluation_figure)

def main(
        kinematics_dataframe_name: str,
        number_of_replicas: int,
        verbose: bool = False):
    """
    ## Description:
    Main entry point to the local fitting procedure.
    """
    
    # (1): Enforce creation of required directory structure:
    current_replica_run_directory = create_relevant_directories(
        data_file_name = kinematics_dataframe_name,
        number_of_replicas = number_of_replicas)
    
    # (X): Determine devices' GPUs:
    gpus = tf.config.list_physical_devices('GPU')

    # (X): If there exist available GPUs...
    if gpus:

        # (X): ... begin iteration over each one and...
        for gpu in gpus:

            # (X): ... enforce growth of memory rather than using all of it:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # (1): Begin iteratng over the replicas:
    for replica_index in range(number_of_replicas):

        # (1.1): Obtain the replica number by adding 1 to the index:
        replica_number = replica_index + 1

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed replica number to be: {replica_number}")

        run_replica(
            kinematics_dataframe_name = kinematics_dataframe_name,
            replica_number = replica_number,
            job_run_directory = current_replica_run_directory)

    # make_predictions(
    #     current_replica_run_directory = current_replica_run_directory,
    #     input_data = raw_kinematics)


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