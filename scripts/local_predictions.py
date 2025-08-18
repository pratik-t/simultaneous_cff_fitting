"""
## Description:
Once we have performed a local fit, we can make predicitons of the CFFs.

## Notes:
"""

# Native Library | argparse
import argparse

# Native Library | gc
import gc

# Native Library | os
import os

# Native Library | re
import re

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

# 3rd Party Library | SciPy:
from scipy.stats import norm

# 3rd Party Library | tqdm:
from tqdm import tqdm

# (X): In order to correctly deserialize a TF model, you need to define
# | custom objects when loading it. And so that requires that we
# | actually import the damn custom layers we made:
from models.architecture import CrossSectionLayer, BSALayer

# utilities > km15
from utilities.km15 import compute_km15_cffs

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

# static_strings > /data
from statics.static_strings import _DIRECTORY_ANALYSIS

# static_strings > /replicas
from statics.static_strings import _DIRECTORY_REPLICAS

# static_strings > /data
from statics.static_strings import _DIRECTORY_DATA

# static_strings > /data/raw
from statics.static_strings import _DIRECTORY_DATA_RAW

# static_strings > /data/replicas
from statics.static_strings import _DIRECTORY_DATA_REPLICAS

# static_strings > /replicas/fits
from statics.static_strings import _DIRECTORY_REPLICAS_FITS

# static_strings > .keras
from statics.static_strings import _TF_FORMAT_KERAS

# static_strings > .eps
from statics.static_strings import _FIGURE_FORMAT_EPS

# static_strings > .svg
from statics.static_strings import _FIGURE_FORMAT_SVG

# static_strings > .png
from statics.static_strings import _FIGURE_FORMAT_PNG

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

def extract_cff_layer_output(model, input_data):
    """
    ## Description:
    Extracts the CFFs from an intermediate layer of the model.
    Recall that in our architecture the CFFs are an intermediate
    layer of 8 nodes that are *then* used to generate cross-
    section and BSA data.
    """
    intermediate_layer_model = tf.keras.Model(
        inputs = model.input,
        outputs = model.get_layer("cff_output_layer").output)
    
    return intermediate_layer_model.predict(input_data)


def get_replica_model_paths(current_replica_run_path):
    """
    ## Description:
    A basic function that finds where the relevant .keras
    replica files are, sorts them, and returns the sorted
    array. Notice that what is returned is just a *list* of
    *paths*!
    """
    return sorted([
        os.path.join(current_replica_run_path, filename)
        for filename in os.listdir(current_replica_run_path)
        if filename.endswith(_TF_FORMAT_KERAS)
    ])

def make_predictions(
        current_run_directory,
        input_data,
        verbose: bool = True):
    """
    ## Description:
    Assuming the replica method was performed, we now make
    predictions with the replica averages.
    """
    
    # (X): Compute the path that we'll store the replica:
    computed_path_of_replica_model = f"{current_run_directory}/{_DIRECTORY_DATA}/{_DIRECTORY_DATA_REPLICAS}"

    # (X): Compute the path of the plots that we'll store the predictions in:
    computed_path_to_plots = f"{current_run_directory}/{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_FITS}"

    # (X): Find a list of paths to the current replicas:
    replica_paths = get_replica_model_paths(computed_path_of_replica_model)

    # (X): Save this in memory so we can use it for plots later:
    number_of_replicas = len(replica_paths)

    if SETTING_VERBOSE:
        print(f"> [VERBOSE]: Found {number_of_replicas} replicas.")

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Found {number_of_replicas} replicas.")

    # (X): Use an f-string to compute the name *and location* of the file!
    computed_path_and_name_of_replica_data = f"{current_run_directory}/{_DIRECTORY_DATA}/{_DIRECTORY_DATA_RAW}/pseudodata_replica_1_data.csv"

    input_data = pd.read_csv(computed_path_and_name_of_replica_data)[[
        _COLUMN_NAME_Q_SQUARED,
        _COLUMN_NAME_X_BJORKEN,
        _COLUMN_NAME_T_MOMENTUM_CHANGE,
        _COLUMN_NAME_LEPTON_MOMENTUM,
        _COLUMN_NAME_AZIMUTHAL_PHI]]

    # (X): Obtain the kinematic settings:
    q_squared_value, x_bjorken_value, t_value, k_value = extract_kinematics(input_data)

    # (X): Compute the title of the residuals plot:
    kinematic_settings_string = rf"$Q^2 = {q_squared_value:.2f}\ \mathrm{{GeV}}^2,\ x_{{\mathrm{{B}}}} = {x_bjorken_value:.3f},\ t = {t_value:.3f}\ \mathrm{{GeV}}^2$"

    # (X): Initalize a list to append CFF predictions:
    all_predictions = []

    # (X): Start iterating over every replica found in the directory:
    for replica in tqdm(
        iterable = replica_paths,
        desc = "Evaluating replicas",
        colour = "green"):

        # (X): We are using a custom layer for CFF evaluation, and Keras is suspicious of that:
        keras.config.enable_unsafe_deserialization()

        # (X.Y): Obtain *the* TF replica model:
        replica_model = tf.keras.models.load_model(
            replica,
            compile = False,
            custom_objects = {
                "CrossSectionLayer": CrossSectionLayer,
                "BSALayer": BSALayer
            })

        # (X.Y): Run through the models makingpredictions:
        predicted_cffs = extract_cff_layer_output(replica_model, input_data)

        # (X.Y): Note the shape of `all_predictions` will be crazy:
        all_predictions.append(predicted_cffs)

    # (X):
    all_predictions = np.array(all_predictions)

    # (X): The last element in the tuple will be the number of CFFs:
    number_of_cffs = all_predictions.shape[-1]

    # (X): Compute the mean of each CFF by using .mean() along axis 1.
    # | This is why it's important to have an idea of what the predictions
    # | array looks like:
    mean_predictions = np.mean(all_predictions, axis = 1)

    # (X): TEMPORARY! Write out the names of the CFFs:
    cff_names = ["Re[H]", "Im[H]", "Re[E]", "Im[E]", "Re[Ht]", "Im[Ht]", "Re[Et]", "Im[Et]"]

    # (X): Prepare to evaluate the KM15 model by extracting the kinematics:
    q_squared, x_bjorken, t = (input_data[_COLUMN_NAME_Q_SQUARED], input_data[_COLUMN_NAME_X_BJORKEN], input_data[_COLUMN_NAME_T_MOMENTUM_CHANGE])

    # (X): Get the KM15 values of the CFFs:
    real_h_km15, imag_h_km15, real_e_km15, real_ht_km15, imag_ht_km15, real_et_km15 = compute_km15_cffs(q_squared.values[0], x_bjorken.values[0], t.values[0])

    # (X): Package CFFs in list corresponding index-wise the the right CFF in `cff_names` above:
    km15_cff_values = [real_h_km15, imag_h_km15, real_e_km15, 0.0, real_ht_km15, imag_ht_km15, real_et_km15, 0.0]

    # (X): Initialize an array containing the 8 CFF predictions *for returning from this function*!:
    cff_mean_predictions = []

    # (X): Now, begin making the predictions:
    for index, cff_name in enumerate(cff_names):

        # (X): Query the i-th "row" of this predictions array:
        data = mean_predictions[:, index]

        # (X): Run a fit to a Gaussian function immediately:
        gaussian_mean, gaussian_stddev = norm.fit(data)

        # (X): Initialize a figure instance for plotting:
        cff_prediction_figure = plt.figure(figsize = (10, 5.5))
        
        # (X): Add the subplot, which returns an Axes:
        cff_prediction_axis = cff_prediction_figure.add_subplot(1, 1, 1)

        # (X): Add a histogram object to the axis:
        cff_prediction_axis.hist(
            data,
            bins = 30,
            density = True,
            alpha = 0.6,
            color = 'skyblue',
            edgecolor = 'black')

        # (X): We need an iterable for the Gaussian fit which will be a *line* to .plot() with:
        burner_x_values_for_gaussian_fit = np.linspace(data.min(), data.max(), 200)

        # (X): Now, fit to a Gaussian and plot the line:
        cff_prediction_axis.plot(
            burner_x_values_for_gaussian_fit,
            norm.pdf(burner_x_values_for_gaussian_fit, gaussian_mean, gaussian_stddev),
            color = "red",
            linestyle = "--",
            label = fr"Gaussian Fit: $\mu = {gaussian_mean:.3f}$, $\sigma = {gaussian_stddev:.3f}$")
        
        # (X): 
        cff_mean_predictions.append(gaussian_mean)
        
        # (X): We extract the corresponding KM15 prediction for the CFF:
        km15_value = km15_cff_values[index]

        # (X): We now plot the KM15 prediction for the given CFF:
        cff_prediction_axis.axvline(
            km15_value,
            color = 'green',
            linestyle = '-',
            linewidth = 2,
            label = f"KM15: {km15_value:.3f}")
        
        # (X): Set the title:
        cff_prediction_axis.set_title(rf"${cff_name}$ Distribution Across $N_{{\mathrm{{replicas}}}} = {number_of_replicas}$ at {kinematic_settings_string}")

        # (X): Set the x-label:
        cff_prediction_axis.set_xlabel(f"${cff_name}$ Value", rotation = 0, labelpad = 17.0, fontsize = 18)

        # (X): Set the y-label:
        cff_prediction_axis.set_ylabel("Density", rotation = 0, labelpad = 17.0, fontsize = 18)

        # (X): Add a legend:
        plt.legend()

        # (X): Enforce a tight layout:
        plt.tight_layout()

        # (X): Save a version of the figure according to .eps format for Overleaf stuff:
        cff_prediction_figure.savefig(
            fname = f"{computed_path_to_plots}/{cff_name}_histogram.{_FIGURE_FORMAT_EPS}",
            format = _FIGURE_FORMAT_EPS)
        
        # (X): Save an immediately-visualizable figure with vector graphics:
        cff_prediction_figure.savefig(
            fname = f"{computed_path_to_plots}/{cff_name}_histogram.{_FIGURE_FORMAT_SVG}",
            format = _FIGURE_FORMAT_SVG)
        
        # (X): Save an immediately-visualizable figure with vector graphics:
        cff_prediction_figure.savefig(
            fname = f"{computed_path_to_plots}/{cff_name}_histogram.{_FIGURE_FORMAT_PNG}",
            format = _FIGURE_FORMAT_PNG)
        
        # (X): Close the figure to avoid memory explosions and etc.:
        plt.close(cff_prediction_figure)

        if SETTING_VERBOSE or SETTING_DEBUG:
            print(f"> [VERBOSE]: Saved: {computed_path_to_plots}")

    if SETTING_VERBOSE or SETTING_DEBUG:
        print("> [VERBOSE]: All histograms generated!")

    # (X): Clear the memory of the replica model:
    del replica_model

    # () Use Keras backend to clear the session:
    K.clear_session()

    # (X): Use garbage collector to clear the memory:
    gc.collect()

    return k_value, q_squared_value, x_bjorken_value, t_value, cff_mean_predictions

def plot_cff_surfaces(cff_predictions_dataframe: pd.DataFrame):
    """
    ## Description:

    """
    x_bjorken = cff_predictions_dataframe[_COLUMN_NAME_X_BJORKEN]
    q_squared = cff_predictions_dataframe[_COLUMN_NAME_Q_SQUARED]
    t = cff_predictions_dataframe[_COLUMN_NAME_T_MOMENTUM_CHANGE]
    minus_t = -t

    fixed_q_squared = q_squared.median()

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed median value of Q^2: {fixed_q_squared}")
        
    fixed_x_bjorken = x_bjorken.median()

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed median value of xB: {fixed_x_bjorken}")

    fixed_t = t.median()

    if SETTING_DEBUG:
        print(f"> [DEBUG]: Computed median value of t: {fixed_t}")

    # (X): TEMPORARY! Write out the names of the CFFs:
    cff_names = ["Re[H]", "Im[H]", "Re[E]", "Im[E]", "Re[Ht]", "Im[Ht]", "Re[Et]", "Im[Et]"]

    # (X): Now, begin making the predictions:
    for index, cff_name in enumerate(cff_names):

        restricted_q_squared = cff_predictions_dataframe[np.isclose(q_squared, fixed_q_squared, rtol = 0.1)]
        restricted_x_bjorken = cff_predictions_dataframe[np.isclose(x_bjorken, fixed_x_bjorken, rtol = 0.1)]
        restricted_t = cff_predictions_dataframe[np.isclose(t, fixed_t, rtol = 0.1)]

        # (X): Set up the CFF vs. -t & x_{B} figure:
        t_versus_x_bjorken_figure = plt.figure(figsize = (10, 6))

        # (X): Set up the CFF vs. Q^{2} & x_{B} figure:
        q_squared_versus_x_bjorken_figure = plt.figure(figsize = (10, 6))

        # (X): Set up the CFF vs. -t & Q^{2} figure:
        t_versus_q_squared_figure = plt.figure(figsize = (10, 6))

        # (X): Add an Axis object to the -t vs. x_{B} figure:
        t_versus_x_bjorken_axis = t_versus_x_bjorken_figure.add_subplot(1, 1, 1, projection = "3d")

        # (X): Add an Axis object to the Q^{2} vs. x_{B} figure:
        q_squared_versus_x_bjorken_axis = q_squared_versus_x_bjorken_figure.add_subplot(1, 1, 1, projection = "3d")

        # (X): Add an Axis object to the -t vs. Q^{2} figure:
        t_versus_q_squared_axis = t_versus_q_squared_figure.add_subplot(1, 1, 1, projection = "3d")

        t_versus_x_bjorken_axis.scatter(
            restricted_q_squared[_COLUMN_NAME_X_BJORKEN],
            -restricted_q_squared[_COLUMN_NAME_T_MOMENTUM_CHANGE],
            restricted_q_squared[cff_name],
            alpha = 0.9,
            s = 10.4)
        t_versus_x_bjorken_axis.set_xlabel(r"$x_B$")
        t_versus_x_bjorken_axis.set_ylabel(r"$-t$")
        t_versus_x_bjorken_axis.set_title(rf"$-t$ vs. $x_B$ at $Q^2 = {fixed_q_squared:.2f}$")

        q_squared_versus_x_bjorken_axis.scatter(
            restricted_t[_COLUMN_NAME_X_BJORKEN],
            -restricted_t[_COLUMN_NAME_Q_SQUARED],
            restricted_t[cff_name],
            alpha = 0.9,
            s = 10.4)
        q_squared_versus_x_bjorken_axis.set_xlabel(r"$x_B$")
        q_squared_versus_x_bjorken_axis.set_ylabel(r"$Q^2$")
        q_squared_versus_x_bjorken_axis.set_title(rf"$Q^2$ vs. $x_B$ at $-t = {fixed_t:.2f}$")

        t_versus_q_squared_axis.scatter(
            restricted_x_bjorken[_COLUMN_NAME_Q_SQUARED],
            -restricted_x_bjorken[_COLUMN_NAME_T_MOMENTUM_CHANGE],
            restricted_x_bjorken[cff_name],
            alpha = 0.9,
            s = 10.4)
        t_versus_q_squared_axis.set_xlabel(r"$Q^2$")
        t_versus_q_squared_axis.set_ylabel(r"$-t$")
        t_versus_q_squared_axis.set_title(rf"$-t$ vs. $Q^2$ at $x_B = {fixed_x_bjorken:.2f}$")
        
        t_versus_x_bjorken_figure.savefig(fname = f"{cff_name}_t_vs_xb_v1.png", dpi = 300)
        q_squared_versus_x_bjorken_figure.savefig(fname = f"{cff_name}_q_vs_xb_v1.png", dpi = 300)
        t_versus_q_squared_figure.savefig(fname = f"{cff_name}_t_vs_q_v1.png", dpi = 300)

        plt.close(t_versus_x_bjorken_figure)
        plt.close(q_squared_versus_x_bjorken_figure)
        plt.close(t_versus_q_squared_figure)

def local_predictions(
        current_replica_run_directory: str,
        input_data: str,
        verbose: bool = False):
    """
    Later!
    """

    # (X): Find the *name* of the replica run:
    computed_path_of_replica_run = f"{_DIRECTORY_ANALYSIS}/{current_replica_run_directory}/"

    # (X): Find how many kinematic bins were tested:
    kinematic_setting_directories = [
        directory for directory in os.listdir(computed_path_of_replica_run)
        if os.path.isdir(os.path.join(computed_path_of_replica_run, directory))
        ]

    results = []

    for kinematic_setting_index, kinematic_setting_name in enumerate(kinematic_setting_directories):

        if SETTING_VERBOSE:
            print(f"> [VERBOSE]: Now making predictions for kinematic set #{kinematic_setting_index + 1}")

        path_to_kinematic_set = f"{computed_path_of_replica_run}/{kinematic_setting_name}"

        # (X): Perform the prediction process:
        k_value, q_squared_value, x_bjorken_value, t_value, cff_mean_predictions = make_predictions(
            current_run_directory = path_to_kinematic_set,
            input_data = False)
        
        # (X): Constuct the row corresponding to the fixed kienamtic settings:
        # | Note the ordered list of CFFs is:  ["Re[H]", "Im[H]", "Re[E]", "Im[E]", "Re[Ht]", "Im[Ht]", "Re[Et]", "Im[Et]"]
        row = {
            'bin': kinematic_setting_index,
            'k': k_value,
            'q_squared': q_squared_value,
            'x_b': x_bjorken_value,
            't': t_value,
            'Re[H]': cff_mean_predictions[0],
            'Im[H]': cff_mean_predictions[1],
            'Re[E]': cff_mean_predictions[2],
            'Im[E]': cff_mean_predictions[3],
            'Re[Ht]': cff_mean_predictions[4],
            'Im[Ht]': cff_mean_predictions[5],
            'Re[Et]': cff_mean_predictions[6],
            'Im[Et]': cff_mean_predictions[7],
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("all_cff_predictions.csv", index=False)


if __name__ == "__main__":

    # (1): Create an instance of the ArgumentParser
    parser = argparse.ArgumentParser(description = "Placeholder")

    # (2): Enforce the path to the datafile:
    parser.add_argument(
        '-r',
        '--replica_file',
        type = str,
        required = True,
        help = "Placeholder")
    
    # (2): Enforce the path to the datafile:
    parser.add_argument(
        '-d',
        '--datafile',
        type = str,
        required = True,
        help = "Placeholder")

    # (5): Ask, but don't enforce debugging verbosity:
    parser.add_argument(
        '-v',
        '--verbose',
        required = False,
        action = 'store_false',
        help = "Placeholder")
    
    arguments = parser.parse_args()

    local_predictions(
        current_replica_run_directory = arguments.replica_file,
        input_data = arguments.datafile,
        verbose = arguments.verbose)
    
    df = pd.read_csv("all_cff_predictions.csv")
    
    plot_cff_surfaces(df)