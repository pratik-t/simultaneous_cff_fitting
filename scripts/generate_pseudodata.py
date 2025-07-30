"""
We use this script to generate pseudodata for training the model.
"""

# Native Package | os
import os

# utilities > km15
from utilities.km15 import compute_km15_cffs

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Pandas:
import pandas as pd

# 3rd Party Library | DifferentialCrossSection
from bkm10_lib.core import DifferentialCrossSection

# 3rd Party Library | BKM10Inputs
from bkm10_lib.inputs import BKM10Inputs

# 3rd Party Library | CFFInputs
from bkm10_lib.cff_inputs import CFFInputs

# static_strings > "bin"
from statics.static_strings import _COLUMN_NAME_KINEMATIC_BIN

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

# static_strings > .eps
from statics.static_strings import _FILE_FORMAT_CSV

_TEST_BEAM_K = 5.75
_TEST_Q_SQUARED = 1.82
_TEST_X_BJORKEN = 0.343
_TEST_T = -0.172

# (X): Range of beam energy --- units in GeV:
K_RANGE = np.linspace(5.5, 11.0, 6)

# (X): Range of Q^{2} --- units in GeV^{2}:
Q2_RANGE = np.linspace(1.0, 5.0, 9)

# (X): Range of x_{B} --- dimensionless!
X_B_RANGE = np.linspace(0.1, 0.6, 6)

# (X): Range of t --- units in GeV^{2}:
T_RANGE = np.linspace(-1.0, -0.1, 10)

# (X): For all of these, we want phi to range from 0 to 360 degrees:
_PHI_ARRAY = np.linspace(0, 2 * np.pi, 15)

SETTING_VERBOSE = False
SETTING_DEBUG = False

def is_physical_cross_section(cross_section):
    return (
        np.all(np.isfinite(cross_section)) and
        np.all(cross_section >= 0) and
        not np.any(np.isnan(cross_section))
    )

def generate_kinematic_scan():

    # (X): Initialize empty array that *will* become the DF:
    rows = []

    set_counter = 0  # Tracks unique kinematic settings
    total_settings = len(K_RANGE) * len(Q2_RANGE) * len(X_B_RANGE) * len(T_RANGE)

    for fixed_k in K_RANGE:
        for fixed_q_squared in Q2_RANGE:
            for fixed_x_bjorken in X_B_RANGE:
                for fixed_t in T_RANGE:
                    set_counter += 1

                    try:
                        
                        # (X): Use the KM15 GPD model to come up with the CFFs:
                        real_h, imag_h, real_e, real_ht, imag_ht, real_et = compute_km15_cffs(fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_k)

                        # # (X): TEMPORARY! DO NOT HAVE KM15 FOR Im[E] or Im[Et]:
                        # imag_e, imag_et = 0.903, 5.383

                        # (X): TEMPORARY! DO NOT HAVE KM15 FOR Im[E] or Im[Et]:
                        imag_e, imag_et = 0., 0.

                        # (X): Pass in the BKM10 kinematic settings:
                        kinematic_inputs = BKM10Inputs(
                            lab_kinematics_k = fixed_k,
                            squared_Q_momentum_transfer = fixed_q_squared,
                            x_Bjorken = fixed_x_bjorken,
                            squared_hadronic_momentum_transfer_t = fixed_t)

                        # (X): Pass in the CFFInputs as KM15 predicts:
                        cff_inputs = CFFInputs(
                            compton_form_factor_h = complex(real_h, imag_h),
                            compton_form_factor_h_tilde = complex(real_ht, imag_ht),
                            compton_form_factor_e = complex(real_e, imag_e),
                            compton_form_factor_e_tilde = complex(real_et, imag_et),
                        )

                        # (X): Specify the target polarization *as a float*:
                        target_polarization = 0.

                        # (X): Specify the beam polarization *as a float*:
                        lepton_polarization = 0.0

                        # (X): We are using the WW relations in this computation:
                        ww_setting = True

                        # (X): Using the setting we wrote earlier, we now need to construct
                        # | all of it into a dictionary that will be passed into the main
                        # | class. It's a lot of information, but we need all of it in order
                        # | for the BKM formalism to evaluate.
                        config_dictionary = {
                            "kinematics": kinematic_inputs,
                            "cff_inputs": cff_inputs,
                            "target_polarization": target_polarization,
                            "lepton_beam_polarization": lepton_polarization,
                            "using_ww": ww_setting
                        }

                        # (X): Instantiate the class for the cross-section:
                        cross_section = DifferentialCrossSection(
                            configuration = config_dictionary,
                            verbose = SETTING_VERBOSE,
                            debugging = SETTING_DEBUG)

                        cross_section_values = cross_section.compute_cross_section(_PHI_ARRAY).real

                        # (X): TEMPORARY! 5% ERRORS:
                        sigma_stat_plus_array = 0.05 * cross_section_values
                        sigma_stat_minus_array = sigma_stat_plus_array

                        if not is_physical_cross_section(cross_section_values):

                            print(f"> [ERROR]: [set = {set_counter}] Unphysical XS for k = {fixed_k}, Q² = {fixed_q_squared}, xB = {fixed_x_bjorken}, t = {fixed_t}")

                        else:

                            for phi, sigma, sigma_plus, sigma_minus in zip(
                                _PHI_ARRAY, cross_section_values, sigma_stat_plus_array, sigma_stat_minus_array
                            ):
                                rows.append({
                                    _COLUMN_NAME_KINEMATIC_BIN: set_counter,
                                    _COLUMN_NAME_LEPTON_MOMENTUM: fixed_k,
                                    _COLUMN_NAME_Q_SQUARED: fixed_q_squared,
                                    _COLUMN_NAME_X_BJORKEN: fixed_x_bjorken,
                                    _COLUMN_NAME_T_MOMENTUM_CHANGE: fixed_t,
                                    _COLUMN_NAME_AZIMUTHAL_PHI: phi,
                                    _COLUMN_NAME_CROSS_SECTION: sigma,
                                    "sigma_stat_plus": sigma_plus,
                                    "sigma_stat_minus": sigma_plus,
                                    "Re[H]": real_h,
                                    "Im[H]": imag_h,
                                    "Re[E]": real_e,
                                    "Im[E]": imag_e,
                                    "Re[Ht]": real_ht,
                                    "Im[Ht]": imag_ht,
                                    "Re[Et]": real_et,
                                    "Im[Et]": imag_et,
                                    })

                    except Exception as e:
                        print(f"> [ERROR]: [set = {set_counter}] Failed at k = {fixed_k}, Q² = {fixed_q_squared}, xB = {fixed_x_bjorken}, t = {fixed_t}\n → {str(e)}")
                        continue

    df = pd.DataFrame(rows)
    
    # (X): Dynamically compute the file path: # (X): Rely on Pandas to correctly read the just-generated .csv file:
    computed_filepath = os.path.join("data", f"total_kinematic_scan_v2.{_FILE_FORMAT_CSV}")

    df.to_csv(computed_filepath, index = False)
    
    print(f"> Saved {len(df)} rows to {computed_filepath}")
    print(f"> Attempted {total_settings} total kinematic settings")
    print(f"> Physical settings retained: {df[_COLUMN_NAME_KINEMATIC_BIN].nunique()}")
    print(f"> Angles per setting: {len(_PHI_ARRAY)}")

def generate():
    """
    ## Description:
    The main function to generate pseudodata.
    """

    # (X): Fix the beam energy k:
    fixed_k = _TEST_BEAM_K

    # (X): Fix the value of Q^{2}:
    fixed_q_squared = _TEST_Q_SQUARED

    # (X): Fix the value of x_{B}:
    fixed_x_bjorken = _TEST_X_BJORKEN

    # (X): Fix the value of t:
    fixed_t = _TEST_T
    
    # (X): Use the KM15 GPD model to come up with the CFFs:
    real_h, imag_h, real_e, real_ht, imag_ht, real_et = compute_km15_cffs(fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_k)

    # (X): TEMPORARY! DO NOT HAVE KM15 FOR Im[E] or Im[Et]:
    imag_e, imag_et = 0.903, 5.383

    # (X): Obtain the BKM10 inputs:
    example_1_kinematic_inputs = BKM10Inputs(
        lab_kinematics_k = fixed_k,
        squared_Q_momentum_transfer = fixed_q_squared,
        x_Bjorken = fixed_x_bjorken,
        squared_hadronic_momentum_transfer_t = fixed_t)

    # (X): Make sure to specify the CFF inputs:
    example_1_cff_inputs = CFFInputs(
    compton_form_factor_h = complex(-0.897, 2.421),
    compton_form_factor_h_tilde = complex(2.444, 1.131),
    compton_form_factor_e = complex(-0.541, 0.903),
    compton_form_factor_e_tilde = complex(2.207, 5.383))

    # (X): Specify the target polarization *as a float*:
    example_1_target_polarization = 0.

    # (X): Specify the beam polarization *as a float*:
    example_1_lepton_polarization = 0.0

    # (X): We are using the WW relations in this computation:
    example_1_ww_setting = True

    # (X): Using the setting we wrote earlier, wvaluese now need to construct
    # | all of it into a dictionary that will be passed into the main
    # | class. It's a lot of information, but we need all of it in order
    # | for the BKM formalism to evaluate.
    example_1_config_dictionary = {
        "kinematics": example_1_kinematic_inputs,
        "cff_inputs": example_1_cff_inputs,
        "target_polarization": example_1_target_polarization,
        "lepton_beam_polarization": example_1_lepton_polarization,
        "using_ww": example_1_ww_setting
    }

    # (X): Instantiate the class for the cross-section:
    example_1_cross_section = DifferentialCrossSection(
        configuration = example_1_config_dictionary,
        verbose = SETTING_VERBOSE,
        debugging = SETTING_DEBUG)

    # (X): `compute_cross_section` returns an array of cross-section values:
    cross_section_values = example_1_cross_section.compute_cross_section(_PHI_ARRAY)
    print(cross_section_values)

    # (X): 
    example_1_cross_section.plot_cross_section(_PHI_ARRAY)

    # (X): TEMPORARY! 5% ERRORS:
    sigma_stat_plus = 0.05 * cross_section_values
    sigma_stat_minus = sigma_stat_plus

    # (X): Create the DataFrame
    df = pd.DataFrame({
        "k": fixed_k,
        "q_squared": fixed_q_squared,
        "x_b": fixed_x_bjorken,
        "t": fixed_t,
        "phi": _PHI_ARRAY,
        "sigma": cross_section_values,
        "sigma_stat_plus": sigma_stat_plus,
        "sigma_stat_minus": sigma_stat_minus,
        "Re[H]": real_h,
        "Im[H]": imag_h,
        "Re[E]": real_e,
        "Im[E]": imag_e,
        "Re[Ht]": real_ht,
        "Im[Ht]": imag_ht,
        "Re[Et]": real_et,
        "Im[Et]": imag_et,
    })

    # (X): Dynamically compute the file path: # (X): Rely on Pandas to correctly read the just-generated .csv file:
    computed_filepath = os.path.join("data", f"synthetic_km15_data.{_FILE_FORMAT_CSV}")

    # (X): Save to CSV
    df.to_csv(computed_filepath, index = False)


if __name__ == "__main__":
    # generate()

    generate_kinematic_scan()
