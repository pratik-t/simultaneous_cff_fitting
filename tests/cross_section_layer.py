"""
We provide a test that serves to numerically cross-check the results of 
eager-execution of the CrossSectionLayer. This is important to verify if
the TF version of the BKM10 computation checks out with earlier-computed
numerics using the BKM10 library and script.
"""

# Native Library | datetime
import datetime

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib
import matplotlib.pyplot as plt

# 3rd Party Library | TensorFlow
import tensorflow as tf

from models.architecture import CrossSectionLayer

SETTING_VERBOSE = True
SETTING_DEBUG = True

def test_cross_section_vs_phi_plot(
        target_polarization: float = 0.0,
        lepton_helicity: float = 0.0,
        ww_setting: bool = True):
    """
    ## Description:
    We just create a plot that we can use for the cross-checking with the
    extant BKM10 code and BKM10 library.
    """

    if SETTING_DEBUG:
        print(f"> Running with target polarization setting of: {target_polarization}")

    if SETTING_DEBUG:
        print(f"> Running with lepton beam polarization setting of: {lepton_helicity}")

    if SETTING_DEBUG:
        print(f"> Are we computing with the WW relations on? {ww_setting}")

    # (X): Obtain the TF CrossSection layer to prepare for eager evalation:
    cross_section_computation = CrossSectionLayer(
        target_polarization = target_polarization,
        lepton_beam_polarization = lepton_helicity,
        using_ww = ww_setting)

    # (X): Fix the kinematic values *in order*: [Q², x_B, t, k]:
    fixed_kinematics_values = [1.82, 0.34, -0.17, 5.75]

    # (X): Standard CFF values *in order*: [Re[H], Im[H], Re[Ht], Im[Ht], Re[E], Im[H], Re[Et], Im[Et]]:
    cffs_values = [-0.897, 2.421, 2.444, 1.131, -0.541, 0.903, 2.207, 5.383]

    # (X): 0° to 360° in 1° steps
    phi_values = np.arange(0., 361., 1., dtype = np.float32)

    # (X): Initialize array that will store iterated kinematics with varying angles:
    all_inputs = []

    # (X): Attach all the phi values to the fixed kinematics array:
    for phi in phi_values:

        # (X): "list-wise" addition:
        kinematics = fixed_kinematics_values + [phi]

        # (X): More "list-wise" addition:
        full_input = kinematics + cffs_values

        # (X): Now add the newly-constructed list to the *main* list:
        all_inputs.append(full_input)

    # (X): Cast the result of the iteration option into a NumPY array:
    all_inputs_np = np.array(all_inputs, dtype = np.float32)

    # (X): Convert to TF tensor:
    all_inputs_tf = tf.convert_to_tensor(all_inputs_np, dtype = tf.float32)

    # (X): Pass the inputs (as tensors):
    computed_cross_sections = cross_section_computation(all_inputs_tf).numpy().flatten()

    if SETTING_VERBOSE:
        print("> [VERBOSE]: Cross sections computed!")
        
    if SETTING_DEBUG:
        print(f"> [DEBUG]: Cross sections computed:\n{computed_cross_sections}")

    # (X): Initialize a figure:
    cross_section_figure = plt.figure(figsize = (8, 5))

    # (X): Add an Axis object to it:
    cross_section_axis = cross_section_figure.add_subplot(1, 1, 1)

    # (X): Plot the relevant stuff:
    cross_section_axis.plot(
        phi_values,
        computed_cross_sections,
        linestyle = "none",
        color = "red",
        alpha = 0.65,
        marker = ".")

    # (X): Set the x-label:
    cross_section_axis.set_xlabel(r"Azimuthal Angle $\phi$ ($\deg$)")

    # (X): Set the y-label:
    cross_section_axis.set_ylabel(r"Differential Cross Section ($nb/GeV^{4}$)")

    # (X): Set the title:
    cross_section_axis.set_title(r"Cross Section vs. $\phi$ with Fixed Kinematics")

    # (X): Add a grid to the plot:
    cross_section_axis.grid(True)
    
    # (X): We create a *unique* timestamp to name the image:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # (X): Initializing the plot name:
    plot_filename = f"x_sec_plot_unp_beam_unp_target_{timestamp}"

    # (X): Some logic that determines the name of the file:
    if lepton_helicity == 0.0:
        if target_polarization != 0.0:
            plot_filename = f"x_sec_plot_unp_beam_unp_target_{timestamp}"

        else:
            plot_filename = f"x_sec_plot_unp_beam_polarized_target_{timestamp}"

    elif lepton_helicity == 1.0:
        if target_polarization != 0.0:
            plot_filename = f"x_sec_plot_plus_beam_unp_target_{timestamp}"

        else:
            plot_filename = f"x_sec_plot_plus_beam_polarized_target_{timestamp}"

    elif lepton_helicity == -1.0:
        if target_polarization != 0.0:
            plot_filename = f"x_sec_plot_minus_beam_unp_target_{timestamp}"

        else:
            plot_filename = f"x_sec_plot_minus_beam_polarized_target_{timestamp}"

    cross_section_figure.savefig(fname = plot_filename)

    plt.close(cross_section_figure)

test_cross_section_vs_phi_plot()
