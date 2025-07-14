"""
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | Matplotlib
import matplotlib.pyplot as plt

# 3rd Party Library | TensorFlow
import tensorflow as tf

from models.architecture import CrossSectionLayer

# (X): Initialize three types of kinematic bins:
kinematics_np = np.array([
    [2.0, 0.3, -0.2, 10.6, 30.0],
    [1.5, 0.2, -0.1, 11.0, 45.0],
    [3.0, 0.4, -0.25, 10.0, 60.0]
], dtype = np.float32)

# (X): Initialize three types of CFF settings --- remember
# | what each slot stands for!
cffs_np = np.array([
    [1.0, 0.1, 0.5, 0.05, 0.2, 0.01, 0.3, 0.02],
    [0.9, 0.2, 0.6, 0.06, 0.25, 0.015, 0.35, 0.03],
    [1.1, 0.15, 0.55, 0.04, 0.22, 0.012, 0.33, 0.025],
], dtype = np.float32)

# (X): Convert the arrays to tensors so TF doesn't explode:
kinematics_tf = tf.convert_to_tensor(kinematics_np)
cffs_tf = tf.convert_to_tensor(cffs_np)

# (X): Instantiate the custom class for computations:
cross_section_layer = CrossSectionLayer(
    target_polarization = 0.0,
    lepton_beam_polarization = 0.0,
    using_ww = True)

# (X): Prepare a tensor to insert into the layer --- involves
# | concatenation!
input_tensor = tf.concat([kinematics_tf, cffs_tf], axis = -1)

# (X): Now just plug-and-chug:
cross_sections = cross_section_layer(input_tensor)
tf.print("> Computed cross sections:\n", cross_sections)

# (X): Obtain the NumPy representation:
cross_sections_np = cross_sections.numpy()
print(f"> Computed cross section values:\n{cross_sections_np}")


def test_cross_section_vs_phi_plot():

    # (X): Obtain the TF CrossSection layer:
    cross_section_layer = CrossSectionLayer(
        target_polarization = 0.0,
        lepton_beam_polarization = 0.0,
        using_ww = True)

    # (X): Fix the kinematic values *in order*: [Q², x_B, t, k]:
    fixed_kinematics_values = [1.82, 0.34, -0.17, 5.75]

    # (X): Standard CFF values *in order*: [Re[H], Im[H], Re[Ht], Im[Ht], Re[E], Im[H], Re[Et], Im[Et]]:
    cffs_values = [-0.897, 2.421, 2.444, 1.131, -0.541, 0.903, 2.207, 5.383]

    # (X): 0° to 360° in 1° steps
    phi_values = np.arange(0, 361, 1, dtype = np.float32)

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
    computed_cross_sections = cross_section_layer(all_inputs_tf).numpy().flatten()

    print(cross_sections)

    # (X): Plot all the stuff:
    plt.figure(figsize = (8, 5))
    plt.plot(phi_values, computed_cross_sections, marker = ".", linestyle = "none", color = "red", alpha = 0.65)
    plt.xlabel(r"Azimuthal Angle $\phi$ ($\deg$)")
    plt.ylabel(r"Differential Cross Section ($nb/GeV^{4}$)")
    plt.title(r"Cross Section vs. $\phi$ with Fixed Kinematics")
    plt.grid(True)
    plt.show()

print("> Now running plots...")
test_cross_section_vs_phi_plot()
