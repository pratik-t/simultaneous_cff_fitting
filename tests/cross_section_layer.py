"""
"""

import tensorflow as tf
import numpy as np

from models.architecture import CrossSectionLayer


kinematics_np = np.array([
    [2.0, 0.3, -0.2, 10.6, 30.0],
    [1.5, 0.2, -0.1, 11.0, 45.0],
    [3.0, 0.4, -0.25, 10.0, 60.0]
], dtype = np.float32)

cffs_np = np.array([
    [1.0, 0.1, 0.5, 0.05, 0.2, 0.01, 0.3, 0.02],
    [0.9, 0.2, 0.6, 0.06, 0.25, 0.015, 0.35, 0.03],
    [1.1, 0.15, 0.55, 0.04, 0.22, 0.012, 0.33, 0.025],
], dtype = np.float32)

kinematics_tf = tf.convert_to_tensor(kinematics_np)

cffs_tf = tf.convert_to_tensor(cffs_np)

cross_section_layer = CrossSectionLayer(
    target_polarization = 0.0,
    lepton_beam_polarization = 0.0,
    using_ww = True)

input_tensor = tf.concat([kinematics_tf, cffs_tf], axis=-1)
cross_sections = cross_section_layer(input_tensor)

tf.print("Computed cross sections:", cross_sections)

cross_sections_np = cross_sections.numpy()
print("Cross section values:", cross_sections_np)
