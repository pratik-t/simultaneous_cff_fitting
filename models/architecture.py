"""
Here, we define the DNN model architecture used for 
any fitting procedure.
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | TensorFlow
import tensorflow as tf

# 3rd Party Library | bkm10:
from bkm10_lib import DifferentialCrossSection, CFFInputs, BKM10Inputs, backend, BKMFormalism

# 3rd Party Library | TensorFlow:
from tensorflow.keras.layers import Input

# 3rd Party Library | TensorFlow:
from tensorflow.keras.layers import Concatenate

# 3rd Party Library | TensorFlow:
from tensorflow.keras.layers import Dense

# 3rd Party Library | TensorFlow:
from tensorflow.keras.layers import Lambda

# 3rd Party Library | TensorFlow:
from tensorflow.keras.models import Model

# 3rd Party Library | TensorFlow:
from tensorflow.keras.utils import register_keras_serializable

from models.loss_functions import simultaneous_fit_loss

from statics.static_strings import _HYPERPARAMETER_LEARNING_RATE
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5

from statics.constants import _MASS_OF_PROTON_IN_GEV, _ELECTROMAGNETIC_FINE_STRUCTURE_CONSTANT, _ELECTRIC_FORM_FACTOR_CONSTANT, _PROTON_MAGNETIC_MOMENT

SETTING_VERBOSE = True
SETTING_DEBUG = False

# (X): EXTREMELY CAREFUL! THIS IS TEMPORARY!
# tf.config.run_functions_eagerly(True)

@register_keras_serializable()
class CrossSectionLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            target_polarization = 0.0,
            lepton_beam_polarization = 0.0,
            using_ww = True,
            **kwargs):
        
        # (1): Inherit Layer class properties:
        super().__init__(**kwargs)

        # (2): Obtain the target polarization:
        self.target_polarization = target_polarization

        # (3): Obtain the beam polarization:
        self.lepton_beam_polarization = lepton_beam_polarization

        # (4): Decide if we're using the WW relations:
        self.using_ww = using_ww

    def call(self, inputs):
        """
        ## Description:
        This function is *required* in order to tell TF what to do when we register this
        as a layer of some ANN. It should contain all the logic that is needed. Our goal here 
        is to anticipate the passage of 5 + 4 different values --- in order, they are: lepton
        beam energy, photon virtuality, Bjorken x, hadronic momentum transfer, the azimuthal 
        angle phi, and the four CFFs. With these 9 inputs, we *compute* a single output that 
        we call the cross section.
        """

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Received inputs: {inputs}")
        
        # (1): Extract only the kinematics, which are *in order*: [Q², x_B, t, k, φ]:
        kinematics = inputs[..., :5]

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Extracted kinematics part of TF layer inputs: {kinematics}")

        # (2): Extract only the CFFs, what are *in order*: [Re[H], Im[H], Re[Ht], Im[Ht], Re[E], Im[H], Re[Et], Im[Et]]:
        cffs = inputs[..., 5:]

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Extracted CFF part of TF layer inputs: {cffs}")

        # (3): To please TF, concatenate the inputs first:
        concatenated_layer_input = [kinematics, cffs]

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Casted/concatenated kinematics and CFFs for passage into TF layer: {concatenated_layer_input}")

        # (4): Immediately pass the concatenated array into the layer's "computation function":
        differential_cross_section = self.compute_cross_section(concatenated_layer_input)

        # (Note): We were not able to successfully use the bkm10 library here due to its complicated
        # | use of the native `complex` class. When `complex` multiplies floats in standard Python or
        # | NumPy, everything is fine. But when TF tries to do this, it requires that essentially
        # | everything be `complex64` because the CFFs are of this type. It will require major 
        # | refactoring in order to discard the use of the `complex` data type in the bkm10 lib, and
        # | we do not have time at the moment to do it.

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed cross section values:\n{differential_cross_section}")

        # (5): Return the computation: a *single value* for the cross-section:
        return differential_cross_section
    
    @tf.function
    def compute_cross_section(self, inputs):
        """
        ## Description:
        This is a *panic* function that will compute ALL of the required
        coefficients that go into the cross section *and* the cross-section
        itself.
        """

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Received inputs: {inputs}")

        # (1): Unpack the inputs into the CFFs and the kinematics.
        # | The inputs will be a KerasTensor of shape (None, 5) and another
        # | KerasTensor of shape (None, 8). That is, the five kinematic
        # | quantities and the eight numbers for the CFFs.
        kinematics, cffs = inputs

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Obtained kinematics from inputs: {kinematics}")

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Obtained CFFs from inputs: {cffs}")

        # (2): Extract the eight CFFs from the DNN:
        real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et = tf.unstack(cffs, axis = -1)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Unstacked CFFs\n> {real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et}")

        # (3): Extract the kinematics from the DNN:
        q_squared, x_bjorken, t, k, phi = tf.unstack(kinematics, axis = -1)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Unstacked kinematics\n> {q_squared, x_bjorken, t, k, phi}")

        # (4): Compute epsilon:
        epsilon = self.calculate_kinematics_epsilon(q_squared, x_bjorken)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed epsilon: {epsilon[0]}")

        # (5): Compute "y":
        y = self.calculate_kinematics_lepton_energy_fraction_y(q_squared, k, epsilon)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed lepton_energy_fraction: {y[0]}")

        # (6): Comute skewness "xi":
        xi = self.calculate_kinematics_skewness_parameter(q_squared, x_bjorken, t)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed skewness: {xi[0]}")

        # (7): Calculate t_minimum
        t_min = self.calculate_kinematics_t_min(q_squared, x_bjorken, epsilon)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed t mimumum: {t_min[0]}")

        # (8): Calculate t':
        t_prime = self.calculate_kinematics_t_prime(t, t_min)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed t prime: {t_prime[0]}")

        # (9): Calculate Ktilde:
        k_tilde = self.calculate_kinematics_k_tilde(q_squared, x_bjorken, y, t, epsilon, t_min)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed K tilde: {k_tilde[0]}")

        # (10): Calculate K:
        capital_k = self.calculate_kinematics_k(q_squared, y, epsilon, k_tilde)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed K: {capital_k[0]}")

        # (11): Calculate k.delta:
        k_dot_delta = self.calculate_k_dot_delta(q_squared, x_bjorken, t, phi, epsilon, y, capital_k)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed K.delta: {k_dot_delta}")

        # (12): Calculate P_{1}:
        p1 = self.calculate_lepton_propagator_p1(q_squared, k_dot_delta)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed propagator p1: {p1}")

        # (13): Calculate P_{2}:
        p2 = self.calculate_lepton_propagator_p2(q_squared, t, k_dot_delta)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed propagator p2: {p2}")

        # (14): Calculate the Electric Form Factor
        fe = self.calculate_form_factor_electric(t)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed F_E: {fe[0]}")

        # (15): Calculate the Magnetic Form Factor
        fg = self.calculate_form_factor_magnetic(fe)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed F_G: {fg[0]}")

        # (16): Calculate the Pauli Form Factor, F2:
        f2 = self.calculate_form_factor_pauli_f2(t, fe, fg)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed F_2: {f2[0]}")

        # (17): Calculate the Dirac Form Factor, F1:
        f1 = self.calculate_form_factor_dirac_f1(fg, f2)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed F_1: {f1[0]}")

        # (18): Calculate prefactor:
        prefactor = self.calculate_bkm10_cross_section_prefactor(q_squared, x_bjorken, epsilon, y)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed BKM10 cross-section prefactor: {prefactor}")

        # (X): Obtain the prefactor for the interference contribution:
        interference_prefactor = tf.constant(1.0, dtype = tf.float32) / (x_bjorken * y**3 * t * p1 * p2)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed interference contribution prefactor: {interference_prefactor}")

        if self.lepton_beam_polarization == 0.:

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Lepton beam detected to be unpolarized: {self.lepton_beam_polarization}")

            contribution_plus = self.calculate_interference_contribution(
                tf.constant(1.0, dtype = tf.float32), q_squared, x_bjorken, t, phi, f1, f2,
                real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht,
                epsilon, y, xi, t_prime, k_tilde, capital_k)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Calculated first contribution to unpolarized cross section: {contribution_plus}")
            
            contribution_minus = self.calculate_interference_contribution(
                tf.constant(-1.0, dtype = tf.float32), q_squared, x_bjorken, t, phi, f1, f2,
                real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht,
                epsilon, y, xi, t_prime, k_tilde, capital_k)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Calculated finals contribution to unpolarized cross section: {contribution_minus}")
            
            # (X): Sum together all the Interference contributions:
            interference_contribution = interference_prefactor * tf.constant(0.5, dtype = tf.float32) * (contribution_plus + contribution_minus)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Calculated interference contribution to amplitude squared: {interference_contribution}")

        elif self.lepton_beam_polarization in (1.0, -1.0):

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Lepton beam detected to be polarized: {self.lepton_beam_polarization}")

            # (X): Sum together all the Interference contributions:
            interference_contribution = interference_prefactor * self.calculate_interference_contribution(
                tf.constant(self.lepton_beam_polarization, dtype = tf.float32), q_squared, x_bjorken, t, phi, f1, f2,
                real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht,
                epsilon, y, xi, t_prime, k_tilde, capital_k)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Calculated interference contribution to amplitude squared: {interference_contribution}")

        else:

            raise NotImplementedError(f"> [ERROR]: The lepton beam value you have chosen, {self.lepton_beam_polarization}, is not supported.")

        # (X): Sum together all the BH contributions:
        # | This is 0 for now!
        bh_contribution = tf.zeros_like(prefactor)

        # (X): Sum together all the DVCS contributions:
        # | This is 0 for now!
        dvcs_contribution = tf.zeros_like(prefactor)

        # (X): Compute the cross-section:
        cross_section = self.convert_to_nb_over_gev4(prefactor * (bh_contribution + dvcs_contribution + interference_contribution))

        # (X): A first pass of computing the cross section:
        # cross_section = real_H**2 + imag_H**2 + tf.constant(0.5, dtype = tf.float32) * tf.cos(phi) * real_E + 0.1 * q_ssquared

        # (X): A second pass of computing the cross section:
        # | This is important: If you do not use *all* of the inputs given to the network, then
        # | TensorFlow will complain that there is nothing to compute gradients with respect to.
        # | This second pass revealed this. The earlier version of it did NOT include any CFFs,
        # | and TensorFlow complained that there were no gradients. All we had to do was multiply by a 
        # | single CFF, and everything worked.
        # cross_section = (prefactor * c0pp_tf * tf.cos(0. * phi)) * real_H**2 + imag_H**2 + tf.constant(0.5, dtype = tf.fdloat32) * tf.cos(phi) * real_E + 0.1 * q_squared

        return cross_section
    
    @tf.function
    def calculate_interference_contribution(
        self,
        lepton_helicity,
        q_squared,
        x_bjorken,
        t,
        phi,
        f1,
        f2,
        real_H,
        imag_H,
        real_Ht,
        imag_Ht,
        real_E,
        imag_E,
        epsilon,
        y,
        xi,
        t_prime,
        k_tilde,
        capital_k):
        
        if self.target_polarization == 0.:

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Target is unpolarized: {self.target_polarization}")

            # (19): Calculate the Curly C:
            curly_c_i_real, curly_c_i_imag = self.calculate_curly_C_unpolarized_interference(
                q_squared, x_bjorken, t, f1, f2, real_H, imag_H, real_Ht, imag_Ht, real_E, imag_E)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed real part of Curly C^I: {curly_c_i_real[0]}")
                print(f"> [DEBUG]: Computed imaginary part of Curly C^I: {curly_c_i_imag[0]}")
            
            # (20): Calculate the Curly C,V:
            curly_c_i_v_real, curly_c_i_v_imag = self.calculate_curly_C_unpolarized_interference_V(
                q_squared, x_bjorken, t, f1, f2, real_H, imag_H, real_E, imag_E)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed real part of Curly C^I,V: {curly_c_i_v_real[0]}")
                print(f"> [DEBUG]: Computed imaginary part of Curly C^I,V: {curly_c_i_v_imag[0]}")
            
            # (21): Calculate the Curly C,A:
            curly_c_i_a_real, curly_c_i_a_imag = self.calculate_curly_C_unpolarized_interference_A(
                q_squared, x_bjorken, t, f1, f2, real_Ht, imag_Ht)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed real part of Curly C^I,A: {curly_c_i_a_real[0]}")
                print(f"> [DEBUG]: Computed imaginary part of Curly C^I,A: {curly_c_i_a_imag[0]}")
            
            # (22): Calculate the common factor:
            common_factor = (tf.sqrt(tf.constant(2.0, dtype = tf.float32) / q_squared) * k_tilde / (tf.constant(2.0, dtype = tf.float32) - x_bjorken))

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed modulating factor on all Curly C^I with effective CFFs: {common_factor}")
            
            # (X): Calculate the Curly C with effective form factors:
            curly_c_i_real_eff, curly_c_i_imag_eff = self.calculate_curly_C_unpolarized_interference(
                q_squared, x_bjorken, t, f1, f2,
                self.compute_cff_effective(xi, real_H, self.using_ww),
                self.compute_cff_effective(xi, imag_H, self.using_ww),
                self.compute_cff_effective(xi, real_Ht, self.using_ww),
                self.compute_cff_effective(xi, imag_Ht, self.using_ww),
                self.compute_cff_effective(xi, real_E, self.using_ww),
                self.compute_cff_effective(xi, imag_E, self.using_ww))
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed first part of real part of Curly C^I with Feff: {curly_c_i_real_eff[0]}")
                print(f"> [DEBUG]: Computed first part of imaginary part of Curly C^I with Feff: {curly_c_i_imag_eff[0]}")
            
            # (X): Multiply the common factor with the Curly C^I thanks to TensorFlow...
            curly_c_i_real_eff = common_factor * curly_c_i_real_eff 
            curly_c_i_imag_eff = common_factor * curly_c_i_imag_eff

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Finally computed real part of Curly C^I with Feff: {curly_c_i_real_eff[0]}")
                print(f"> [DEBUG]: Finally computed imaginary part of Curly C^I with Feff: {curly_c_i_imag_eff[0]}")
            
            # (X): Calculate the Curly C,V with effective form factors:
            curly_c_i_v_real_eff, curly_c_i_v_imag_eff = self.calculate_curly_C_unpolarized_interference_V(
                q_squared, x_bjorken, t, f1, f2,
                self.compute_cff_effective(xi, real_H, self.using_ww),
                self.compute_cff_effective(xi, imag_H, self.using_ww),
                self.compute_cff_effective(xi, real_E, self.using_ww),
                self.compute_cff_effective(xi, imag_E, self.using_ww))
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed first part of real part of Curly C^I,V with Feff: {curly_c_i_v_real_eff[0]}")
                print(f"> [DEBUG]: Computed first part of imaginary part of Curly C^I,V with Feff: {curly_c_i_v_imag_eff[0]}")
            
            # (X): Multiply the common factor with the Curly C^I,V thanks to TensorFlow...
            curly_c_i_v_real_eff = common_factor * curly_c_i_v_real_eff
            curly_c_i_v_imag_eff = common_factor * curly_c_i_v_imag_eff

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Finally computed real part of Curly C^I,V with Feff: {curly_c_i_v_real_eff[0]}")
                print(f"> [DEBUG]: Finally computed imaginary part of Curly C^I,V with Feff: {curly_c_i_v_imag_eff[0]}")
            
            # (X): Calculate the Curly C,A with effective form factors:
            curly_c_i_a_real_eff, curly_c_i_a_imag_eff = self.calculate_curly_C_unpolarized_interference_A(
                q_squared, x_bjorken, t, f1, f2,
                self.compute_cff_effective(xi, real_Ht, self.using_ww),
                self.compute_cff_effective(xi, imag_Ht, self.using_ww))
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed first part of real part of Curly C^I,A with Feff: {curly_c_i_a_real_eff[0]}")
                print(f"> [DEBUG]: Computed first part of imaginary part of Curly C^I,A with Feff: {curly_c_i_a_imag_eff[0]}")

            # (X): Multiply the common factor with the Curly C^I,A thanks to TensorFlow...
            curly_c_i_real_eff = common_factor * curly_c_i_real_eff
            curly_c_i_a_real_eff = common_factor * curly_c_i_a_real_eff

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Finally computed real part of Curly C^I,A with Feff: {curly_c_i_real_eff[0]}")
                print(f"> [DEBUG]: Finally computed imaginary part of Curly C^I,A with Feff: {curly_c_i_a_real_eff[0]}")

            # (X): Compute the three C++(n = 0) unpolarized coefficients with TF:
            c0pp_tf = self.calculate_c_0_plus_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C0++ (c0pp_tf): {c0pp_tf[0]}")

            c0ppv_tf = self.calculate_c_0_plus_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV0++ (c0ppv_tf): {c0ppv_tf[0]}")

            c0ppa_tf = self.calculate_c_0_plus_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA0++ (c0ppa_tf): {c0ppa_tf[0]}")

            # (X): Compute the three C++(n = 1) unpolaried coefficients with TF:
            c1pp_tf = self.calculate_c_1_plus_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C1++ (c1pp_tf): {c1pp_tf[0]}")

            c1ppv_tf = self.calculate_c_1_plus_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, t_prime, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV1++ (c1ppv_tf): {c1ppv_tf[0]}")

            c1ppa_tf = self.calculate_c_1_plus_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, t_prime, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA1++ (c1ppa_tf): {c1ppa_tf[0]}")

            # (X): Compute the three C++(n = 2) unpolaried coefficients with TF:
            c2pp_tf = self.calculate_c_2_plus_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, t_prime, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C2++ (c2pp_tf): {c2pp_tf[0]}")

            c2ppv_tf = self.calculate_c_2_plus_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, t_prime, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV2++ (c2ppv_tf): {c2ppv_tf[0]}")
                
            c2ppa_tf = self.calculate_c_2_plus_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, t_prime, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA2++ (c2ppa_tf): {c2ppa_tf[0]}")

            # (X): Compute the three C++(n = 3) unpolaried coefficients with TF:
            c3pp_tf = self.calculate_c_3_plus_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C3++ (c3pp_tf): {c3pp_tf[0]}")
                
            c3ppv_tf = self.calculate_c_3_plus_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV3++ (c3ppv_tf): {c3ppv_tf[0]}")

            c3ppa_tf = self.calculate_c_3_plus_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, t_prime, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA3++ (c3ppa_tf): {c3ppa_tf[0]}")

            # (X): Compute the three C0+(n = 0) unpolarized coefficients with TF:
            c00p_tf = self.calculate_c_0_zero_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C0+ (c00p_tf): {c00p_tf[0]}")

            c00pv_tf = self.calculate_c_0_zero_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV00+ (c00pv_tf): {c00pv_tf[0]}")

            c00pa_tf = self.calculate_c_0_zero_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA00+ (c00pa_tf): {c00pa_tf[0]}")

            # (X): Compute the three C0+(n = 1) unpolaried coefficients with TF:
            c10p_tf = self.calculate_c_1_zero_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, t_prime)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C1++ (c10p_tf): {c10p_tf[0]}")

            c10pv_tf  = self.calculate_c_1_zero_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV10+ (c10pv_tf): {c10pv_tf[0]}")

            c10pa_tf  = self.calculate_c_1_zero_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA10+ (c10pa_tf): {c10pa_tf[0]}")

            # (X): Compute the three C0+(n = 2) unpolaried coefficients with TF:
            c20p_tf = self.calculate_c_2_zero_plus_unpolarized(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C2++ (c20p_tf): {c20p_tf[0]}")

            c20pv_tf = self.calculate_c_2_zero_plus_unpolarized_V(q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV20+ (c20pv_tf): {c20pv_tf[0]}")

            c20pa_tf = self.calculate_c_2_zero_plus_unpolarized_A(q_squared, x_bjorken, t, epsilon, y, t_prime, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA20+ (c20pa_tf): {c20pa_tf[0]}")

            # (X): Compute the three C0+(n = 3) unpolaried coefficients with TF:
            c30p_tf = tf.zeros_like(c0pp_tf)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized C3++ (c30p_tf): {c30p_tf[0]}")

            c30pv_tf = tf.zeros_like(c0pp_tf)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CV30+ (c30pv_tf): {c30pv_tf[0]}")

            c30pa_tf = tf.zeros_like(c0pp_tf)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized CA30+ (c30pa_tf): {c30pa_tf[0]}")

            # (X): Compute the three S++(n = 1) unpolaried coefficients with TF:
            s1pp_tf = self.calculate_s_1_plus_plus_unpolarized(lepton_helicity, q_squared, x_bjorken, epsilon, y, t_prime, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S1++ (s1pp_tf): {s1pp_tf[0]}")

            s1ppv_tf = self.calculate_s_1_plus_plus_unpolarized_V(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S1V++ (s1ppv_tf): {s1ppv_tf[0]}")

            s1ppa_tf = self.calculate_s_1_plus_plus_unpolarized_A(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, t_prime, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S1A++ (s1ppa_tf): {s1ppa_tf[0]}")

            # (X): Compute the three S++(n = 2) unpolaried coefficients with TF:
            s2pp_tf = self.calculate_s_2_plus_plus_unpolarized(lepton_helicity, q_squared, x_bjorken, epsilon, y, t_prime)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S2++ (s2pp_tf): {s2pp_tf[0]}")
                
            s2ppv_tf = self.calculate_s_2_plus_plus_unpolarized_V(lepton_helicity, q_squared, x_bjorken, t, epsilon, y)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S2V++ (s2ppv_tf): {s2ppv_tf[0]}")

            s2ppa_tf = self.calculate_s_2_plus_plus_unpolarized_A(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, t_prime)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized SA2++ (s2ppa_tf): {s2ppa_tf[0]}")

            # (X): Compute the three S0+(n = 1) unpolaried coefficients with TF:
            s10p_tf = self.calculate_s_1_zero_plus_unpolarized(lepton_helicity, q_squared, epsilon, y, k_tilde)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S10+ (s10p_tf): {s10p_tf[0]}")

            s10pv_tf  = self.calculate_s_1_zero_plus_unpolarized_V(lepton_helicity, q_squared, x_bjorken, t, epsilon, y)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized SV10+ (s10pv_tf): {s10pv_tf[0]}")

            s10pa_tf  = self.calculate_s_1_zero_plus_unpolarized_A(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized SA10+ (s10pa_tf): {s10pa_tf[0]}")

            # (X): Compute the three S0+(n = 2) unpolaried coefficients with TF:
            s20p_tf = self.calculate_s_2_zero_plus_unpolarized(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S20+ (s20p_tf): {s20p_tf[0]}")

            s20pv_tf = self.calculate_s_2_zero_plus_unpolarized_V(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S2V0+ (s20pv_tf): {s20pv_tf[0]}")

            s20pa_tf = self.calculate_s_2_zero_plus_unpolarized_A(lepton_helicity, q_squared, x_bjorken, t, epsilon, y, capital_k)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized S2A0+ (s20pa_tf): {s20pa_tf[0]}")
            
            # (X): Obtain the c_{0} coefficient:
            c_0 = (
                c0pp_tf * curly_c_i_real + c0ppv_tf * curly_c_i_v_real + c0ppa_tf * curly_c_i_a_real +
                c00p_tf * curly_c_i_real_eff + c00pv_tf * curly_c_i_v_real_eff + c00pa_tf * curly_c_i_a_real_eff)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized c_0: {c_0[0]}")
            
            # (X): Obtain the c_{1} coefficient:
            c_1 = (
                c1pp_tf * curly_c_i_real + c1ppv_tf * curly_c_i_v_real + c1ppa_tf * curly_c_i_a_real +
                c10p_tf * curly_c_i_real_eff + c10pv_tf * curly_c_i_v_real_eff + c10pa_tf * curly_c_i_a_real_eff)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized c_1: {c_1[0]}")
            
            # (X): Obtain the c_{2} coefficient:
            c_2 = (
                c2pp_tf * curly_c_i_real + c2ppv_tf * curly_c_i_v_real + c2ppa_tf * curly_c_i_a_real +
                c20p_tf * curly_c_i_real_eff + c20pv_tf * curly_c_i_v_real_eff + c20pa_tf * curly_c_i_a_real_eff)
            
            # c_2 = lepton_helicity * tf.constant(-0.03259012849881058, dtype = tf.float32) * tf.ones_like(c0pp_tf)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized c_2: {c_2[0]}")
            
            # (X): Obtain the c_{3} coefficient:
            c_3 = (
                c3pp_tf * curly_c_i_real + c3ppv_tf * curly_c_i_v_real + c3ppa_tf * curly_c_i_a_real +
                c30p_tf * curly_c_i_real_eff + c30pv_tf * curly_c_i_v_real_eff + c30pa_tf * curly_c_i_a_real_eff)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized c_3: {c_3[0]}")
            
            # (X): Obtain the s_{1} coefficient:
            s_1 = (
                s1pp_tf * curly_c_i_imag + s1ppv_tf * curly_c_i_v_imag + s1ppa_tf * curly_c_i_a_imag +
                s10p_tf * curly_c_i_imag_eff + s10pv_tf * curly_c_i_v_imag_eff + s10pa_tf * curly_c_i_a_imag_eff)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized s_1: {s_1[0]}")
            
            # (X): Obtain the s_{2} coefficient:
            s_2 = (
                s2pp_tf * curly_c_i_imag + s2ppv_tf * curly_c_i_v_imag + s2ppa_tf * curly_c_i_a_imag +
                s20p_tf * curly_c_i_imag_eff + s20pv_tf * curly_c_i_v_imag_eff + s20pa_tf * curly_c_i_a_imag_eff)
            
            if SETTING_DEBUG:
                print(f"> [DEBUG]: Computed unpolarized s_2: {s_2[0]}")

            interference =  (
                c_0 * tf.cos(tf.constant(0.0, dtype = tf.float32) * (tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(phi))) +
                c_1 * tf.cos(tf.constant(1.0, dtype = tf.float32) * (tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(phi))) +
                c_2 * tf.cos(tf.constant(2.0, dtype = tf.float32) * (tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(phi))) +
                c_3 * tf.cos(tf.constant(3.0, dtype = tf.float32) * (tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(phi))) +
                s_1 * tf.sin(tf.constant(1.0, dtype = tf.float32) * (tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(phi))) +
                s_2 * tf.sin(tf.constant(2.0, dtype = tf.float32) * (tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(phi))))
            
            return interference
        
        else:

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Target detected to be polarized: {self.target_polarization}")

            raise NotImplementedError("Not yet...")
                
    
    @tf.function
    def convert_degrees_to_radians(self, degrees):
        """
        ## Description:
        Converts a number in degrees (0-360) to radians
        using the standard formula.
        """
        return (degrees * tf.constant(np.pi, dtype = tf.float32) / tf.constant(180.0, dtype = tf.float32))
    
    @tf.function
    def convert_to_nb_over_gev4(self, number: float) -> float:
        """
        ## Description:
        Convert a number in units of GeV^{-6} to nb/GeV^{4}. For reference,
        the number is 389379 or about 3.9e6 (= 4.0e6), and it is 
        multiplied by whatever `number` is passed in.

        ## Arguments:

            1. number (float)

        ## Returns:

            1. number_in_nb_over_GeV4 (float)
        """
        _CONVERSION_FACTOR = .389379 * 1000000.
        number_in_nb_over_GeV4 = tf.constant(_CONVERSION_FACTOR, dtype = tf.float32) * number
        return number_in_nb_over_GeV4

    @tf.function
    def calculate_kinematics_epsilon(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate Epsilon right away:
            epsilon = (tf.constant(2.0, dtype = tf.float32) * x_Bjorken * _MASS_OF_PROTON_IN_GEV) / tf.sqrt(squared_Q_momentum_transfer)

            # (tf.constant(1.0, dtype = tf.float32)1): If verbose, print the result:
            if verbose:
                tf.print(f"> Calculated epsilon to be:\n{epsilon}")

            # (2): Return Epsilon:
            return epsilon
        
        except Exception as ERROR:
            tf.print(f"> Error in computing kinematic epsilon:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
    
    @tf.function
    def calculate_kinematics_lepton_energy_fraction_y(
        self,
        squared_Q_momentum_transfer: float,
        lab_kinematics_k: float,
        epsilon: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the y right away:
            lepton_energy_fraction_y = tf.sqrt(squared_Q_momentum_transfer) / (epsilon * lab_kinematics_k)

            # (tf.constant(1.0, dtype = tf.float32)1): If verbose output, then print the result:
            if verbose:
                tf.print(f"> Calculated y to be:\n{lepton_energy_fraction_y}")

            # (2): Return the calculation:
            return lepton_energy_fraction_y
        
        except Exception as ERROR:
            tf.print(f"> Error in computing lepton_energy_fraction_y:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_kinematics_skewness_parameter(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        verbose: bool = False) -> float:
        try:

            # (1): The Numerator:
            numerator = (tf.constant(1.0, dtype = tf.float32) + (squared_hadronic_momentum_transfer_t / (tf.constant(2.0, dtype = tf.float32) * squared_Q_momentum_transfer)))

            # (2): The Denominator:
            denominator = (tf.constant(2.0, dtype = tf.float32) - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))

            # (3): Calculate the Skewness Parameter:
            skewness_parameter = x_Bjorken * numerator / denominator

            # (3.1): If verbose, print the output:
            if verbose:
                tf.print(f"> Calculated skewness xi to be:\n{skewness_parameter}")

            # (4): Return Xi:
            return skewness_parameter
        
        except Exception as ERROR:
            tf.print(f"> Error in computing skewness xi:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
    
    @tf.function
    def calculate_kinematics_t_min(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        epsilon: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate 1 - x_{B}:
            one_minus_xb = tf.constant(1.0, dtype = tf.float32) - x_Bjorken

            # (2): Calculate the numerator:
            numerator = (tf.constant(2.0, dtype = tf.float32) * one_minus_xb * (tf.constant(1.0, dtype = tf.float32) - tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2))) + epsilon**2

            # (3): Calculate the denominator:
            denominator = (tf.constant(4.0, dtype = tf.float32) * x_Bjorken * one_minus_xb) + epsilon**2

            # (4): Obtain the t minimum
            t_minimum = -tf.constant(1.0, dtype = tf.float32) * squared_Q_momentum_transfer * numerator / denominator

            # (tf.constant(4.0, dtype = tf.float32)1): If verbose, print the result:
            if verbose:
                tf.print(f"> Calculated t_minimum to be:\n{t_minimum}")

            # (5): Print the result:
            return t_minimum

        except Exception as ERROR:
            tf.print(f"> Error calculating t_minimum: \n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_kinematics_t_prime(
        self,
        squared_hadronic_momentum_transfer_t: float,
        squared_hadronic_momentum_transfer_t_minimum: float,
        verbose: bool = False) -> float:
        try:

            # (1): Obtain the t_prime immediately
            t_prime = squared_hadronic_momentum_transfer_t - squared_hadronic_momentum_transfer_t_minimum

            # (tf.constant(1.0, dtype = tf.float32)1): If verbose, print the result:
            if verbose:
                tf.print(f"> Calculated t prime to be:\n{t_prime}")

            # (2): Return t_prime
            return t_prime

        except Exception as ERROR:
            tf.print(f"> Error calculating t_prime:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_kinematics_k_tilde(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        lepton_energy_fraction_y: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        squared_hadronic_momentum_transfer_t_minimum: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate recurring quantity t_{min} - t
            tmin_minus_t = squared_hadronic_momentum_transfer_t_minimum - squared_hadronic_momentum_transfer_t

            # (2): Calculate the duplicate quantity 1 - x_{B}
            one_minus_xb = tf.constant(1.0, dtype = tf.float32) - x_Bjorken

            # (3): Calculate the crazy root quantity:
            second_root_quantity = (one_minus_xb * tf.sqrt((tf.constant(1.0, dtype = tf.float32) + epsilon**2))) + ((tmin_minus_t * (epsilon**2 + (tf.constant(4.0, dtype = tf.float32) * one_minus_xb * x_Bjorken))) / (tf.constant(4.0, dtype = tf.float32) * squared_Q_momentum_transfer))

            # (4): Calculate the first annoying root quantity:
            first_root_quantity = tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - lepton_energy_fraction_y**2 * epsilon**2 / tf.constant(4.0, dtype = tf.float32))
            
            # (5): Calculate K_tilde
            k_tilde = tf.sqrt(tmin_minus_t) * tf.sqrt(second_root_quantity)

            # (6): Print the result of the calculation:
            if verbose:
                tf.print(f"> Calculated k_tilde to be:\n{k_tilde}")

            # (7) Return:
            return k_tilde

        except Exception as ERROR:
            tf.print(f"> Error in calculating K_tilde:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_kinematics_k(
        self,
        squared_Q_momentum_transfer: float,
        lepton_energy_fraction_y: float,
        epsilon: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the amazing prefactor:
            prefactor = tf.sqrt(((tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y + (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))) / squared_Q_momentum_transfer))

            # (2): Calculate the remaining part of the term:
            kinematic_k = prefactor * k_tilde

            # (tf.constant(2.0, dtype = tf.float32)1); If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated kinematic K to be:\n{kinematic_k}")

            # (3): Return the value:
            return kinematic_k

        except Exception as ERROR:
            tf.print(f"> Error in calculating derived kinematic K:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_k_dot_delta(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        azimuthal_phi: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        kinematic_k: float,
        verbose: bool = False):
        try:
        
            # (1): The prefactor: \frac{Q^{2}}{2 y (1 + \varepsilon^{2})}
            prefactor = squared_Q_momentum_transfer / (tf.constant(2.0, dtype = tf.float32) * lepton_energy_fraction_y * (tf.constant(1.0, dtype = tf.float32) + epsilon**2))

            # (2): Second term in parentheses: Phi-Dependent Term: 2 K tf.cos(\phi)
            phi_dependence = tf.constant(2.0, dtype = tf.float32) * kinematic_k * tf.cos(tf.constant(np.pi, dtype = tf.float32) - self.convert_degrees_to_radians(azimuthal_phi))
            
            # (3): Prefactor of third term in parentheses: \frac{t}{Q^{2}}
            ratio_delta_to_q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (4): Second term in the third term's parentheses: x_{B} (2 - y)
            bjorken_scaling = x_Bjorken * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)

            # (5): Third term in the third term's parentheses: \frac{y \varepsilon^{2}}{2}
            ratio_y_epsilon = lepton_energy_fraction_y * epsilon**2 / tf.constant(2.0, dtype = tf.float32)

            # (6): Adding up all the "correction" pieces to the prefactor, written as (1 + correction)
            correction = phi_dependence - (ratio_delta_to_q_squared * (tf.constant(1.0, dtype = tf.float32) - bjorken_scaling + ratio_y_epsilon)) + (ratio_y_epsilon)

            # (7): Writing it explicitly as "1 + correction"
            in_parentheses = tf.constant(1.0, dtype = tf.float32) + correction

            # (8): The actual equation:
            k_dot_delta_result = -tf.constant(1.0, dtype = tf.float32) * prefactor * in_parentheses

            # (9): If verbose, print the output:
            if verbose:
                tf.print(f"> Calculated k dot delta: {k_dot_delta_result}")

            # (9): Return the number:
            return k_dot_delta_result
        
        except Exception as E:
            tf.print(f"> Error in calculating k.Delta:\n> {E}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_lepton_propagator_p1(
        self,
        squared_Q_momentum_transfer: float,
        k_dot_delta: float,
        verbose:bool = False) -> float:
        try:
            p1_propagator = tf.constant(1.0, dtype = tf.float32) + (tf.constant(2.0, dtype = tf.float32) * (k_dot_delta / squared_Q_momentum_transfer))
            
            if verbose:
                tf.print(f"> Computed the P1 propagator to be:\n{p1_propagator}")

            return p1_propagator
        
        except Exception as E:
            tf.print(f"> Error in computing p1 propagator:\n> {E}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_lepton_propagator_p2(
        self,
        squared_Q_momentum_transfer: float,
        squared_hadronic_momentum_transfer_t: float,
        k_dot_delta: float,
        verbose: bool = False) -> float:
        try:
            p2_propagator = (-tf.constant(2.0, dtype = tf.float32) * (k_dot_delta / squared_Q_momentum_transfer)) + (squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer)
            
            if verbose:
                tf.print(f"> Computed the P2 propagator to be:\n{p2_propagator}")

            return p2_propagator
        
        except Exception as E:
            tf.print(f"> Error in computing p2 propagator:\n> {E}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_form_factor_electric(
        self,
        squared_hadronic_momentum_transfer_t: float,
        verbose: bool = False) -> float:
        try:
            
            # (1): Calculate the mysterious denominator:
            denominator = tf.constant(1.0, dtype = tf.float32) - (squared_hadronic_momentum_transfer_t / _ELECTRIC_FORM_FACTOR_CONSTANT)

            # (2): Calculate the F_{E}:
            form_factor_electric = tf.constant(1.0, dtype = tf.float32) / (denominator**2)

            if verbose:
                tf.print(f"> Calculated electric form factor as: {form_factor_electric}")

            return form_factor_electric

        except Exception as ERROR:
            tf.print(f"> Error in calculating electric form factor:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_form_factor_magnetic(
        self,
        electric_form_factor: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the F_{M}:
            form_factor_magnetic = _PROTON_MAGNETIC_MOMENT * electric_form_factor

            if verbose:
                tf.print(f"> Calculated magnetic form factor as: {form_factor_magnetic}")

            return form_factor_magnetic

        except Exception as ERROR:
            tf.print(f"> Error in calculating magnetic form factor:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_form_factor_pauli_f2(
        self,
        squared_hadronic_momentum_transfer_t: float,
        electric_form_factor: float,
        magnetic_form_factor: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate tau:
            tau = -tf.constant(1.0, dtype = tf.float32) * squared_hadronic_momentum_transfer_t / (tf.constant(4.0, dtype = tf.float32) * _MASS_OF_PROTON_IN_GEV**2)

            # (2): Calculate the numerator:
            numerator = magnetic_form_factor - electric_form_factor

            # (3): Calculate the denominator:
            denominator = tf.constant(1.0, dtype = tf.float32) + tau
        
            # (4): Calculate the Pauli form factor:
            pauli_form_factor = numerator / denominator

            if verbose:
                tf.print(f"> Calculated Fermi form factor as: {pauli_form_factor}")

            return pauli_form_factor

        except Exception as ERROR:
            tf.print(f"> Error in calculating Fermi form factor:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_form_factor_dirac_f1(
        self,
        magnetic_form_factor: float,
        pauli_f2_form_factor: float,
        verbose: bool = False) -> float:
        try:
        
            # (1): Calculate the Dirac form factor:
            dirac_form_factor = magnetic_form_factor - pauli_f2_form_factor

            if verbose:
                tf.print(f"> Calculated Dirac form factor as: {dirac_form_factor}")

            return dirac_form_factor

        except Exception as ERROR:
            tf.print(f"> Error in calculating Dirac form factor:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def compute_cff_effective(
        self,
        skewness_parameter: float,
        compton_form_factor: complex,
        use_ww: bool = False,
        verbose: bool = False) -> complex:
        try:

            # (1): Do the calculation in one line:
            if use_ww:
                cff_effective = tf.constant(2.0, dtype = tf.float32) * compton_form_factor / (tf.constant(1.0, dtype = tf.float32) + skewness_parameter)
            else:
                cff_effective = -tf.constant(2.0, dtype = tf.float32) * skewness_parameter * compton_form_factor / (tf.constant(1.0, dtype = tf.float32) + skewness_parameter)

            # (tf.constant(1.0, dtype = tf.float32)1): If verbose, log the output:
            if verbose:
                tf.print(f"> Computed the CFF effective to be:\n{cff_effective}")

            # (2): Return the output:
            return cff_effective

        except Exception as ERROR:
            tf.print(f"> Error in calculating F_effective:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_bkm10_cross_section_prefactor(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the numerator of the prefactor
            numerator = _ELECTROMAGNETIC_FINE_STRUCTURE_CONSTANT**3 * lepton_energy_fraction_y**2 * x_Bjorken

            # (2): Calculate the denominator of the prefactor:
            denominator = tf.constant(8.0, dtype = tf.float32) * tf.constant(np.pi, dtype = tf.float32) * squared_Q_momentum_transfer**2 * tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (3): Construct the prefactor:
            prefactor = numerator / denominator

            if verbose:
                tf.print(f"> Calculated BKM10 cross-section prefactor to be:\n{prefactor}")

            # (4): Return the prefactor:
            return prefactor

        except Exception as ERROR:
            tf.print(f"> Error calculating BKM10 cross section prefactor:\n> {ERROR}")
            return 0

    @tf.function
    def calculate_curly_C_unpolarized_interference(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        Dirac_form_factor_F1: float,
        Pauli_form_factor_F2: float,
        real_H: tf.Tensor,
        imag_H: tf.Tensor,
        real_Ht: tf.Tensor,
        imag_Ht: tf.Tensor,
        real_E: tf.Tensor,
        imag_E: tf.Tensor,
        verbose: bool = False) -> float:

        # (1): Calculate the first two terms: weighted CFFs:
        weighted_cffs_real = (Dirac_form_factor_F1 * real_H) - (squared_hadronic_momentum_transfer_t * Pauli_form_factor_F2 * real_E / (tf.constant(4.0, dtype = tf.float32) * _MASS_OF_PROTON_IN_GEV**2))
        weighted_cffs_imag = (Dirac_form_factor_F1 * imag_H) - (squared_hadronic_momentum_transfer_t * Pauli_form_factor_F2 * imag_E / (tf.constant(4.0, dtype = tf.float32) * _MASS_OF_PROTON_IN_GEV**2))

        # (2): Calculate the next term:
        second_term_real = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) * real_Ht / (tf.constant(2.0, dtype = tf.float32) - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
        second_term_imag = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) * imag_Ht / (tf.constant(2.0, dtype = tf.float32) - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))

        # (3): Add them together:
        curly_C_unpolarized_interference_real = weighted_cffs_real + second_term_real
        curly_C_unpolarized_interference_imag = weighted_cffs_imag + second_term_imag

        # (tf.constant(4.0, dtype = tf.float32)1): If verbose, print the calculation:
        if verbose:
            tf.print(f"> Calculated Curly C interference unpolarized target to be:\n{curly_C_unpolarized_interference_real. curly_C_unpolarized_interference_imag}")

        # (5): Return the output:
        return curly_C_unpolarized_interference_real, curly_C_unpolarized_interference_imag

    @tf.function
    def calculate_curly_C_unpolarized_interference_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        Dirac_form_factor_F1: float,
        Pauli_form_factor_F2: float,
        real_H: tf.Tensor,
        imag_H: tf.Tensor,
        real_E: tf.Tensor,
        imag_E: tf.Tensor,
        verbose: bool = False) -> float:

        # (1): Calculate the first two terms: weighted CFFs:
        cff_term_real = real_H + real_E
        cff_term_imag = imag_H + imag_E

        # (2): Calculate the next term:
        second_term = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) / (tf.constant(2.0, dtype = tf.float32) - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))

        # (3): Add them together:
        curly_C_unpolarized_interference_V_real = cff_term_real * second_term
        curly_C_unpolarized_interference_V_imag = cff_term_imag * second_term

        # (tf.constant(4.0, dtype = tf.float32)1): If verbose, print the calculation:
        if verbose:
            tf.print(f"> Calculated Curly C interference V unpolarized target to be:\n{curly_C_unpolarized_interference_V_real}")

        # (5): Return the output:
        return curly_C_unpolarized_interference_V_real, curly_C_unpolarized_interference_V_imag
        
    @tf.function
    def calculate_curly_C_unpolarized_interference_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        Dirac_form_factor_F1: float,
        Pauli_form_factor_F2: float,
        real_Ht: tf.Tensor,
        imag_Ht: tf.Tensor,
        verbose: bool = False) -> float:

        # (1): Calculate the next term:
        xb_modulation = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) / (tf.constant(2.0, dtype = tf.float32) - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))

        # (2): Add them together:
        curly_C_unpolarized_interference_A_real = real_Ht * xb_modulation
        curly_C_unpolarized_interference_A_imag = imag_Ht * xb_modulation

        # (3.1): If verbose, print the calculation:
        if verbose:
            tf.print(f"> Calculated Curly C interference A unpolarized target to be:\n{curly_C_unpolarized_interference_A_real}")

        # (4): Return the output:
        return curly_C_unpolarized_interference_A_real, curly_C_unpolarized_interference_A_imag
    
    @tf.function
    def calculate_c_0_plus_plus_unpolarized(
        self,
        squared_Q_momentum_transfer,
        x_Bjorken,
        squared_hadronic_momentum_transfer_t,
        epsilon,
        lepton_energy_fraction_y,
        k_tilde):
        """
        """

        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate 1 + sqrt(1 + epsilon^{2}):
            one_plus_root_epsilon_stuff = tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared

            # (4): Calculate 2 - x_{B}:
            two_minus_xb = tf.constant(2.0, dtype = tf.float32) - x_Bjorken

            # (5): Caluclate 2 - y:
            two_minus_y = tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y

            # (6): Calculate the first term in the brackets:
            first_term_in_brackets = k_tilde**2 * two_minus_y**2 / (squared_Q_momentum_transfer * root_one_plus_epsilon_squared)

            # (7): Calculate the first part of the second term in brackets:
            second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)))
            
            # (8): Calculate the numerator of the second part of the second term in brackets:
            second_term_in_brackets_second_part_numerator = tf.constant(2.0, dtype = tf.float32) * x_Bjorken * t_over_Q_squared * (two_minus_xb + tf.constant(0.5, dtype = tf.float32) * (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32)) + tf.constant(0.5, dtype = tf.float32) * epsilon**2 / x_Bjorken) + epsilon**2
            
            # (9): Calculate the second part of the second term in brackets:
            second_term_in_brackets_second_part =  tf.constant(1.0, dtype = tf.float32) + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
            
            # (10): Calculate the prefactor:
            prefactor = -tf.constant(4.0, dtype = tf.float32) * two_minus_y * one_plus_root_epsilon_stuff / tf.pow(root_one_plus_epsilon_squared, 4)

            # (11): Calculate the coefficient
            c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)

            # (12): Return the coefficient:
            return c_0_plus_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_0_plus_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_0_plus_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        k_tilde: float,
        verbose: bool = False) -> float:

        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the recurrent quantity 1 + sqrt(1 + epsilon^2):
            one_plus_root_epsilon_stuff = tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared

            # (4): Compute the first term in the brackets:
            first_term_in_brackets = (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)**2 * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)

            # (5): First multiplicative term in the second term in the brackets:
            second_term_first_multiplicative_term = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (6): Second multiplicative term in the second term in the brackets:
            second_term_second_multiplicative_term = one_plus_root_epsilon_stuff / tf.constant(2.0, dtype = tf.float32)

            # (7): Third multiplicative term in the second term in the brackets:
            second_term_third_multiplicative_term = tf.constant(1.0, dtype = tf.float32) + t_over_Q_squared

            # (8): Fourth multiplicative term numerator in the second term in the brackets:
            second_term_fourth_multiplicative_term = tf.constant(1.0, dtype = tf.float32) + (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32) + (tf.constant(2.0, dtype = tf.float32) * x_Bjorken)) * t_over_Q_squared / one_plus_root_epsilon_stuff

            # (9): Fourth multiplicative term in its entirety:
            second_term_in_brackets = second_term_first_multiplicative_term * second_term_second_multiplicative_term * second_term_third_multiplicative_term * second_term_fourth_multiplicative_term

            # (10): The prefactor in front of the brackets:
            coefficient_prefactor = tf.constant(8.0, dtype = tf.float32) * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4

            # (11): The entire thing:
            c_0_plus_plus_V_unp = coefficient_prefactor * (first_term_in_brackets + second_term_in_brackets)

            # (11.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_0_plus_plus_V_unp to be:\n{c_0_plus_plus_V_unp}")

            # (12): Return the coefficient:
            return c_0_plus_plus_V_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_0_plus_plus_V_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)    

    @tf.function
    def calculate_c_0_plus_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the recurrent quantity 1 + sqrt(1 + epsilon^2):
            one_plus_root_epsilon_stuff = tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared

            # (4): Calculate 2 - y:
            two_minus_y = tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y

            # (5): Calculate Ktilde^{2}/squaredQ:
            ktilde_over_Q_squared = k_tilde**2 / squared_Q_momentum_transfer

            # (6): Calculate the first term in the curly brackets:
            curly_bracket_first_term = two_minus_y**2 * ktilde_over_Q_squared * (one_plus_root_epsilon_stuff - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) / (tf.constant(2.0, dtype = tf.float32) * root_one_plus_epsilon_squared)

            # (7): Calculate inner parentheses term:
            deepest_parentheses_term = (x_Bjorken * (tf.constant(2.0, dtype = tf.float32) + one_plus_root_epsilon_stuff - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) / one_plus_root_epsilon_stuff + (one_plus_root_epsilon_stuff - tf.constant(2.0, dtype = tf.float32))) * t_over_Q_squared

            # (8): Calculate the square-bracket term:
            square_bracket_term = one_plus_root_epsilon_stuff * (one_plus_root_epsilon_stuff - x_Bjorken + deepest_parentheses_term) / tf.constant(2.0, dtype = tf.float32) - (tf.constant(2.0, dtype = tf.float32) * ktilde_over_Q_squared)

            # (9): Calculate the second bracket term:
            curly_bracket_second_term = (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) * square_bracket_term

            # (10): Calculate the prefactor: 
            coefficient_prefactor = tf.constant(8.0, dtype = tf.float32) * two_minus_y * t_over_Q_squared / root_one_plus_epsilon_squared**4

            # (11): The entire thing:
            c_0_plus_plus_A_unp = coefficient_prefactor * (curly_bracket_first_term + curly_bracket_second_term)

            # (11.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_0_plus_plus_A_unp to be:\n{c_0_plus_plus_A_unp}")

            # (12): Return the coefficient:
            return c_0_plus_plus_A_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_0_plus_plus_A_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_1_plus_plus_unpolarized(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate 1 + sqrt(1 + epsilon^{2}):
            one_plus_root_epsilon_stuff = tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared

            # (4): Calculate first term in first brackets
            first_bracket_first_term = (tf.constant(1.0, dtype = tf.float32) + (tf.constant(1.0, dtype = tf.float32) - x_Bjorken) * (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32)) / (tf.constant(2.0, dtype = tf.float32) * x_Bjorken) + epsilon**2 / (tf.constant(4.0, dtype = tf.float32) * x_Bjorken)) * x_Bjorken * t_over_Q_squared

            # (5): Calculate the first bracket term:
            first_bracket_term = first_bracket_first_term - 3. * epsilon**2 / tf.constant(4.0, dtype = tf.float32)

            # (6): Calculate the second bracket term:
            second_bracket_term = tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - 3. * x_Bjorken) * t_over_Q_squared + (tf.constant(1.0, dtype = tf.float32) - root_one_plus_epsilon_squared + 3. * epsilon**2) * x_Bjorken * t_over_Q_squared / (one_plus_root_epsilon_stuff - epsilon**2)

            # (7): Calculate the crazy coefficient with all the y's:
            fancy_y_coefficient = tf.constant(2.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * lepton_energy_fraction_y + lepton_energy_fraction_y**2 + epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(2.0, dtype = tf.float32)

            # (8): Calculate the entire second term:
            second_term = -tf.constant(4.0, dtype = tf.float32) * shorthand_k * fancy_y_coefficient * (one_plus_root_epsilon_stuff - epsilon**2) * second_bracket_term / root_one_plus_epsilon_squared**5

            # (9): Calculate the first term:
            first_term = -tf.constant(16.0, dtype = tf.float32) * shorthand_k * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) * first_bracket_term / root_one_plus_epsilon_squared**5

            # (10): Calculate the coefficient
            c_1_plus_plus_unp = first_term + second_term
            
            # (11.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_1_plus_plus_unp to be:\n{c_1_plus_plus_unp}")

            # (12): Return the coefficient:
            return c_1_plus_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_1_plus_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_1_plus_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
        root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

        # (2): Calculate the recurrent quantity t/Q^{2}:
        t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

        # (3): Calculate the first bracket term:
        first_bracket_term = (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)**2 * (tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared)

        # (4): Compute the first part of the second term in brackets:
        second_bracket_term_first_part = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)

        # (5): Compute the second part of the second term in brackets:
        second_bracket_term_second_part = tf.constant(0.5, dtype = tf.float32) * (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_prime / squared_Q_momentum_transfer

        # (6): The prefactor in front of the brackets:
        coefficient_prefactor = tf.constant(16.0, dtype = tf.float32) * shorthand_k * x_Bjorken * t_over_Q_squared / tf.pow(root_one_plus_epsilon_squared, 5)

        # (7): The entire thing:
        c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)

        # (7.1): If verbose, log the output:
        if verbose:
            tf.print(f"> Calculated c_1_plus_plus_V_unp to be:\n{c_1_plus_plus_V_unp}")

        # (12): Return the coefficient:
        return c_1_plus_plus_V_unp

    @tf.function
    def calculate_c_1_plus_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate t'/Q^{2}
            t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (4): Calculate 1 - x_{B}:
            one_minus_xb = tf.constant(1.0, dtype = tf.float32) - x_Bjorken

            # (5): Calculate 1 - 2 x_{B}:
            one_minus_2xb = tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken

            # (6): Calculate a fancy, annoying quantity:
            fancy_y_stuff = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)

            # (7): Calculate the second contribution to the first term in brackets:
            first_bracket_term_second_part = tf.constant(1.0, dtype = tf.float32) - one_minus_2xb * t_over_Q_squared + (tf.constant(4.0, dtype = tf.float32) * x_Bjorken * one_minus_xb + epsilon**2) * t_prime_over_Q_squared / (tf.constant(4.0, dtype = tf.float32) * root_one_plus_epsilon_squared)

            # (8): Calculate the second bracket term:
            second_bracket_term = tf.constant(1.0, dtype = tf.float32) - tf.constant(0.5, dtype = tf.float32) * x_Bjorken + tf.constant(0.25, dtype = tf.float32) * (one_minus_2xb + root_one_plus_epsilon_squared) * (tf.constant(1.0, dtype = tf.float32) - t_over_Q_squared) + (tf.constant(4.0, dtype = tf.float32) * x_Bjorken * one_minus_xb + epsilon**2) * t_prime_over_Q_squared / (tf.constant(2.0, dtype = tf.float32) * root_one_plus_epsilon_squared)

            # (9): Calculate the prefactor:
            prefactor = -tf.constant(16.0, dtype = tf.float32) * shorthand_k * t_over_Q_squared / root_one_plus_epsilon_squared**4
            
            # (10): The entire thing:
            c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)**2 * second_bracket_term)

            # (10.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_1_plus_plus_A_unp to be:\n{c_1_plus_plus_A_unp}")

            # (11): Return the coefficient:
            return c_1_plus_plus_A_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_1_plus_plus_A_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_2_plus_plus_unpolarized(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the first bracket quantity:
            first_bracket_term = tf.constant(2.0, dtype = tf.float32) * epsilon**2 * k_tilde**2 / (root_one_plus_epsilon_squared * (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared) * squared_Q_momentum_transfer)
        
            # (4): Calculate the second bracket quantity:
            second_bracket_term = x_Bjorken * t_prime * t_over_Q_squared * (tf.constant(1.0, dtype = tf.float32) - x_Bjorken - tf.constant(0.5, dtype = tf.float32) * (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32)) + tf.constant(0.5, dtype = tf.float32) * epsilon**2 / x_Bjorken) / squared_Q_momentum_transfer

            # (5): Calculate the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) / root_one_plus_epsilon_squared**4
            
            # (6): Calculate the coefficient
            c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
            
            # (6.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_2_plus_plus_unp to be:\n{c_2_plus_plus_unp}")

            # (7): Return the coefficient:
            return c_2_plus_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_2_plus_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_2_plus_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate t'/Q^{2}
            t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (4): Calculate the major term:
            major_term = (tf.constant(4.0, dtype = tf.float32) * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)) + tf.constant(0.5, dtype = tf.float32) * (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * (tf.constant(1.0, dtype = tf.float32) + t_over_Q_squared) * t_prime_over_Q_squared

            # (5): Calculate the prefactor: 
            prefactor = tf.constant(8.0, dtype = tf.float32) * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
            
            # (6): The entire thing:
            c_2_plus_plus_V_unp = prefactor * major_term

            # (6.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_2_plus_plus_V_unp to be:\n{c_2_plus_plus_V_unp}")

            # (7): Return the coefficient:
            return c_2_plus_plus_V_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_2_plus_plus_V_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_2_plus_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        k_tilde: float,
        verbose: bool = False) -> float:

        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate t'/Q^{2}
            t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (4): Calculate the first bracket term:
            first_bracket_term = tf.constant(4.0, dtype = tf.float32) * (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)

            # (5): Calculate the second bracket term:
            second_bracket_term = (3.  - root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken + epsilon**2 / x_Bjorken ) * x_Bjorken * t_prime_over_Q_squared

            # (6): Calculate the prefactor: 
            prefactor = tf.constant(4.0, dtype = tf.float32) * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) * t_over_Q_squared / root_one_plus_epsilon_squared**4
            
            # (7): The entire thing:
            c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)

            # (7.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_2_plus_plus_A_unp to be:\n{c_2_plus_plus_A_unp}")

            # (8): Return the coefficient:
            return c_2_plus_plus_A_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_2_plus_plus_A_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_3_plus_plus_unpolarized(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the major term:
            major_term = (tf.constant(1.0, dtype = tf.float32) - x_Bjorken) * t_over_Q_squared + tf.constant(0.5, dtype = tf.float32) * (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32)) * (tf.constant(1.0, dtype = tf.float32) + t_over_Q_squared)
        
            # (4): Calculate the "intermediate" term:
            intermediate_term = (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32)) / root_one_plus_epsilon_squared**5

            # (5): Calculate the prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * shorthand_k * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))
            
            # (6): Calculate the coefficient
            c_3_plus_plus_unp = prefactor * intermediate_term * major_term
            
            # (6.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_3_plus_plus_unp to be:\n{c_3_plus_plus_unp}")

            # (7): Return the coefficient:
            return c_3_plus_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_3_plus_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_3_plus_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the major term:
            major_term = root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32) + (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared

            # (4): Calculate he prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * shorthand_k * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**5
            
            # (5): The entire thing:
            c_3_plus_plus_V_unp = prefactor * major_term

            # (5.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_3_plus_plus_V_unp to be:\n{c_3_plus_plus_V_unp}")

            # (7): Return the coefficient:
            return c_3_plus_plus_V_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_3_plus_plus_V_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_3_plus_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        """
        """

        try:

            # (1): Calculate the main term:
            main_term = squared_hadronic_momentum_transfer_t * t_prime * (x_Bjorken * (tf.constant(1.0, dtype = tf.float32) - x_Bjorken) + epsilon**2 / tf.constant(4.0, dtype = tf.float32)) / squared_Q_momentum_transfer**2

            # (2): Calculate the prefactor:
            prefactor = tf.constant(16.0, dtype = tf.float32) * shorthand_k * (tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)) / (tf.constant(1.0, dtype = tf.float32) + epsilon**2)**2.5
            
            # (3): The entire thing:
            c_3_plus_plus_A_unp = prefactor * main_term

            # (3.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_3_plus_plus_A_unp to be:\n{c_3_plus_plus_A_unp}")

            # (4): Return the coefficient:
            return c_3_plus_plus_A_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_2_plus_plus_A_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)

    @tf.function
    def calculate_c_0_zero_plus_unpolarized(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the bracket quantity:
            bracket_quantity = epsilon**2 + squared_hadronic_momentum_transfer_t * (tf.constant(2.0, dtype = tf.float32) - tf.constant(6.0, dtype = tf.float32) * x_Bjorken - epsilon**2) / (tf.constant(3.0, dtype = tf.float32) * squared_Q_momentum_transfer)
            
            # (2): Calculate part of the prefactor:
            prefactor = tf.constant(12.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32)) * shorthand_k * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))) / tf.pow(tf.constant(1.0, dtype = tf.float32) + epsilon**2, tf.constant(2.5, dtype = tf.float32))
            
            # (3): Calculate the coefficient:
            c_0_zero_plus_unp = prefactor * bracket_quantity
            
            # (3.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_0_zero_plus_unp to be:\n{c_0_zero_plus_unp}")

            # (4): Return the coefficient:
            return c_0_zero_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_0_zero_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_0_zero_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (2): Calculate the main part of the thing:
            main_part = x_Bjorken * t_over_Q_squared * (tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared)

            # (3): Calculate the prefactor:
            prefactor = tf.constant(24.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32)) * shorthand_k * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (lepton_energy_fraction_y**2 * epsilon**2 / tf.constant(4.0, dtype = tf.float32))) / (tf.constant(1.0, dtype = tf.float32) + epsilon**2)**2.5

            # (4): Stitch together the coefficient:
            c_0_zero_plus_V_unp = prefactor * main_part

            # (tf.constant(4.0, dtype = tf.float32)1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_0_zero_plus_V_unp to be:\n{c_0_zero_plus_V_unp}")

            # (5): Return the coefficient:
            return c_0_zero_plus_V_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_0_zero_plus_V_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_0_zero_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (2): Calculate the recurrent quantity 8 - 6x_{B} + 5 epsilon^{2}:
            fancy_xb_epsilon_term = tf.constant(8.0, dtype = tf.float32) - tf.constant(6.0, dtype = tf.float32) * x_Bjorken + tf.constant(5.0, dtype = tf.float32) * epsilon**2

            # (3): Compute the bracketed term:
            brackets_term = tf.constant(1.0, dtype = tf.float32) - t_over_Q_squared * (tf.constant(2.0, dtype = tf.float32) - tf.constant(12.0, dtype = tf.float32) * x_Bjorken * (tf.constant(1.0, dtype = tf.float32) - x_Bjorken) - epsilon**2) / fancy_xb_epsilon_term

            # (4): Calculate the prefactor:
            prefactor = tf.constant(4.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32)) * shorthand_k * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (lepton_energy_fraction_y**2 * epsilon**2 / tf.constant(4.0, dtype = tf.float32))) / tf.pow(tf.constant(1.0, dtype = tf.float32) + epsilon**2, tf.constant(2.5, dtype = tf.float32))

            # (5): Stitch together the coefficient:
            c_0_zero_plus_A_unp = prefactor * t_over_Q_squared * fancy_xb_epsilon_term * brackets_term

            # (5.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_0_zero_plus_A_unp to be:\n{c_0_zero_plus_A_unp}")

            # (6): Return the coefficient:
            return c_0_zero_plus_A_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_0_zero_plus_A_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_1_zero_plus_unpolarized(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate t'/Q^{2}
            t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (4): Calculate 1 - x_{B}:
            one_minus_xb = tf.constant(1.0, dtype = tf.float32) - x_Bjorken

            # (5): Calculate the annoying y quantity:
            y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (6): Calculate the first term:
            first_bracket_term = (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * x_Bjorken + (epsilon**2 / tf.constant(4.0, dtype = tf.float32))) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
            
            # (7): Calculate the second term:
            second_bracket_term = y_quantity * (tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared) * (epsilon**2 - tf.constant(2.0, dtype = tf.float32) * (tf.constant(1.0, dtype = tf.float32) + (epsilon**2 / (tf.constant(2.0, dtype = tf.float32) * x_Bjorken))) * x_Bjorken * t_over_Q_squared) / root_one_plus_epsilon_squared
            
            # (8): Calculate part of the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) / root_one_plus_epsilon_squared**4
            
            # (9): Calculate the coefficient:
            c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
            
            # (9.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_1_zero_plus_unp to be:\n{c_1_zero_plus_unp}")

            # (9): Return the coefficient:
            return c_1_zero_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_1_zero_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_1_zero_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (2): Calculate the huge y quantity:
            y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (3): Calculate the major part:
            major_part = (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)**2 * k_tilde**2 / squared_Q_momentum_transfer + (tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared)**2 * y_quantity

            # (4): Calculate the prefactor:
            prefactor = tf.constant(16.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) * x_Bjorken * t_over_Q_squared / (tf.constant(1.0, dtype = tf.float32) + epsilon**2)**2.5

            # (5): Stitch together the coefficient:
            c_1_zero_plus_V_unp = prefactor * major_part

            # (5.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_1_zero_plus_V_unp to be:\n{c_1_zero_plus_V_unp}")

            # (6): Return the coefficient:
            return c_1_zero_plus_V_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_1_zero_plus_V_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_1_zero_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate 1 - 2x_{B}:
            one_minus_2xb = tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken

            # (4): Calculate the annoying y quantity:
            y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (5): Calculate the first part of the second term:
            second_term_first_part = (tf.constant(1.0, dtype = tf.float32) - one_minus_2xb * t_over_Q_squared) * y_quantity

            # (6); Calculate the second part of the second term:
            second_term_second_part = tf.constant(4.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken + 3. * epsilon**2 + t_over_Q_squared * (tf.constant(4.0, dtype = tf.float32) * x_Bjorken * (tf.constant(1.0, dtype = tf.float32) - x_Bjorken) + epsilon**2)
            
            # (7): Calculate the first term:
            first_term = k_tilde**2 * one_minus_2xb * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y)**2 / squared_Q_momentum_transfer
            
            # (8): Calculate part of the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
            
            # (9): Calculate the coefficient:
            c_1_zero_plus_unp_A = prefactor * (first_term + second_term_first_part * second_term_second_part)
            
            # (9.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_1_zero_plus_unp_A to be:\n{c_1_zero_plus_unp_A}")

            # (10): Return the coefficient:
            return c_1_zero_plus_unp_A

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_1_zero_plus_unp_A for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_2_zero_plus_unpolarized(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity epsilon^2/2:
            epsilon_squared_over_2 = epsilon**2 / tf.constant(2.0, dtype = tf.float32)

            # (3): Calculate the annoying y quantity:
            y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (4): Calculate the bracket term:
            bracket_term = tf.constant(1.0, dtype = tf.float32) + ((tf.constant(1.0, dtype = tf.float32) + epsilon_squared_over_2 / x_Bjorken) / (tf.constant(1.0, dtype = tf.float32) + epsilon_squared_over_2)) * x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (5): Calculate the prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) * shorthand_k * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) / root_one_plus_epsilon_squared**5
            
            # (6): Calculate the coefficient:
            c_2_zero_plus_unp = prefactor * (tf.constant(1.0, dtype = tf.float32) + epsilon_squared_over_2) * bracket_term
            
            # (6.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_2_zero_plus_unp to be:\n{c_2_zero_plus_unp}")

            # (7): Return the coefficient:
            return c_2_zero_plus_unp

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_2_zero_plus_unp for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_2_zero_plus_unpolarized_V(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the annoying y quantity:
            y_quantity = tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)))

            # (4): Calculate the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32)) * y_quantity * shorthand_k * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**5
            
            # (5): Calculate the coefficient:
            c_2_zero_plus_unp_V = prefactor * (tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared)
            
            # (5.1): If verbose, log the output:
            if verbose:
                tf.print(f"> Calculated c_2_zero_plus_unp_V to be:\n{c_2_zero_plus_unp_V}")

            # (6): Return the coefficient:
            return c_2_zero_plus_unp_V

        except Exception as ERROR:
            tf.print(f"> Error in calculating c_2_zero_plus_unp_V for Interference Term:\n> {ERROR}")
            return tf.constant(0.0, dtype = tf.float32)
        
    @tf.function
    def calculate_c_2_zero_plus_unpolarized_A(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
        root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

        # (2): Calculate the recurrent quantity t/Q^{2}:
        t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

        # (3): Calculate t'/Q^{2}
        t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer

        # (4): Calculate 1 - x_{B}:
        one_minus_xb = tf.constant(1.0, dtype = tf.float32) - x_Bjorken

        # (5): Calculate the annoying y quantity:
        y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

        # (6): Calculate the bracket term:
        bracket_term = one_minus_xb + tf.constant(0.5, dtype = tf.float32) * t_prime_over_Q_squared * (tf.constant(4.0, dtype = tf.float32) * x_Bjorken * one_minus_xb + epsilon**2) / root_one_plus_epsilon_squared
        
        # (7): Calculate part of the prefactor:
        prefactor = tf.constant(8.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) * shorthand_k * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * t_over_Q_squared / root_one_plus_epsilon_squared**4
        
        # (8): Calculate the coefficient:
        c_2_zero_plus_unp_A = prefactor * bracket_term
        
        # (tf.constant(8.0, dtype = tf.float32)1): If verbose, log the output:
        if verbose:
            tf.print(f"> Calculated c_2_zero_plus_unp_A to be:\n{c_2_zero_plus_unp_A}")

        # (9): Return the coefficient:
        return c_2_zero_plus_unp_A

    @tf.function
    def calculate_s_1_plus_plus_unpolarized(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the quantity t'/Q^{2}: 
            tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (3): Calculate the bracket term:
            bracket_term = tf.constant(1.0, dtype = tf.float32) + ((tf.constant(1.0, dtype = tf.float32) - x_Bjorken + tf.constant(0.5, dtype = tf.float32) * (root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32))) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
            
            # (4): Calculate the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) / root_one_plus_epsilon_squared**2

            # (5): Calculate the coefficient
            s_1_plus_plus_unp = prefactor * bracket_term
            
            # (5.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_1_plus_plus_unp to be:\n{s_1_plus_plus_unp}")

            # (6): Return the coefficient:
            return s_1_plus_plus_unp

        except Exception as ERROR:
            print(f"> Error in calculating s_1_plus_plus_unp for Interference Term:\n> {ERROR}")
            return 0.

    @tf.function
    def calculate_s_1_plus_plus_unpolarized_V(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the bracket term:
            bracket_term = root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32) + (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared

            # (4): Calculate the prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4

            # (5): Calculate the coefficient
            s_1_plus_plus_unp_V = prefactor * bracket_term
            
            # (5.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_1_plus_plus_unp_V to be:\n{s_1_plus_plus_unp_V}")

            # (6): Return the coefficient:
            return s_1_plus_plus_unp_V

        except Exception as ERROR:
            print(f"> Error in calculating s_1_plus_plus_unp_V for Interference Term:\n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_s_1_plus_plus_unpolarized_A(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the quantity t'/Q^{2}:
            tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (4): Calculate the bracket term:
            one_minus_2xb = tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken

            # (5): Calculate the bracket term:
            bracket_term = tf.constant(1.0, dtype = tf.float32) - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (tf.constant(2.0, dtype = tf.float32) * root_one_plus_epsilon_squared)

            # (6): Calculate the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * t_over_Q_squared / root_one_plus_epsilon_squared**2

            # (7): Calculate the coefficient
            s_1_plus_plus_unp_A = prefactor * bracket_term
            
            # (7.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_1_plus_plus_unp_A to be:\n{s_1_plus_plus_unp_A}")

            # (8): Return the coefficient:
            return s_1_plus_plus_unp_A

        except Exception as ERROR:
            print(f"> Error in calculating s_1_plus_plus_unp_A for Interference Term:\n> {ERROR}")
            return 0.

    @tf.function
    def calculate_s_2_plus_plus_unpolarized(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the quantity t'/Q^{2}:
            tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (3): Calculate a fancy, annoying quantity:
            fancy_y_stuff = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)

            # (4): Calculate the first bracket term:
            first_bracket_term = (epsilon**2 - x_Bjorken * (root_one_plus_epsilon_squared - 1.)) / (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken)

            # (5): Calculate the second bracket term:
            second_bracket_term = (tf.constant(2.0, dtype = tf.float32) * x_Bjorken + epsilon**2) * tPrime_over_Q_squared / (tf.constant(2.0, dtype = tf.float32) * root_one_plus_epsilon_squared)

            # (6): Calculate the prefactor:
            prefactor = -tf.constant(4.0, dtype = tf.float32) * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * (tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3

            # (7): Calculate the coefficient
            s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)
            
            # (7.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_2_plus_plus_unp to be:\n{s_2_plus_plus_unp}")

            # (6): Return the coefficient:
            return s_2_plus_plus_unp

        except Exception as ERROR:
            print(f"> Error in calculating s_2_plus_plus_unp for Interference Term:\n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_s_2_plus_plus_unpolarized_V(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate a fancy, annoying quantity:
            fancy_y_stuff = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)

            # (4): Calculate the bracket term:
            one_minus_2xb = tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken

            # (5): Calculate the bracket term:
            bracket_term = root_one_plus_epsilon_squared - tf.constant(1.0, dtype = tf.float32) + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared

            # (6): Calculate the parentheses term:
            parentheses_term = tf.constant(1.0, dtype = tf.float32) - one_minus_2xb * t_over_Q_squared

            # (7): Calculate the prefactor:
            prefactor = -tf.constant(4.0, dtype = tf.float32) * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4

            # (8): Calculate the coefficient
            s_2_plus_plus_unp_V = prefactor * parentheses_term * bracket_term
            
            # (8.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_2_plus_plus_unp_V to be:\n{s_2_plus_plus_unp_V}")

            # (9): Return the coefficient:
            return s_2_plus_plus_unp_V

        except Exception as ERROR:
            print(f"> Error in calculating s_2_plus_plus_unp_V for Interference Term:\n> {ERROR}")
            return
        
    @tf.function
    def calculate_s_2_plus_plus_unpolarized_A(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        t_prime: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the quantity t'/Q^{2}:
            tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer

            # (4): Calculate a fancy, annoying quantity:
            fancy_y_stuff = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)

            # (5): Calculate the last term:
            last_term = tf.constant(1.0, dtype = tf.float32) + (tf.constant(4.0, dtype = tf.float32) * (tf.constant(1.0, dtype = tf.float32) - x_Bjorken) * x_Bjorken + epsilon**2) * t_over_Q_squared / (4. - tf.constant(2.0, dtype = tf.float32) * x_Bjorken + 3. * epsilon**2)

            # (6): Calculate the middle term:
            middle_term = tf.constant(1.0, dtype = tf.float32) + root_one_plus_epsilon_squared - tf.constant(2.0, dtype = tf.float32) * x_Bjorken

            # (7): Calculate the prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4

            # (8): Calculate the coefficient
            s_2_plus_plus_unp_A = prefactor * middle_term * last_term
            
            # (8.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_2_plus_plus_unp_A to be:\n{s_2_plus_plus_unp_A}")

            # (9): Return the coefficient:
            return s_2_plus_plus_unp_A

        except Exception as ERROR:
            print(f"> Error in calculating s_2_plus_plus_unp_A for Interference Term:\n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_s_1_zero_plus_unpolarized(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        k_tilde: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the  quantity (1 + epsilon^2)^{2}:
            root_one_plus_epsilon_squared = (tf.constant(1.0, dtype = tf.float32) + epsilon**2)**2

            # (2): Calculate the huge y quantity:
            y_quantity = tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)))

            # (3): Calculate the coefficient
            s_1_zero_plus_unp = tf.constant(8.0, dtype = tf.float32) * lepton_helicity * tf.sqrt(2.) * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * lepton_energy_fraction_y * y_quantity * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)
            
            # (3.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_1_zero_plus_unp to be:\n{s_1_zero_plus_unp}")

            # (4): Return the coefficient:
            return s_1_zero_plus_unp

        except Exception as ERROR:
            print(f"> Error in calculating s_1_zero_plus_unp for Interference Term:\n> {ERROR}")
            return 0.   
        
    @tf.function
    def calculate_s_1_zero_plus_unpolarized_V(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the quantity (1 + epsilon^2)^{2}:
            one_plus_epsilon_squared_squared = (tf.constant(1.0, dtype = tf.float32) + epsilon**2)**2

            # (2): Calculate the quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate a fancy, annoying quantity:
            fancy_y_stuff = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)

            # (4): Calculate the bracket term:
            bracket_term = tf.constant(4.0, dtype = tf.float32) * (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared * (tf.constant(1.0, dtype = tf.float32) + x_Bjorken * t_over_Q_squared) + epsilon**2 * (tf.constant(1.0, dtype = tf.float32) + t_over_Q_squared)**2

            # (5): Calculate the prefactor:
            prefactor = tf.constant(4.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * fancy_y_stuff) * lepton_helicity * lepton_energy_fraction_y * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / one_plus_epsilon_squared_squared

            # (6): Calculate the coefficient
            s_1_zero_plus_unp_V = prefactor * bracket_term
            
            # (6.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_1_zero_plus_unp_V to be:\n{s_1_zero_plus_unp_V}")

            # (7): Return the coefficient:
            return s_1_zero_plus_unp_V

        except Exception as ERROR:
            print(f"> Error in calculating s_1_zero_plus_unp_V for Interference Term:\n> {ERROR}")
            return 0.
    
    @tf.function
    def calculate_s_1_zero_plus_unpolarized_A(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the quantity (1 + epsilon^2)^{2}:
            one_plus_epsilon_squared_squared = (tf.constant(1.0, dtype = tf.float32) + epsilon**2)**2

            # (2): Calculate a fancy, annoying quantity:
            fancy_y_stuff = tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (3): Calculate the prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * tf.sqrt(2.) * lepton_helicity * lepton_energy_fraction_y * (tf.constant(2.0, dtype = tf.float32) - lepton_energy_fraction_y) * (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) / one_plus_epsilon_squared_squared

            # (4): Calculate the coefficient
            s_1_zero_plus_unp_A = prefactor * fancy_y_stuff * squared_hadronic_momentum_transfer_t * shorthand_k**2 / squared_Q_momentum_transfer
            
            # (4.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_1_zero_plus_unp_A to be:\n{s_1_zero_plus_unp_A}")

            # (5): Return the coefficient:
            return s_1_zero_plus_unp_A

        except Exception as ERROR:
            print(f"> Error in calculating s_1_zero_plus_unp_A for Interference Term:\n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_s_2_zero_plus_unpolarized(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        """
        """

        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity epsilon^2/2:
            epsilon_squared_over_2 = epsilon**2 / 2.

            # (3): Calculate the annoying y quantity:
            y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (4): Calculate the bracket term:
            bracket_term = tf.constant(1.0, dtype = tf.float32) + ((tf.constant(1.0, dtype = tf.float32) + epsilon_squared_over_2 / x_Bjorken) / (tf.constant(1.0, dtype = tf.float32) + epsilon_squared_over_2)) * x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (5): Calculate the prefactor:
            prefactor = tf.constant(8.0, dtype = tf.float32) * lepton_helicity * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) * shorthand_k * lepton_energy_fraction_y / root_one_plus_epsilon_squared**4
            
            # (6): Calculate the coefficient:
            s_2_zero_plus_unp = prefactor * (tf.constant(1.0, dtype = tf.float32) + epsilon_squared_over_2) * bracket_term
            
            # (6.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_2_zero_plus_unp to be:\n{s_2_zero_plus_unp}")

            # (7): Return the coefficient:
            return s_2_zero_plus_unp

        except Exception as ERROR:
            print(f"> Error in calculating s_2_zero_plus_unp for Interference Term:\n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_s_2_zero_plus_unpolarized_V(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate the annoying y quantity:
            y_quantity = tf.sqrt(tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32)))

            # (4): Calculate the prefactor:
            prefactor = -tf.constant(8.0, dtype = tf.float32) * tf.sqrt(2.) * lepton_helicity * y_quantity * shorthand_k * lepton_energy_fraction_y * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
            
            # (5): Calculate the coefficient:
            s_2_zero_plus_unp_V = prefactor * (tf.constant(1.0, dtype = tf.float32) - (tf.constant(1.0, dtype = tf.float32) - tf.constant(2.0, dtype = tf.float32) * x_Bjorken) * t_over_Q_squared)
            
            # (5.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated s_2_zero_plus_unp_V to be:\n{s_2_zero_plus_unp_V}")

            # (6): Return the coefficient:
            return s_2_zero_plus_unp_V

        except Exception as ERROR:
            print(f"> Error in calculating s_2_zero_plus_unp_V for Interference Term:\n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_s_2_zero_plus_unpolarized_A(
        self,
        lepton_helicity: float,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        epsilon: float,
        lepton_energy_fraction_y: float,
        shorthand_k: float,
        verbose: bool = False) -> float:
        try:

            # (1): Calculate the recurrent quantity sqrt(1 + epsilon^2):
            root_one_plus_epsilon_squared = tf.sqrt(tf.constant(1.0, dtype = tf.float32) + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate 1 - x_{B}:
            one_minus_xb = tf.constant(1.0, dtype = tf.float32) - x_Bjorken

            # (4): Calculate the annoying y quantity:
            y_quantity = tf.constant(1.0, dtype = tf.float32) - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / tf.constant(4.0, dtype = tf.float32))

            # (5): Calculate the main term:
            main_term = tf.constant(4.0, dtype = tf.float32) * one_minus_xb + tf.constant(2.0, dtype = tf.float32) * epsilon**2 + tf.constant(4.0, dtype = tf.float32) * t_over_Q_squared * (tf.constant(4.0, dtype = tf.float32) * x_Bjorken * one_minus_xb + epsilon**2)
            
            # (6): Calculate part of the prefactor:
            prefactor = -tf.constant(2.0, dtype = tf.float32) * tf.sqrt(tf.constant(2.0, dtype = tf.float32) * y_quantity) * lepton_helicity * shorthand_k * lepton_energy_fraction_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
            
            # (7): Calculate the coefficient:
            c_2_zero_plus_unp_A = prefactor * main_term
            
            # (7.1): If verbose, log the output:
            if verbose:
                print(f"> Calculated c_2_zero_plus_unp_A to be:\n{c_2_zero_plus_unp_A}")

            # (8): Return the coefficient:
            return c_2_zero_plus_unp_A

        except Exception as ERROR:
            print(f"> Error in calculating c_2_zero_plus_unp_A for Interference Term:\n> {ERROR}")
            return 0.

@register_keras_serializable()
class BSALayer(tf.keras.layers.Layer):

    def call(self, inputs):

        # (X): Unpack the inputs into the CFFs and the kinematics:
        kinematics, cffs = inputs

        # (X): Extract the eight CFFs from the DNN:
        real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et = tf.unstack(cffs, axis = -1)

        # (X): Extract the kinematics from the DNN:
        q_squared, x_bjorken, t, k, phi = tf.unstack(kinematics, axis = -1)

        # (X): DUMMY COMPUTATION FOR NOW:
        bsa = real_H**2 + imag_H**2 + tf.constant(0.5, dtype = tf.float32) * tf.cos(phi) * real_E + 0.1 * q_squared

        # (X): Re-cast the BSA into a single value (I think):
        return tf.expand_dims(bsa, axis = -1)

class SimultaneousFitModel(tf.keras.Model):

    def __init__(self, model):
        super(SimultaneousFitModel, self).__init__()

        self.model = model

    def train_step(self, data):
        """
        ## Description:
        This particular function is *required* if you are going to 
        inherit a tf Model class. 
        """

        # (X): Unpack the data:
        x_training_data, y_training_data = data

        if SETTING_DEBUG:
            print("> [DEBUG]: Unpacked training data.")

        # (X): Use TensorFlow's GradientTape to unfold each step of the training scheme:
        with tf.GradientTape() as gradient_tape:
            
            if SETTING_DEBUG:
                tf.print(f"> [DEBUG]: Now unraveling gradient tape...")

            # (X): Evaluate the model by passing in the input data:
            predicted_cff_values = self.model(x_training_data, training = True)

            if SETTING_DEBUG:
                tf.print(f"> [DEBUG]: Predicted CFF values: {predicted_cff_values}")

            # (X): Use the custom-defined loss function to compute a scalar loss:
            computed_loss = simultaneous_fit_loss(y_training_data, predicted_cff_values, x_training_data)

            if SETTING_DEBUG:
                tf.print(f"> [DEBUG]: Loss computed! {computed_loss}")

        # (X): Compute the gradients during backpropagation:
        computed_gradients = gradient_tape.gradient(computed_loss, self.trainable_variables)

        if SETTING_DEBUG:
            tf.print(f"> [DEBUG]: Computed batch gradients: {computed_gradients}")

        # (X): Call the TF model's optimizer:
        self.optimizer.apply_gradients(
            zip(
                computed_gradients,
                self.trainable_variables
                ))

        if SETTING_DEBUG:
            print("> [DEBUG]: Gradients applied with optimizer!")

def build_simultaneous_model():
    """
    ## Description:
    We initialize a DNN model used to predict the eight CFFs:
    """

    # (1): Initialize the Network with Uniform Random Sampling: [-0.1, -0.1]:
    initializer = tf.keras.initializers.RandomUniform(
        minval = -0.1,
        maxval = 0.1,
        seed = None)
    
    # (X): Define the input to the DNN:
    input_kinematics = Input(shape = (5, ), name = "input_layer")

    # (X): Slice Q², xB, t (first 3 components) to be fed into the neural network
    input_cff_features = Lambda(lambda x: x[:, :3], name = "kinematics_input_split")(input_kinematics)

    # (X): Pass the inputs through a densely-connected hidden layer:
    x = Dense(
        _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1,
        activation = "relu",
        kernel_initializer = initializer)(input_kinematics)

    # (X): Pass the inputs through a densely-connected hidden layer:
    x = Dense(
        _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2,
        activation = "relu",
        kernel_initializer = initializer)(x)

    # (X): Pass the inputs through a densely-connected hidden layer:
    x = Dense(
        _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3,
        activation = "relu",
        kernel_initializer = initializer)(x)

    # (X): Pass the inputs through a densely-connected hidden layer:
    x = Dense(
        _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4,
        activation = "relu",
        kernel_initializer = initializer)(x)

    # (X): Pass the inputs through a densely-connected hidden layer:
    output_cffs = tf.keras.layers.Dense(
        _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5,
        activation = "linear",
        kernel_initializer = initializer,
        name = "cff_output_layer")(x)
    
    # (X): Concatenate the two:
    full_input = Concatenate(axis = -1)([input_kinematics, output_cffs])

    # (8): Compute, algorithmically, the cross section:
    cross_section_value = CrossSectionLayer()(full_input)

    # (8): Compute, algorithmically, the BSA:
    # | We are NOT READY FOR THIS YET:
    # bsa_value = BSALayer()([input_kinematics, output_cffs])

    # (9): Define the model as as Keras Model:
    simultaneous_fit_model = Model(
        inputs = input_kinematics,
        outputs = cross_section_value,
        name = "cross-section-model")

    if SETTING_DEBUG or SETTING_VERBOSE:
        print(simultaneous_fit_model.summary())

    # (X): Compile the model with a fixed learning rate using Adam and the custom loss:
    simultaneous_fit_model.compile(
        optimizer = tf.keras.optimizers.Adam(_HYPERPARAMETER_LEARNING_RATE),
        loss = tf.keras.losses.MeanSquaredError())

    # (X): Return the model:
    return simultaneous_fit_model