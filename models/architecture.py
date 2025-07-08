"""
Here, we define the DNN model architecture used for 
any fitting procedure.
"""

# 3rd Party Library | NumPy

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

SETTING_VERBOSE = True
SETTING_DEBUG = True

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

        # (X): DUMMY COMPUTATION FOR NOW:
        differential_cross_section = self.compute_cross_section(inputs)

        # (X): The calculation requires that we use TF not NumPy to do stuff:
        backend.set_backend('tensorflow')

        # # (X): Set up the BKM10 kinematic inputs:
        # bkm_inputs = BKM10Inputs(
        #     squared_Q_momentum_transfer = q_squared,
        #     x_Bjorken = x_bjorken,
        #     squared_hadronic_momentum_transfer_t = t,
        #     lab_kinematics_k = k)

        # # (X): Set up the BKM10 CFF inputs:
        # cff_inputs = CFFInputs(
        #     compton_form_factor_h = backend.math.complex(real_H, imag_H),
        #     compton_form_factor_h_tilde = backend.math.complex(real_Ht, imag_Ht),
        #     compton_form_factor_e = backend.math.complex(real_E, imag_E),
        #     compton_form_factor_e_tilde = backend.math.complex(real_Et, imag_Et))

        # # (X): Construct the required configuration dictionary:
        # configuration = {
        #     "kinematics": bkm_inputs,
        #     "cff_inputs": cff_inputs,
        #     "target_polarization": self.target_polarization,
        #     "lepton_beam_polarization": self.lepton_beam_polarization,
        #     "using_ww": self.using_ww
        # }

        # # (X): Compute the differential cross section accordingly:
        # differential_cross_section = DifferentialCrossSection(configuration, verbose = True).compute_cross_section(phi)

        # (X): Re-cast sigma into a single value (I think):
        return tf.expand_dims(differential_cross_section, axis = -1)
    
    @tf.function
    def compute_cross_section(self, inputs):
        """
        ## Description:
        This is a *panic* function that will compute ALL of the required
        coefficients that go into the cross section *and* the cross-section
        itself.
        """
        # (X): The calculation requires that we use TF not NumPy to do stuff:
        backend.set_backend('tensorflow')

        # (X): Unpack the inputs into the CFFs and the kinematics.
        # | The inputs will be a KerasTensor of shape (None, 5) and another
        # | KerasTensor of shape (None, 8). That is, the five kinematic
        # | quantities and the eight numbers for the CFFs.
        kinematics, cffs = inputs

        # (X): Extract the eight CFFs from the DNN:
        real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et = tf.unstack(cffs, axis = -1)

        # (X): Extract the kinematics from the DNN:
        q_squared, x_bjorken, t, k, phi = tf.unstack(kinematics, axis = -1)

        # (X): Compute epsilon:
        epsilon = self.calculate_kinematics_epsilon(q_squared, x_bjorken)

        # (X): Compute "y":
        y = self.calculate_kinematics_lepton_energy_fraction_y(q_squared, k, epsilon)

        # (X): Comute skewness "xi":
        xi = self.calculate_kinematics_skewness_parameter(q_squared, x_bjorken, t)

        # (X):
        t_min = self.calculate_kinematics_t_min(q_squared, x_bjorken, epsilon)

        # (X):
        t_prime = self.calculate_kinematics_t_prime(t, t_min)

        # (X):
        k_tilde = self.calculate_kinematics_k_tilde(q_squared, x_bjorken, y, t, epsilon, t_min)

        # (X):
        capital_k = self.calculate_kinematics_k(q_squared, y, epsilon, k_tilde)

        # (X):
        k_dot_delta = self.calculate_k_dot_delta(q_squared, x_bjorken, t, phi, epsilon, y, k)

        # (X): Compute the three n = 0 unpolarized coefficients with TF:
        c0pp_tf = self.calculate_c_0_plus_plus_unpolarized(q_squared, t, t, epsilon, k, k)
        # c0ppv_tf  = bkm_formalism.calculate_c_0_plus_plus_unpolarized_v()
        # c0ppa_tf  = bkm_formalism.calculate_c_0_plus_plus_unpolarized_a()

        # # (X): Compute the three n = 1 unpolaried coefficients with TF:
        # c1pp_tf = bkm_formalism.calculate_c_1_plus_plus_unpolarized()
        # c1ppv_tf  = bkm_formalism.calculate_c_1_plus_plus_unpolarized_v()
        # c1ppa_tf  = bkm_formalism.calculate_c_1_plus_plus_unpolarized_a()

        # # (X): Compute the three n = 2 unpolaried coefficients with TF:
        # c2pp_tf = bkm_formalism.calculate_c_2_plus_plus_unpolarized()
        # c2ppv_tf  = bkm_formalism.calculate_c_2_plus_plus_unpolarized_v()
        # c2ppa_tf  = bkm_formalism.calculate_c_2_plus_plus_unpolarized_a()

        # # (X): Compute the three n = 3 unpolaried coefficients with TF:
        # c3pp_tf = bkm_formalism.calculate_c_3_plus_plus_unpolarized()
        # c3ppv_tf  = bkm_formalism.calculate_c_3_plus_plus_unpolarized_v()
        # c3ppa_tf  = bkm_formalism.calculate_c_3_plus_plus_unpolarized_a()

        # (X): Compute the nightmare that is curly C:
        # curly_c0pp_tf = bkm_formalism.calculate_curly_c_unpolarized()

        # coefficient_c_0 = c0pp_tf * curly_c0pp_tf

        cross_section = c0pp_tf * backend.math.cos(0. * phi)

        # cross_section = real_H**2 + imag_H**2 + 0.5 * tf.cos(phi) * real_E + 0.1 * q_squared

        return cross_section
    
    @tf.function
    def calculate_kinematics_epsilon(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float, 
        verbose: bool = False) -> float:
        try:

            # (1): Calculate Epsilon right away:
            epsilon = (2. * x_Bjorken * _MASS_OF_PROTON_IN_GEV) / tf.sqrt(squared_Q_momentum_transfer)

            # (1.1): If verbose, print the result:
            if verbose:
                print(f"> Calculated epsilon to be:\n{epsilon}")

            # (2): Return Epsilon:
            return epsilon
        
        except Exception as ERROR:
            print(f"> Error in computing kinematic epsilon:\n> {ERROR}")
            return 0.
    
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

            # (1.1): If verbose output, then print the result:
            if verbose:
                print(f"> Calculated y to be:\n{lepton_energy_fraction_y}")

            # (2): Return the calculation:
            return lepton_energy_fraction_y
        
        except Exception as ERROR:
            print(f"> Error in computing lepton_energy_fraction_y:\n> {ERROR}")
            return 0.

    @tf.function
    def calculate_kinematics_skewness_parameter(
        self,
        squared_Q_momentum_transfer: float,
        x_Bjorken: float,
        squared_hadronic_momentum_transfer_t: float,
        verbose: bool = False) -> float:
        try:

            # (1): The Numerator:
            numerator = (1. + (squared_hadronic_momentum_transfer_t / (2. * squared_Q_momentum_transfer)))

            # (2): The Denominator:
            denominator = (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))

            # (3): Calculate the Skewness Parameter:
            skewness_parameter = x_Bjorken * numerator / denominator

            # (3.1): If verbose, print the output:
            if verbose:
                print(f"> Calculated skewness xi to be:\n{skewness_parameter}")

            # (4): Return Xi:
            return skewness_parameter
        
        except Exception as ERROR:
            print(f"> Error in computing skewness xi:\n> {ERROR}")
            return 0.
    
    @tf.function
    def calculate_kinematics_t_min(
        self,
        squared_Q_momentum_transfer: float, 
        x_Bjorken: float, 
        epsilon: float, 
        verbose: bool = False) -> float:
        try:

            # (1): Calculate 1 - x_{B}:
            one_minus_xb = 1. - x_Bjorken

            # (2): Calculate the numerator:
            numerator = (2. * one_minus_xb * (1. - tf.sqrt(1. + epsilon**2))) + epsilon**2

            # (3): Calculate the denominator:
            denominator = (4. * x_Bjorken * one_minus_xb) + epsilon**2

            # (4): Obtain the t minimum
            t_minimum = -1. * squared_Q_momentum_transfer * numerator / denominator

            # (4.1): If verbose, print the result:
            if verbose:
                print(f"> Calculated t_minimum to be:\n{t_minimum}")

            # (5): Print the result:
            return t_minimum

        except Exception as ERROR:
            print(f"> Error calculating t_minimum: \n> {ERROR}")
            return 0.
        
    @tf.function
    def calculate_kinematics_t_prime(
        self,
        squared_hadronic_momentum_transfer_t: float,
        squared_hadronic_momentum_transfer_t_minimum: float,
        verbose: bool = False) -> float:
        try:

            # (1): Obtain the t_prime immediately
            t_prime = squared_hadronic_momentum_transfer_t - squared_hadronic_momentum_transfer_t_minimum

            # (1.1): If verbose, print the result:
            if verbose:
                print(f"> Calculated t prime to be:\n{t_prime}")

            # (2): Return t_prime
            return t_prime

        except Exception as ERROR:
            print(f"> Error calculating t_prime:\n> {ERROR}")
            return 0.
        
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
            one_minus_xb = 1. - x_Bjorken

            # (3): Calculate the crazy root quantity:
            second_root_quantity = (one_minus_xb * tf.sqrt((1. + epsilon**2))) + ((tmin_minus_t * (epsilon**2 + (4. * one_minus_xb * x_Bjorken))) / (4. * squared_Q_momentum_transfer))

            # (4): Calculate the first annoying root quantity:
            first_root_quantity = tf.sqrt(1. - lepton_energy_fraction_y - lepton_energy_fraction_y**2 * epsilon**2 / 4.)
            
            # (5): Calculate K_tilde
            k_tilde = tf.sqrt(tmin_minus_t) * tf.sqrt(second_root_quantity)

            # (6): Print the result of the calculation:
            if verbose:
                print(f"> Calculated k_tilde to be:\n{k_tilde}")

            # (7) Return:
            return k_tilde

        except Exception as ERROR:
            print(f"> Error in calculating K_tilde:\n> {ERROR}")
            return 0.
        
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
            prefactor = tf.sqrt(((1. - lepton_energy_fraction_y + (epsilon**2 * lepton_energy_fraction_y**2 / 4.)) / squared_Q_momentum_transfer))

            # (2): Calculate the remaining part of the term:
            kinematic_k = prefactor * k_tilde

            # (2.1); If verbose, log the output:
            if verbose:
                print(f"> Calculated kinematic K to be:\n{kinematic_k}")

            # (3): Return the value:
            return kinematic_k

        except Exception as ERROR:
            print(f"> Error in calculating derived kinematic K:\n> {ERROR}")
            return 0.
        
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
            prefactor = squared_Q_momentum_transfer / (2. * lepton_energy_fraction_y * (1. + epsilon**2))

            # (2): Second term in parentheses: Phi-Dependent Term: 2 K tf.cos(\phi)
            phi_dependence = 2. * kinematic_k * tf.cos(tf.constant(np.pi) - convert_degrees_to_radians(azimuthal_phi))
            
            # (3): Prefactor of third term in parentheses: \frac{t}{Q^{2}}
            ratio_delta_to_q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (4): Second term in the third term's parentheses: x_{B} (2 - y)
            bjorken_scaling = x_Bjorken * (2. - lepton_energy_fraction_y)

            # (5): Third term in the third term's parentheses: \frac{y \varepsilon^{2}}{2}
            ratio_y_epsilon = lepton_energy_fraction_y * epsilon**2 / 2.

            # (6): Adding up all the "correction" pieces to the prefactor, written as (1 + correction)
            correction = phi_dependence - (ratio_delta_to_q_squared * (1. - bjorken_scaling + ratio_y_epsilon)) + (ratio_y_epsilon)

            # (7): Writing it explicitly as "1 + correction"
            in_parentheses = 1. + correction

            # (8): The actual equation:
            k_dot_delta_result = -1. * prefactor * in_parentheses

            # (8.1): If verbose, print the output:
            if verbose:
                print(f"> Calculated k dot delta: {k_dot_delta_result}")

            # (9): Return the number:
            return k_dot_delta_result
        
        except Exception as E:
            print(f"> Error in calculating k.Delta:\n> {E}")
            return 0.

    @tf.function
    def calculate_c_0_plus_plus_unpolarized(self,
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
            root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)

            # (2): Calculate the recurrent quantity t/Q^{2}:
            t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (3): Calculate 1 + sqrt(1 + epsilon^{2}):
            one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared

            # (4): Calculate 2 - x_{B}:
            two_minus_xb = 2. - x_Bjorken

            # (5): Caluclate 2 - y:
            two_minus_y = 2. - lepton_energy_fraction_y

            # (6): Calculate the first term in the brackets:
            first_term_in_brackets = k_tilde**2 * two_minus_y**2 / (squared_Q_momentum_transfer * root_one_plus_epsilon_squared)

            # (7): Calculate the first part of the second term in brackets:
            second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
            
            # (8): Calculate the numerator of the second part of the second term in brackets:
            second_term_in_brackets_second_part_numerator = 2. * x_Bjorken * t_over_Q_squared * (two_minus_xb + 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * epsilon**2 / x_Bjorken) + epsilon**2
            
            # (9): Calculate the second part of the second term in brackets:
            second_term_in_brackets_second_part =  1. + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
            
            # (10): Calculate the prefactor:
            prefactor = -4. * two_minus_y * one_plus_root_epsilon_stuff / tf.pow(root_one_plus_epsilon_squared, 4)

            # (11): Calculate the coefficient
            c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)

            # (12): Return the coefficient:
            return c_0_plus_plus_unp

        except Exception as ERROR:
            print(f"> Error in calculating c_0_plus_plus_unp for Interference Term:\n> {ERROR}")
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
        bsa = real_H**2 + imag_H**2 + 0.5 * tf.cos(phi) * real_E + 0.1 * q_squared

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
                print(f"> [DEBUG]: Now unraveling gradient tape...")

            # (X): Evaluate the model by passing in the input data:
            predicted_cff_values = self.model(x_training_data, training = True)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Predicted CFF values: {predicted_cff_values}")

            # (X): Use the custom-defined loss function to compute a scalar loss:
            computed_loss = simultaneous_fit_loss(y_training_data, predicted_cff_values, x_training_data)

            if SETTING_DEBUG:
                print(f"> [DEBUG]: Loss computed! {computed_loss}")

        # (X): Compute the gradients during backpropagation:
        computed_gradients = gradient_tape.gradient(computed_loss, self.trainable_variables)

        if SETTING_DEBUG:
            print(f"> [DEBUG]: Computed batch gradients: {computed_gradients}")


        # (X): Call the TF model's optimizer:
        self.optimizer.apply_gradients(
            zip(
                computed_gradients,
                self.trainable_variables
            )
        )

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
        activation = "relu",
        kernel_initializer = initializer,
        name = "cff_output_layer")(x)
    
    # (4): Combine the kinematics as a single list:
    # kinematics_and_cffs = Concatenate(axis = 1)([input_kinematics, output_cffs])

    # (8): Compute, algorithmically, the cross section:
    cross_section_value = CrossSectionLayer()([input_kinematics, output_cffs])

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