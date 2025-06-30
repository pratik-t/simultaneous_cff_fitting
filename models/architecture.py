"""
Here, we define the DNN model architecture used for 
any fitting procedure.
"""

# 3rd Party Library | TensorFlow:
import tensorflow as tf

# 3rd Party Library | bkm10:
from bkm10_lib import DifferentialCrossSection, CFFInputs, BKM10Inputs, backend

# 3rd Party Library | TensorFlow:
from tensorflow.keras.layers import Input, Concatenate, Dense

# 3rd Party Library | TensorFlow:
from tensorflow.keras.models import Model

from models.loss_functions import simultaneous_fit_loss

from statics.static_strings import _HYPERPARAMETER_LEARNING_RATE
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4
from statics.static_strings import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5

SETTING_VERBOSE = True
SETTING_DEBUG = True

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

        # (X): Unpack the inputs into the CFFs and the kinematics.
        # | The inputs will be a KerasTensor of shape (None, 5) and another
        # | KerasTensor of shape (None, 8). That is, the five kinematic
        # | quantities and the eight numbers for the CFFs.
        kinematics, cffs = inputs

        # (X): Extract the eight CFFs from the DNN:
        real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et = tf.unstack(cffs, axis = -1)

        # (X): Extract the kinematics from the DNN:
        q_squared, x_bjorken, t, k, phi = tf.unstack(kinematics, axis = -1)

        # (X): DUMMY COMPUTATION FOR NOW:
        # differential_cross_section = real_H**2 + imag_H**2 + 0.5 * tf.cos(phi) * real_E + 0.1 * q_squared

        # (X): The calculation requires that we use TF not NumPy to do stuff:
        backend.set_backend('tensorflow')

        # (X): Set up the BKM10 kinematic inputs:
        bkm_inputs = BKM10Inputs(
            squared_Q_momentum_transfer = q_squared,
            x_Bjorken = x_bjorken,
            squared_hadronic_momentum_transfer_t = t,
            lab_kinematics_k = k)

        # (X): Set up the BKM10 CFF inputs:
        cff_inputs = CFFInputs(
            compton_form_factor_h = backend.math.complex(real_H, imag_H),
            compton_form_factor_h_tilde = backend.math.complex(real_Ht, imag_Ht),
            compton_form_factor_e = backend.math.complex(real_E, imag_E),
            compton_form_factor_e_tilde = backend.math.complex(real_Et, imag_Et))

        # (X): Construct the required configuration dictionary:
        configuration = {
            "kinematics": bkm_inputs,
            "cff_inputs": cff_inputs,
            "target_polarization": self.target_polarization,
            "lepton_beam_polarization": self.lepton_beam_polarization,
            "using_ww": self.using_ww
        }

        # (X): Compute the differential cross section accordingly:
        differential_cross_section = DifferentialCrossSection(configuration).compute_cross_section(phi)

        # (X): Re-cast sigma into a single value (I think):
        return tf.expand_dims(differential_cross_section, axis = -1)
    
class BSALayer(tf.keras.layers.Layer):

    def call(self, inputs):

        # (X): Unpack the inputs into the CFFs and the kinematics:
        kinematics, cffs = inputs

        # (X): Extract the eight CFFs from the DNN:
        real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et = tf.unstack(cffs, axis = -1)

        # (X): Extract the kinematics from the DNN:
        q_squared, x_bjorken, t, phi = tf.unstack(kinematics, axis = -1)

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

        # (X): Use TensorFlow's GradientTape to unfold each step of the training scheme:
        with tf.GradientTape() as gradient_tape:

            # (X): Evaluate the model by passing in the input data:
            predicted_cff_values = self.model(x_training_data, training = True)

            # (X): Use the custom-defined loss function to compute a scalar loss:
            computed_loss = simultaneous_fit_loss(y_training_data, predicted_cff_values, x_training_data)

        # (X): Compute the gradients during backpropagation:
        computed_gradients = gradient_tape.gradient(computed_loss, self.trainable_variables)

        # (X): Call the TF model's optimizer:
        self.optimizer.apply_gradients(
            zip(
                computed_gradients,
                self.trainable_variables
            )
        )

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
        kernel_initializer = initializer)(x)
    
    # (4): Combine the kinematics as a single list:
    # kinematics_and_cffs = Concatenate(axis = 1)([input_kinematics, output_cffs])

    # (8): Compute, algorithmically, the cross section:
    cross_section_value = CrossSectionLayer()([input_kinematics, output_cffs])

    # (8): Compute, algorithmically, the BSA:
    bsa_value = BSALayer()([input_kinematics, output_cffs])

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