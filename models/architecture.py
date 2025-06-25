"""
Here, we define the DNN model architecture used for 
any fitting procedure.
"""

# 3rd Party Library | TensorFlow:
import tensorflow as tf

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
    input_kinematics = Input(shape=(5 ,), name = "input_layer")

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
    cff_predictions = Concatenate(axis = 1)([input_kinematics, output_cffs])

    # (8): Compute, algorithmically, the cross section:
    cross_section_value = CrossSectionLayer()(cff_predictions)

    # (8): Compute, algorithmically, the BSA
    bsa_value = BSALayer()(cff_predictions)

    # # (9): Define the model as as Keras Model:
    simultaneous_fit_model = Model(
        inputs = input_kinematics,
        outputs = cross_section_value,
        name = "cross-section-model")

    if SETTING_DEBUG or SETTING_VERBOSE:
        print(simultaneous_fit_model.summary())

    # (X):: Compile the model with a fixed learning rate using Adam and the custom loss:
    simultaneous_fit_model.compile(
        optimizer = tf.keras.optimizers.Adam(_HYPERPARAMETER_LEARNING_RATE),
        loss = tf.keras.losses.MeanSquaredError())

    # (X): Return the model:
    return simultaneous_fit_model
