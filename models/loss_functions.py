"""
The code containing the custom loss functions.

## Notes:

1. 2025/07/24:
    The class `SimultaneousFitLoss`, while as it stands doing nothing more
    than unraveling Keras' standard `MeanSquaredError()` for the purpose of
    future customization/scaling/generalizability, was giving much worse fits.
    In other words, using MSE() was performing MUCH better than the custom
    `SimultaneousFitLoss()` class even though the custom class was coded to do
    nothing more than unravel the same logic as MSE().
"""

# 3rd Party Libraries | TensorFlow:
import tensorflow as tf

# 3rd Party Libraries | bkm10_lib:
from bkm10_lib import DifferentialCrossSection, CFFInputs

class SimultaneousFitLoss(tf.keras.losses.Loss):
    """
    Welcome to the main loss function that we will use in training the
    network for the simultaneous fitting.
    """

    def __init__(self, name = "simultaneous_fit_loss"):
        
        super().__init__(name = name)

    def call(self, y_true, y_pred):
        """
        ## Description:
        A function that provides the logic to compare the DNN's predicitions
        with the true values it is trying to fit.

        ## Arguments:
        - y_true: tensor of true target values
        - y_pred: tensor of predicted values
        
        ## Returns:
        - scalar tensor: the mean squared error across all examples and outputs
        """
        # (1): Compute the element-wise difference
        residuals = y_true - y_pred

        # (2): Square the difference
        squared_residuals = tf.square(residuals)

        # (3): Compute the mean over all elements
        simultaneous_fit_loss = tf.reduce_mean(squared_residuals)

        return simultaneous_fit_loss

class MeanSquaredError(tf.keras.losses.Loss):
    """
    A custom loss function that does everything TF's MSE should do. We
    made this just to show that we understand its inner workings.
    """

    def __init__(self, name = "mean_squared_error"):
        
        super().__init__(name = name, reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, true_values, predicted_values):
        """
        ## Description:
        Provides the necessary logic to compare DNN predictions with
        true values.

        ## Arguments:
        - y_true: tensor of true target values
        - y_pred: tensor of predicted values
        
        ## Returns:
        - scalar tensor: the mean squared error across all examples and outputs
        """
        # (1): Compute the element-wise difference
        residuals = true_values - predicted_values

        # (2): Square the difference
        squared_residuals = tf.square(residuals)

        # (3): Compute the mean over all elements
        mean_squared_error = tf.reduce_mean(squared_residuals)

        return mean_squared_error