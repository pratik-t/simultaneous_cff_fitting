"""
We need to verify the custom loss is able to return numbers at the least.
"""

# Native Library | unittest
import unittest

# 3rd Party Libraries | TensorFlow:
import tensorflow as tf

from models.loss_functions import MeanSquaredError, SimultaneousFitLoss

class TestCustomLoss(unittest.TestCase):
    """
    Tests any custom losses we build.
    """

    def test_scalar_input(self):
        """
        ## Description:
        Ensure we can eagerly generate numbers with the custom loss
        function *and* that it matches with the native MSE() function 
        provides by TF.
        """

        # (X): Define some true values for the first part of the MSE:
        true_values = tf.constant([[1.0, 2.0], [3.0, 4.0]])

        # (X): Define some predicted values for the other part of the MSE:
        predicted_values = tf.constant([[1.1, 1.9], [2.9, 4.2]])

        # (X): We test if the MSE() and the CURRENT custom loss match EXACTLY.
        # | This will CHANGE IN THE FUTURE.
        self.assertAlmostEqual(
            first = tf.keras.losses.MeanSquaredError()(true_values, predicted_values),
            second = MeanSquaredError()(true_values, predicted_values),
            places = 8)
        
        self.assertAlmostEqual(
            first = tf.keras.losses.MeanSquaredError()(true_values, predicted_values),
            second = SimultaneousFitLoss()(true_values, predicted_values),
            places = 8)

if __name__ == "__main__":
    unittest.main()
