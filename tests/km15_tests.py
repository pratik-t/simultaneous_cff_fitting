"""
Testing KM15 outputs.
"""

# Native Library | unittest
import unittest

# 3rd Party Library | NumPy:
import numpy as np

# utilities > km15
from utilities.km15 import compute_km15_cffs

_TEST_Q_SQUARED = 1.82
_TEST_X_BJORKEN = 0.343
_TEST_T = -0.172

class TestKM15CFFs(unittest.TestCase):
    """
    We test the scripts that us the KM15 GPD models to generate CFFs that
    we then compare to the DNN's prediction.
    """

    def test_output_ranges(self):
        """
        ## Description:
        Ensure that the KM15 functions we are using are producing the correct
        values because we use them to compare if the fit was decent.

        ## Notes:
        THE TEST VALUES ARE OBVIOUSLY STOLEN DIRECTLY FROM THE FUNCTION OUTPUT. THEY WERE NOT DERIVED
        USING OTHER SOFTWARE. THIS MAKES IT SO THE TEST WILL TRIVIALLY PASS EVERY TIME. THIS TEST
        IS CURRENTLY MEANINGLESS, BUT WE WROTE IT HERE AS A PLACEHOLDER FOR FUTURE, BETTER TESTS.
        """
        real_h, imag_h, real_e, real_ht, im_ht, real_et = compute_km15_cffs(_TEST_Q_SQUARED, _TEST_X_BJORKEN, _TEST_T)

        self.assertAlmostEqual(real_h, -2.774174980557727)
        self.assertAlmostEqual(imag_h, 2.9136429850046968)
        self.assertAlmostEqual(real_e, 2.211953495028034)
        self.assertAlmostEqual(real_ht, 1.2435792535554773)
        self.assertAlmostEqual(im_ht, 1.3907833342037639)
        self.assertAlmostEqual(real_et, 141.31637219464)

if __name__ == "__main__":
    unittest.main()
