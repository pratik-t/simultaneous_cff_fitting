"""
Testing KM15 outputs.
"""

# Native Library | unittest
import unittest

# 3rd Party Library | NumPy:
import numpy as np

# utilities > km15
from utilities.km15 import compute_km15_cffs

class TestKM15CFFs(unittest.TestCase):

    def test_output_ranges(self):
        """
        ## Description:
        Make sure the output values are finite and real.
        """
        q2, xb, t = 1.82, 0.343, -0.172
        result = compute_km15_cffs(q2, xb, t)
        for value in result:
            print(value)


if __name__ == "__main__":
    unittest.main()
