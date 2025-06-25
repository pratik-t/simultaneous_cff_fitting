"""
The code containing the custom loss functions.
"""

# 3rd Party Libraries | bkm10_lib:
from bkm10_lib import DifferentialCrossSection, CFFInputs

def simultaneous_loss(true_values, predicted_values, kinematic_settings):
    """
    ### Description:
    We need a custom TF loss to minimize due to the inclusion of 
    several observable quantities.
    """
    # (X): We use the bkm10_lib to fit the kinematic settings:

    # (X): We need to extract the CFFs from the predicted values:

    # (X): We use the bkm10_lib to set up th
    cff_settings = CFFInputs(
        compton_form_factor_h = complex(1., 1.),
        compton_form_factor_h_tilde = complex(1., 1.),
        compton_form_factor_e = complex(1., 1.),
        compton_form_factor_e_tilde = complex(1., 1.))
