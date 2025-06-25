"""
A centralized source of truth for the various names we 
call things in this code.
"""

# (1): argparser's description:
_ARGPARSE_DESCRIPTION = "Run DNN Replicas to extract the CFFs."

# (2): argparers *argument flag* for the datafile:
_ARGPARSE_ARGUMENT_INPUT_DATAFILE = '--input-datafile'

# (3): argparser's description for the argument `input-datafile`:
_ARGPARSE_ARGUMENT_DESCRIPTION_INPUT_DATAFILE = 'Path to the input CSV file.'

# (4): argparers *argument flag* for the datafile:
_ARGPARSE_ARGUMENT_KINEMATIC_SET_NUMBER = '--kinematic-set'

# (5): argparser's description for the argument `kinematic-set`:
_ARGPARSE_ARGUMENT_DESCRIPTION_KINEMATIC_SET_NUMBER = 'An integer specifying which kinematic set to analyze.'

# (6): argparer's *argument flag* for the datafile:
_ARGPARSE_ARGUMENT_NUMBER_REPLICAS = '--number-of-replicas'

# (7): argparser's description for the argument `kinematic-set`:
_ARGPARSE_ARGUMENT_DESCRIPTION_NUMBER_REPLICAS = 'The number of DNN Replicas to run.'

# (8): argparer's *argument flag* for the datafile:
_ARGPARSE_ARGUMENT_VERBOSE = '--verbose'

# (9): argparer's *argument flag* for the datafile:
_ARGPARSE_ARGUMENT_DESCRIPTION_VERBOSE = 'Enable verbose logging.'

# (10): "Generalized" column name for kinematic set/bin:
_COLUMN_NAME_KINEMATIC_BIN = "bin"

# (11): "Generalized" column name for lepton beam energy:
_COLUMN_NAME_LEPTON_MOMENTUM = "k"

# (12): "Generalized" column name for Q^{2}:
_COLUMN_NAME_Q_SQUARED = "QQ"

# (13): "Generalized" column name for t (hadronic momentum transfer):
_COLUMN_NAME_T_MOMENTUM_CHANGE = "t"

# (14): "Generalized" column name for x_Bjokren:
_COLUMN_NAME_X_BJORKEN = "x_b"

# (14): "Generalized" column name for azimuthal phi angle:
_COLUMN_NAME_AZIMUTHAL_PHI = "phi"