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
_COLUMN_NAME_Q_SQUARED = "q_squared"

# (13): "Generalized" column name for t (hadronic momentum transfer):
_COLUMN_NAME_T_MOMENTUM_CHANGE = "t"

# (14): "Generalized" column name for x_Bjokren:
_COLUMN_NAME_X_BJORKEN = "x_b"

# (14): "Generalized" column name for azimuthal phi angle:
_COLUMN_NAME_AZIMUTHAL_PHI = "phi"

# (X): Required subdirectories | analysis:
_DIRECTORY_ANALYSIS = 'analysis'

# (X): Required subdirectories | analysis > data:
_DIRECTORY_DATA = 'data'

# (X): Required subdirectories | analysis > data > raw:
_DIRECTORY_DATA_RAW = 'raw'

# (X): Required subdirectories | analysis > data > replicas:
_DIRECTORY_DATA_REPLICAS = 'replicas'

# (X): Required subdirectories | analysis > replicas:
_DIRECTORY_REPLICAS = 'replicas'

# (X): Required subdirectories | analysis > replicas > fits:
_DIRECTORY_REPLICAS_FITS = 'fits'

# (X): Required subdirectories | analysis > replicas > losses:
_DIRECTORY_REPLICAS_LOSSES = 'losses'

# (X): Required subdirectories | analysis > replicas > performance:
_DIRECTORY_REPLICAS_PERFORMANCE = 'performance'

# (X): Required subdirectories | analysis > replicas > pseudodata:
_DIRECTORY_REPLICAS_PSEUDODATA = 'pseudodata'

# (X): Required subdirectories list:
REQUIRED_SUBDIRECTORIES_LIST = [
    f"{_DIRECTORY_DATA}/{_DIRECTORY_DATA_RAW}",
    f"{_DIRECTORY_DATA}/{_DIRECTORY_DATA_REPLICAS}",
    f"{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_FITS}",
    f"{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_LOSSES}",
    f"{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_PERFORMANCE}",
    f"{_DIRECTORY_REPLICAS}/{_DIRECTORY_REPLICAS_PSEUDODATA}",
]

# (X): DNN Hyperparameters | Learning Rate:
_HYPERPARAMETER_LEARNING_RATE = 0.001

# (X): DNN Hyperparameters | LR "patience":
_HYPERPARAMETER_LR_PATIENCE = 400

# (X): DNN Hyperparameters | LR factor:
_HYPERPARAMETER_LR_FACTOR = 0.9

# (X): DNN Hyperparameters | EarlyStop "patience":
_HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER = 1000

# (X): DNN Hyperparameters | Neurons in 1st Layer:
_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1 = 64

# (X): DNN Hyperparameters | Neurons in 2nd Layer:
_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2 = 64

# (X): DNN Hyperparameters | Neurons in 3rd Layer:
_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3 = 64

# (X): DNN Hyperparameters | Neurons in 4th Layer:
_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4 = 64

# (X): DNN Hyperparameters | Neurons in 5th Layer:
_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5 = 8

# (X): DNN Training Settings | Number of Replicas:
_HYPERPARAMETER_NUMBER_OF_REPLICAS = 300

# (X): DNN Training Settings | Number of Replicas:
_HYPERPARAMETER_NUMBER_OF_EPOCHS = 400

# (X): DNN Training Settings | Number of Replicas:
_HYPERPARAMETER_BATCH_SIZE = 16

# (X): DNN train/test split *decimal*:
_DNN_TRAIN_TEST_SPLIT_PERCENTAGE = 0.2

# (X): DNN verbosity setting:
_DNN_VERBOSE_SETTING = 2

# (X):
_TF_FORMAT_KERAS = "keras"

# (X):
_TF_FORMAT_H5 = ".5"

# (X):
_FIGURE_FORMAT_EPS = "eps"

# (X):
_FIGURE_FORMAT_SVG = "svg"

# (X):
_FIGURE_FORMAT_PNG = "png"

# (X):
_FILE_FORMAT_CSV = "csv"

# TEMPORARY!
_COLUMN_NAME_CROSS_SECTION = "sigma"
_COLUMN_NAME_CROSS_SECTION_ERROR = "sigma_stat_plus"