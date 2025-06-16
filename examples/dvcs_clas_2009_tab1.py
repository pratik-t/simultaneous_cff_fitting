"""
We attempt to make a few plots from the paper that features this data 
so that we can confirm that the we can reproduce them and, equivalently
that we've copied the data correctly.
"""

# External Library | Pandas
import pandas as pd

# External Library | Matplotlib
import matplotlib.pyplot as plt

# (1): Specify the name (and location) of the data file:
DATA_FILE = './data/dvcs_CLAS_2009_tab1.csv'

# (2): Use Pandas to read the .csv:
pandas_dataframe = pd.read_csv(
    filepath_or_buffer = DATA_FILE,
    delimiter = ",")

# (3): Verify the data columns:
print(f"> Found columns:\n> {pandas_dataframe.columns}")

# (4): Partition the DF on a fixed kinematic set, which, in this example, is fixed on 1.22:
fixed_kinematic_set_dataframe = pandas_dataframe[pandas_dataframe['q_squared'] == 1.22]

# (5): Start to wrestle with matploitlib:
figure, axis = plt.subplots(figsize = (8, 5))

# (6): We need an errorbar plot to try to match Fig. 14:
axis.errorbar(
    x = fixed_kinematic_set_dataframe['phi'],
    y = fixed_kinematic_set_dataframe['ALU'],
    yerr = fixed_kinematic_set_dataframe['del_ALU'],
    fmt = 'none',
    marker = "+",
    color = "blue")

# (7): Set the x-label:
axis.set_xlabel(r"Azimuthal Angle $\phi$ (degrees)", fontsize = 14)

# (8): Set the y-label:
axis.set_ylabel(r"$A_{LU}$", fontsize = 14)

# (9): Set the y-limit as it appears to be in the Fig. 14:
axis.set_ylim(
    bottom = -0.45,
    top = 0.45)

# (10): Save the figure:
figure.savefig("./examples/dvcs_clas_2009_fig14.png")
