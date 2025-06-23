"""
We put logic here that handle the creation, deletion, and "scanning" of 
directory contents.
"""

# (1): Native Library | os
import os

def does_directory_exist(path_to_directory: os.path) -> bool:
    """
    ## Description:
    Here, we just look for a given os.path within a given
    directory context.

    ## Arguments:
    path_to_directory: os.path

    ## Returns:
    does_the_path_exist: bool
    """

    # (1): Immediately run the os method:
    does_the_path_exist = os.path.exists(path_to_directory)

    # (2): Run the result of exists --- should be `bool`, but this is unchecked!
    return does_the_path_exist
    
def create_directory(filepath_to_directory):
    """
    ## Description:
    Rely on os' `.makedirs()` to construct a *nested*
    folder tree.

    ## Arguments:
    filepath_to_directory: os.path

    ## Returns:
    Nothing!
    """

    # (1): Immediately run os's `.makedirs()` method:
    os.makedirs(filepath_to_directory)
        
def create_replica_directories(kinematic_set_number, replica_number):
    """
    ## Description:
    Later...

    ## Arguments:
    kinematic_set_number: int 

    replica_number: int

    ## Returns:
    Nothing!
    """

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # Wait a second...