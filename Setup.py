# Setup the output folders if they are not existing already

import os

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - directory_path (str): The path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Create main saving folder:
create_directory("saved")
create_directory("saved/statistics")

# Create subfolders for each manipulation
create_directory("saved/manipulation_1")
create_directory("saved/manipulation_2")
create_directory("saved/manipulation_3")

# Create the subfolders for flowers and macbeth for each manipulation folders

# Manipulation 1
create_directory("saved/manipulation_1/flowers")
create_directory("saved/manipulation_1/macbeth")

# Subfolder for each interpolation methods tested
create_directory("saved/manipulation_1/flowers/Linear")
create_directory("saved/manipulation_1/flowers/Hermite")
create_directory("saved/manipulation_1/flowers/Spline")

create_directory("saved/manipulation_1/macbeth/Linear")
create_directory("saved/manipulation_1/macbeth/Hermite")
create_directory("saved/manipulation_1/macbeth/Spline")

# Manipulation 2
create_directory("saved/manipulation_2/flowers")
create_directory("saved/manipulation_2/macbeth")

# Manipulation 3
create_directory("saved/manipulation_3/flowers")
create_directory("saved/manipulation_3/macbeth")
