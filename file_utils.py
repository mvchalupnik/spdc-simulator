import numpy as np
from datetime import datetime
import os


def get_current_time():
    """Return current time as a string."""
    return datetime.now().strftime('%H-%M-%S_')

def create_directory(data_directory_path: str, subfolder_name: str = None):
    """
    Create a new directory if directory for the current day does not exist.

    :param data_directory_path: The path to the new directory.
    :param subfolder_name: The name subfolder in the directory

    :return dir_string: The path to the created directory, including subfolder if applied
    """
    date_string = datetime.today().strftime('%Y_%m_%d')
    dir_string = data_directory_path + "/" + date_string

    if not os.path.exists(data_directory_path):
        os.mkdir(data_directory_path)

    if not os.path.exists(dir_string):
        os.mkdir(dir_string)

    if subfolder_name is not None:
        dir_string = dir_string + '/' + subfolder_name + '/'
        if not os.path.exists(dir_string):
            os.mkdir(dir_string)

    return dir_string
