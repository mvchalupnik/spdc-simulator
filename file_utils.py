from datetime import datetime
import os
import pickle
import json
import uuid


def get_current_time():
    """Return current time and file identifier as a string."""
    return datetime.now().strftime("%Y_%m_%d_%H-%M-%S_ID-") + str(uuid.uuid4())

def create_directory(data_directory_path: str, subfolder_name: str = None):
    """
    Create a new directory if directory for the current day does not exist.

    :param data_directory_path: The path to the new directory.
    :param subfolder_name: The name of an optional subfolder to create in the directory and save data within.

    :return dir_string: The path to the created directory, including subfolder if applied
    """
    date_string = datetime.today().strftime("%Y_%m_%d")
    dir_string = data_directory_path + "/" + date_string

    if not os.path.exists(data_directory_path):
        os.mkdir(data_directory_path)

    if not os.path.exists(dir_string):
        os.mkdir(dir_string)

    if subfolder_name is not None:
        dir_string = dir_string + "/" + subfolder_name + "/"
        if not os.path.exists(dir_string):
            os.mkdir(dir_string)

    return dir_string

def save_data(data_to_pickle: list, simulation_parameters: dict, save_directory_name: str, time_str: str, data_name:str):
    """ Save pickled data and save parameter dict to a text file.

    :param data_to_pickle: A list of data to save via pickle.
    :param simulation_parameters: A dict of the parameters used for the simulation.
    :param save_directory_name: The name of the directory to save to.
    :param time_str: The string containing the date and time stamp.
    :param data_name: The name for the data.
    """
    # Save data to a pickled file #Should turn this into a function to reduce duplicate code TODO
    with open(f"{save_directory_name}/{time_str}_{data_name}.pkl", "wb",) as file:
        pickle.dump(data_to_pickle, file)

    # Save parameters to a pickled file
    with open(f"{save_directory_name}/{time_str}_{data_name}_params.pkl", "wb",) as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory_name}/{time_str}_{data_name}_params.txt", "w",) as file:
        file.write(json.dumps(simulation_parameters))

def save_time_info(time_elapsed: float, save_directory_name: str, time_str: str, data_name:str):
    """ Save info on how long the function took to run to a text file.

    :param time_elapsed: Time elapsed during function call.
    :param save_directory_name: The name of the directory to save to.
    :param time_str: The string containing the date and time stamp.
    :param data_name: The name for the data.
    """
    # Save time to a text file
    time_info = {"Time Elapsed in seconds": time_elapsed}
    with open(f"{save_directory_name}/{time_str}_{data_name}_time.txt", "w",) as file:
        file.write(json.dumps(time_info))
