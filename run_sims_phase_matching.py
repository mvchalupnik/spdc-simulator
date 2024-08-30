from simulate_spdc import simulate_phase_matching_function, C
from file_utils import create_directory
import numpy as np


def run_phase_matching_sims():
    """ Run simulations to plot the phase-matching function (sinc function) for different conditions.

    Parameters:
    :param thetap: Angle theta in Radians along which pump photon enters BBO crystal (about y-axis).
    :param omega0: Nominal angular frequency of the signal and idler.
    :param fraction_delta_omega: The range in angular frequency, as a fraction of omega0, to sweep
        the signal and idler over.
    :param momentum_signal_x: The x component of the k-vector of the signal photon (meters).
    :param momentum_signal_y: The y component of the k-vector of the signal photon (meters).
    :param momentum_idler_x: The x component of the k-vector of the idler photon (meters).
    :param momentum_idler_y: The y component of the k-vector of the idler photon (meters).
    :param crystal_length: The length of the crystal (meters).
    :param phase_matching_type: The type of phase-matching (type I or type II).
    :param save_directory: The name of the directory to store plots in.
    """
    dir_string = create_directory(data_directory_path="plots")

    down_conversion_wavelength = 810e-9 # Wavelength of down-converted photons in meters
    crystal_length = 0.002  # Length of the nonlinear crystal in meters

    # Simulate phase-matching function
    ## Type I
    simulation_parameters = {
        "thetap": 28.95 * np.pi / 180,
        "omega0": (2 * np.pi * C) / down_conversion_wavelength,
        "fraction_delta_omega": 0.05,
        "momentum_signal_x": 0.042,
        "momentum_signal_y": 0,
        "momentum_idler_x": -0.042,
        "momentum_idler_y": 0,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,
        "save_directory": dir_string,
    }
    simulate_phase_matching_function(simulation_parameters=simulation_parameters)

    # The momentum locations are at the intersections of the two cones in momentum space.
    simulation_parameters = {
        "thetap": 41.78 * np.pi / 180,
        "omega0": (2 * np.pi * C) / down_conversion_wavelength,
        "fraction_delta_omega": 0.016,
        "momentum_signal_x": -0.000232,
        "momentum_signal_y": -0.03609,
        "momentum_idler_x": 0.000232,
        "momentum_idler_y": 0.03609,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "save_directory": dir_string,
    }
    simulate_phase_matching_function(simulation_parameters=simulation_parameters)

if __name__ == "__main__":
    run_phase_matching_sims()
