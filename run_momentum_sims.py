from simulate_spdc import simulate_ring_momentum, C

from file_utils import create_directory
import numpy as np


def run_momentum_sims():
    """
    Run simulations to simulate the momentum of type I and type II down-converted photons from a BBO crystal.

    Parameters for parameter dict:
    :param num_plot_qx_points: The number of points to plot along x in momentum space.
    :param num_plot_qy_points: The number of points to plot along y in momentum space.
    :param thetap: Angle theta in Radians along which pump photon enters BBO crystal (about y-axis).
    :param omegai: Angular frequency of the idler.
    :param omegas: Angular frequency of the signal.
    :param signal_x_pos: The x position of the signal photon (meters).
    :param signal_y_pos: The y position of the signal photon (meters).
    :param idler_x_pos: The x position of the idler photon (meters).
    :param idler_y_pos: The y position of the idler photon (meters).
    :param momentum_span_x: The fraction of total momentum to plot over, in the x direction.
    :param momentum_span_y: The fraction of total momentum to plot over, in the y direction.
    :param pump_waist_size: Size of pump beam waist (meter).
    :param pump_waist_distance: Distance of pump waist from crystal (meters).
    :param z_pos: The view location in the z direction, from crystal (meters).
    :param crystal_length: The length of the crystal (meters).
    :param phase_matching_type: The type of phase-matching (type I or type II).
    :param save_directory: The name of the directory to store plots in.
    """
    dir_string = create_directory(data_directory_path="plots")

    down_conversion_wavelength = (
        810e-9  # Wavelength of down-converted photons in meters
    )
    w0 = 388e-6  # beam waist in meters, page 8
    d = 107.8e-2  # pg 15
    z_pos = 35e-3  # 35 millimeters, page 15
    crystal_length = 0.002  # Length of the nonlinear crystal in meters

    ######### SIMULATE RING MOMENTUM
    ## Type I
    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": 28.95 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span_x": 0.045,
        "momentum_span_y": 0.045,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    # Type II
    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": 40.48 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span_x": 0.09,
        "momentum_span_y": 0.04,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": 40.99 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span_x": 0.051 * 2,
        "momentum_span_y": 0.05,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": 41.40 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span_x": 0.0585 * 2,
        "momentum_span_y": 0.06,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": 41.78 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span_x": 0.064 * 2,
        "momentum_span_y": 0.08,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)


if __name__ == "__main__":
    run_momentum_sims()
