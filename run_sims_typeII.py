from simulate_spdc import simulate_rings, C

from file_utils import create_directory
import numpy as np

def run_sims_type_II():
    """
    Run simulations to reproduce the simulated type II down-converted photon intensity in
    Figure 9 from Suman Karan et al 2020 J. Opt. 22 083501.

    Parameters for parameter dict:
    :param thetap: Angle theta in Radians along which pump photon enters BBO crystal (about y-axis).
    :param omegai: Angular frequency of the idler.
    :param omegas: Angular frequency of the signal.
    :param momentum_span_wide_x: One half of the interval of k-vector (by fraction of maximum k-vector)
        along x for the idler, to integrate over.
    :param momentum_span_wide_y: One half of the interval of k-vector (by fraction of maximum k-vector)
        along y for the idler, to integrate over.
    :param momentum_span_narrow_x: One half of the interval of the difference in k-vectors along x
        (by fraction of maximum k-vector) for the signal and idler.
    :param momentum_span_narrow_y: One half of the interval of the difference in k-vectors along y
        (by fraction of maximum k-vector) for the signal and idler.
    :param num_samples_momentum_wide_x: The number of samples to integrate over along x for the momentum_span_wide_x interval.
    :param num_samples_momentum_wide_y: The number of samples to integrate over along y for the momentum_span_wide_y interval.
    :param num_samples_momentum_narrow_x: The number of samples to integrate over along y for the momentum_span_narrow_x interval.
    :param num_samples_momentum_narrow_y: The number of samples to integrate over along y for the momentum_span_narrow_y interval.
    :param pump_waist_size: Size of pump beam waist (meter).
    :param pump_waist_distance: Distance of pump waist from crystal (meters).
    :param z_pos: The view location in the z direction, from crystal (meters).
    :param crystal_length: The length of the crystal (meters).
    :param phase_matching_type: The type of phase-matching (type I or type II).
    :param num_jobs: The number of jobs to parallelize the batched function evaluation over.
    :param save_directory: The name of the directory to store plots in.
    """
    dir_string = create_directory(data_directory_path="plots")

    pump_wavelength = 405e-9 # Pump wavelength in meters
    down_conversion_wavelength = 810e-9 # Wavelength of down-converted photons in meters
    w0 = 388e-6 # beam waist in meters, page 8
    d = 107.8e-2 # pg 15
    z_pos = 35e-3 # 35 millimeters, page 15
    crystal_length = 0.002  # Length of the nonlinear crystal in meters


    ############## SIMULATE RINGS
    # Four different incident pump angles
    simulation_parameters = {
        "thetap": 40.48 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.08,
        "momentum_span_wide_y": 0.04,
        "momentum_span_narrow_x": 0.07,
        "momentum_span_narrow_y": 0.07,
        "num_samples_momentum_wide_x": 1040,
        "num_samples_momentum_wide_y": 520,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "thetap": 40.99 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.051,
        "momentum_span_wide_y": 0.05,
        "momentum_span_narrow_x": 0.06,
        "momentum_span_narrow_y": 0.06,
        "num_samples_momentum_wide_x": 714,
        "num_samples_momentum_wide_y": 700,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "thetap": 41.40 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.0585*0.8,
        "momentum_span_wide_y": 0.06*0.8,
        "momentum_span_narrow_x": 0.06,
        "momentum_span_narrow_y": 0.06,
        "num_samples_momentum_wide_x": 913,
        "num_samples_momentum_wide_y": 936,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "thetap": 41.78 * np.pi / 180,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.0384*0.9,
        "momentum_span_wide_y": 0.064*0.9,
        "momentum_span_narrow_x": 0.06,
        "momentum_span_narrow_y": 0.06,
        "num_samples_momentum_wide_x": 768,
        "num_samples_momentum_wide_y": 1280,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)


if __name__=="__main__":
    run_sims_type_II()
