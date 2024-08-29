from simulate_spdc import simulate_rings, simulate_ring_momentum, simulate_conditional_probability, C

from file_utils import create_directory
import numpy as np

def run_sims_type_I():
    """
    Run simulations to reproduce the simulated type I down-converted photon intensity in
    Figure 8 from Suman Karan et al 2020 J. Opt. 22 083501.

    Parameters: TODO
    z is the distance away from the crystal along pump propagation direction.

    """
    dir_string = create_directory(data_directory_path="plots")

    pump_wavelength = 405e-9 # Pump wavelength in meters
    down_conversion_wavelength = 810e-9 # Wavelength of down-converted photons in meters
    w0 = 388e-6 # beam waist in meters, page 8
    d = 107.8e-2 # pg 15
    z_pos = 35e-3 # 35 millimeters, page 15
    crystal_length = 0.002  # Length of the nonlinear crystal in meters


    ################ SIMULATE RINGS
    simulation_parameters = {
        "thetap": 28.64 * np.pi / 180,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.045,
        "momentum_span_wide_y": 0.045,
        "momentum_span_narrow_x": 0.035,
        "momentum_span_narrow_y": 0.035,
        "num_samples_momentum_wide_x": 800,
        "num_samples_momentum_wide_y": 800,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "thetap": 28.74 * np.pi / 180,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.045,
        "momentum_span_wide_y": 0.045,
        "momentum_span_narrow_x": 0.035,
        "momentum_span_narrow_y": 0.035,
        "num_samples_momentum_wide_x": 800,
        "num_samples_momentum_wide_y": 800,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "thetap": 28.95 * np.pi / 180,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide_x": 0.045,
        "momentum_span_wide_y": 0.045,
        "momentum_span_narrow_x": 0.035,
        "momentum_span_narrow_y": 0.035,
        "num_samples_momentum_wide_x": 800,
        "num_samples_momentum_wide_y": 800,
        "num_samples_momentum_narrow_x": 50,
        "num_samples_momentum_narrow_y": 50,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,
        "num_jobs": 20,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)


if __name__=="__main__": 
    run_sims_type_I()
