from simulate_spdc import simulate_rings, simulate_ring_momentum, simulate_conditional_probability, C

from file_utils import create_directory
import numpy as np

def main():
    """ main function """
    print("Hello world")
    dir_string = create_directory(data_directory_path="plots")

    pump_wavelength = 405e-9# 405.9e-9 # Pump wavelength in meters
    down_conversion_wavelength = 810e-9# 811.8e-9 # Wavelength of down-converted photons in meters
    thetap = 28.95 * np.pi / 180
  #  thetap = 28.84 * np.pi / 180
   # thetap = 28.64 * np.pi / 180
    thetap = 40.99 * np.pi / 180
    thetap = 41.78 * np.pi / 180

    w0 = 388e-6 # beam waist in meters, page 8
    d = 107.8e-2 # pg 15
    z_pos = 35e-3 # 35 millimeters, page 15
    crystal_length = 0.002  # Length of the nonlinear crystal in meters

    ######### SIMULATE RING MOMENTUM

    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span": 0.05,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,#["2s", "2i"],
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "num_plot_qx_points": 500,
        "num_plot_qy_points": 500,
        "thetap": 28.84 * np.pi / 180,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span": 0.045,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,#["2s", "2i"],
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    ######### SIMULATE CONDITIONAL PROBABILITY
    simulation_parameters = {
        "thetap": 28.84 * np.pi / 180,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "num_plot_x_points": 100,
        "idler_x_span": 0.003,
        "signal_x_pos": 0.0016,
        "momentum_span_wide": 0.045,
        "momentum_span_narrow": 0.001,
        "num_samples_momentum_wide": 200,
        "num_samples_momentum_narrow": 20,
        "idler_y_pos": 0,
        "signal_y_pos": 0,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 1,
        "simulation_cores": 4,
        "save_directory": dir_string,
    }

    simulate_conditional_probability(simulation_parameters=simulation_parameters)


    # ################ SIMULATE RINGS

    simulation_parameters = {
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "momentum_span_wide": 0.045,
        "momentum_span_narrow": 0.001,
        "num_samples_momentum_wide": 80, #800 for good result
        "num_samples_momentum_narrow": 50, #50 for good result
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "phase_matching_type": 2,
        "simulation_cores": 4,
        "save_directory": dir_string,
    }

    simulate_rings(simulation_parameters=simulation_parameters)


if __name__=="__main__": 
    main()