from simulate_spdc import simulate_rings, simulate_ring_momentum, simulate_ring_slice, C

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

    w0 = 388e-6 # beam waist in meters, page 8
    d = 107.8e-2 # pg 15
    z_pos = 35e-3 # 35 millimeters, page 15
    crystal_length = 0.002  # Length of the nonlinear crystal in meters

    ######### SIMULATE RING MOMENTUM

    simulation_parameters = {
        "num_plot_qx_points": 25,
        "num_plot_qy_points": 25,
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
        "save_directory": dir_string,
    }
    simulate_ring_momentum(simulation_parameters=simulation_parameters)

    ######### SIMULATE RING SLICE
    simulation_parameters = {
        "num_plot_x_points": 300,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_span": 0.003,
        "idler_x_span": 0.003,
        "idler_x_increment": 0.0016,
        "momentum_span": 0.05,
        "num_samples_coarse_momentum": 25,
        "num_samples_fine_momentum": 20000,
        "grid_width_fraction_momentum": 0.1,
        "fraction_of_coarse_points_momentum": 0.001,
        "idler_y_pos": 0,
        "signal_y_pos": 0,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "simulation_cores": 4,
        "save_directory": dir_string,
        "random_seed": 1
    }

    simulate_ring_slice(simulation_parameters=simulation_parameters)

    simulation_parameters = {
        "num_plot_x_points": 100,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_span": 0.003,
        "idler_x_span": 0.003,
        "idler_x_increment": 0.0001,
        "momentum_span": 0.06,
        "num_samples_coarse_momentum": 30,
        "num_samples_fine_momentum": 20000,
        "grid_width_fraction_momentum": 0.1,
        "fraction_of_coarse_points_momentum": 0.001,
        "idler_y_pos": 0,
        "signal_y_pos": 0,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "simulation_cores": 60,
        "save_directory": dir_string,
        "random_seed": 1
    }

    simulate_ring_slice(simulation_parameters=simulation_parameters)

    # ################ SIMULATE RINGS

    simulation_parameters = {
        "num_plot_x_points": 2,
        "num_plot_y_points": 2,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "x_span": 3e-3,
        "y_span": 3e-3,
        "momentum_span": 0.06,
        "num_samples_coarse_momentum": 30,
        "num_samples_fine_momentum": 20000,
        "grid_width_fraction_momentum": 0.1,
        "fraction_of_coarse_points_momentum": 0.001,
        "grid_integration_size": 20,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "simulation_cores": 112,
        "save_directory": dir_string,
        "random_seed": 1
    }

    simulate_rings(simulation_parameters=simulation_parameters)


if __name__=="__main__": 
    main()