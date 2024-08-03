import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy
from joblib import Parallel, delayed
from scipy.stats import qmc
import functools
from scipy.stats import norm
import itertools
import pickle
import json
from file_utils import get_current_time, create_directory 

# Constants
C = 2.99792e8  # Speed of light, in meters per second

# def monte_carlo_integration_position(f, dq, dr, num_samples=1):
#     # Generate random samples within the bounds [-dr, dr] for each variable
#     # x_samples = np.random.uniform(-dr, dr, num_samples)
#     # y_samples = np.random.uniform(-dr, dr, num_samples)

#     # Evaluate the function at each sample point
#     func_values = np.zeros(num_samples, dtype='complex128') # Technically won't be complex here
#     x_sample = dr# x_samples[n] 0.00025 when integrating to 1 mm
#     y_sample = 0 #0#y_samples[n]
#     g = functools.partial(f, x_pos_integrate=x_sample, y_pos_integrate=y_sample)
#     func_values[0] = monte_carlo_integration_momentum(g, dq)

#     # Calculate the average value of the function
#     avg_value = np.mean(func_values)
    
#     # The volume of the integration region
#     volume = (2 * dr)**2
    
#     # Estimate the integral as the average value times the volume
#     integral_estimate = avg_value * volume
    
#     return func_values[0]

def monte_carlo_integration_momentum(f, dq, num_samples=20000):
    """
    Integrate function `f` using the Monte Carlo method along four dimensions of momentum,
    qx, qy for both the signal and idler photons.
    """
    ## Generate random samples within the bounds [-dq, dq] for each variable
    qix_samples = np.random.uniform(-dq, dq, num_samples)
    qiy_samples = np.random.uniform(-dq, dq, num_samples)
    qsx_samples = np.random.uniform(-dq, dq, num_samples)
    qsy_samples = np.random.uniform(-dq, dq, num_samples)

    # Evaluate the function at each sample point
    func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples)

    # Calculate the average value of the function
    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dq)**4
    
    # Estimate the integral as the average value times the volume
    integral_estimate = avg_value * volume

    # Square the integral at the end
    integral_estimate_sq = np.abs(integral_estimate)**2
    
    return integral_estimate_sq

def grid_integration_position(f, dq, dr, num_samples=6):
    """
    Integrate along x and y. First pass function to be integrated along four dimensions
    of momentum. 
    """
    # Generate from a grid samples within the bounds [-dr, dr] for each variable
    x_samples = np.linspace(-dr, dr, num_samples)
    y_samples = np.linspace(-dr, dr, num_samples)
    coord_pairs = list(itertools.product(x_samples, y_samples))

    # Evaluate the function at each sample point
    func_values = np.zeros(len(coord_pairs), dtype='complex128') # Technically won't be complex here
    for n in range(len(coord_pairs)):
        x_sample, y_sample = coord_pairs[n]
        g = functools.partial(f, x_pos_integrate=x_sample, y_pos_integrate=y_sample)
        func_values[n] = monte_carlo_integration_momentum(g, dq)

    # Calculate the average value of the function
    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dr)**2
    
    # Estimate the integral as the average value times the volume
    integral_estimate = avg_value * volume
    
    return integral_estimate

def n_o(wavelength):
    """
    Ordinary refractive index for BBO crystal, from Sellmeier equations for BBO.
    
    :param wavelength: Wavelength of light entering the crystal.
    """
    lambda_sq = wavelength**2
    return np.sqrt(np.abs(2.7405 + 0.0184 / (lambda_sq - 0.0179) - 0.0155 * lambda_sq))

def n_e(wavelength):
    """
    Extraordinary refractive index for BBO crystal, from Sellmeier equations for BBO.

    :param wavelength: Wavelength of light entering the crystal.
    """
    lambda_sq = wavelength**2
    return np.sqrt(np.abs(2.3730 + 0.0128 / (lambda_sq - 0.0156) - 0.0044 * lambda_sq))

def phase_matching(delta_k, L):
    """
    Return the phase matching function given a delta_k and length L. 
    
    :param delta_k: Change in wave vector k
    :param L: Length of crystal in meters
    """
    return L * np.sinc(delta_k * L / 2) * np.exp(1j * delta_k * L / 2)

def alpha(thetap, lambdap):
    """ Return the alpha_p coefficient as a function of pump incidence tilt angle theta_p and 
    pump wavelength lambda.

    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param lambdap: Pump wavelength, in meters
    """
    alpha = ((n_o(lambdap)**2 - n_e(lambdap)**2) * np.sin(thetap) * np.cos(thetap)) / \
    (n_o(lambdap)**2 * np.sin(thetap)**2 + n_e(lambdap)**2 * np.cos(thetap)**2)
    return alpha

def beta(thetap, lambdap):
    """ Return the beta_p coefficient as a function of pump incidence tilt angle theta_p and 
    pump wavelength lambda.

    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param lambdap: Pump wavelength, in meters
    """
    beta = (n_o(lambdap) * n_e(lambdap)) / \
    (n_o(lambdap)**2 * np.sin(thetap)**2 + n_e(lambdap)**2 * np.cos(thetap)**2)
    return beta

def gamma(thetap, lambdap):
    """ Return the gamma coefficient as a function of pump incidence tilt angle theta_p and 
    pump wavelength lambda.

    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param lambdap: Pump wavelength, in meters
    """
    gamma = n_o(lambdap) / \
    np.sqrt((n_o(lambdap)**2 * np.sin(thetap)**2 + n_e(lambdap)**2 * np.cos(thetap)**2))
    return gamma

def eta(thetap, lambdap):
    """
    Return the eta coefficient as a function of pump incidence tilt angle theta_p and 
    pump wavelength lambda.
    
    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param lambdap: Pump wavelength, in meters
    """
    eta = (n_o(lambdap) * n_e(lambdap)) / \
    np.sqrt((n_o(lambdap)**2 * np.sin(thetap)**2 + n_e(lambdap)**2 * np.cos(thetap)**2))
    return eta

def delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas):
    """ Return delta_k for type I phase matching, for BBO crystal.
    
    :param qsx: k-vector in the x direction for signal
    :param qix: k-vector in the x direction for idler
    :param qsy: k-vector in the y direction for signal
    :param qiy: k-vector in the y direction for idler
    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param omegap: Angular frequency of the pump beam (TODO, overdefined)
    :param omegai: Angular frequency of the idler beam
    :param omegas: Angular frequency of the signal beam
    """
    lambdas = (2 * np.pi * C) / omegas
    lambdai = (2 * np.pi * C) / omegai
    lambdap = (2 * np.pi * C) / omegap

    qpx = qsx + qix # Conservation of momentum
    qpy = qsy + qiy # Conservation of momentum
    qs_abs = np.sqrt(qsx**2 + qsy**2)
    qi_abs = np.sqrt(qix**2 + qiy**2)

    delta_k = n_o(lambdas) * omegas / C + n_o(lambdai) * omegai / C - eta(thetap, lambdap) * omegap / C + \
    C / (2 * eta(thetap, lambdap) * omegap) * (beta(thetap, lambdap)**2 * qpx**2 + gamma(thetap, lambdap)**2 * qpy**2) + \
    alpha(thetap, lambdap) * (qsx + qix) - C / (2 * n_o(lambdas) * omegas) * qs_abs**2 - \
    C / (2 * n_o(lambdai) * omegai) * qi_abs**2

    return delta_k

def pump_function(qpx, qpy, kp, omega, w0, d):
    """ Function for the Gaussian pump beam. (equation 31)

    :param qpx: k-vector in the x direction for pump
    :param qpy: k-vector in the y direction for pump
    :param kp: k-vector in the z direction for pump
    :param omega: Pump frequency
    """
    qp_abs = np.sqrt(qpx**2 + qpy**2)
    V = np.exp(-qp_abs**2 * w0**2 / 4) * np.exp(-1j * qp_abs**2 * d / (2 * kp))
    return V

def get_rate_integrand(x_pos, y_pos, thetap, omegai, omegas, simulation_parameters):
    """
    Return the integrand used to calculate rates and conditional probabilities
    """
    # Also can multiply by detector efficiencies, and a constant dependent on epsilon_0 and chi_2

    # z is distance away from crystal along pump propagation direction
    ks = omegas / C
    ki = omegai / C
    kpz = (omegas + omegai) / C # This is on page 8 in the bottom paragraph on the left column
    omegap = omegas + omegai # This is on page 8 in the bottom paragraph on the left column
#    momentum_span = simulation_parameters.get("momentum_span")
    z_pos = simulation_parameters["z_pos"]
    w0 = simulation_parameters["pump_waist_size"]
    d = simulation_parameters["pump_waist_distance"]
    crystal_length = simulation_parameters["crystal_length"]

    def rate_integrand(qix, qiy, qsx, qsy, x_pos_integrate, y_pos_integrate, integrate_over):
        if integrate_over == "signal": # TODO fix to be less confusing
            # Fix idler, integrate over signal
            xs_pos = x_pos_integrate
            ys_pos = y_pos_integrate
            xi_pos = x_pos
            yi_pos = y_pos
        elif integrate_over == "idler":
            # Fix signal, integrate over idler
            xs_pos = x_pos
            ys_pos = y_pos
            xi_pos = x_pos_integrate
            yi_pos = y_pos_integrate
        else:
            # Raise error #TODO actually raise error
            print("error")

        qs_dot_rhos = (qsx * xs_pos + qsy * ys_pos)
        qi_dot_rhoi = (qix * xi_pos + qiy * yi_pos)
        qs_abs = np.sqrt(qsx**2 + qsy**2)
        qi_abs = np.sqrt(qix**2 + qiy**2)

        integrand = np.exp(1j * (ks + ki) * z_pos) * pump_function(qix + qsx, qiy + qsy, kpz, omegap, w0, d) * phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * \
        np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))
        return integrand

    # dqix = (omegai / C) * momentum_span
    # dqiy = (omegai / C) * momentum_span
    # dqsx = (omegas / C) * momentum_span
    # dqsy = (omegas / C) * momentum_span

    return rate_integrand # Returning a function?


def calculate_conditional_probability(x_pos, y_pos, thetap, omegai, omegas, dr, simulation_parameters):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param x_pos: Location of signal (idler) photon in the x direction a distance z away from the crystal
    :param y_pos: Location of signal (idler) photon in the y direction a distance z away from the crystal
    :param dr: One half the area of real space centered around the origin which the signal and idler
        will be integrated over.
    """
    x_sample = dr
    y_sample = 0 #TODO fix
    momentum_span = simulation_parameters.get("momentum_span")
    num_samples = simulation_parameters["num_momentum_integration_points"]
    dqi = (omegai / C) * momentum_span

    # dqsx = (omegas / C) * momentum_span # May want to incorporate this
    # dqsy = (omegas / C) * momentum_span

    rate_integrand = get_rate_integrand(x_pos, y_pos, thetap, omegai, omegas, simulation_parameters)
    rate_integrand_signal = functools.partial(rate_integrand, integrate_over="idler", x_pos_integrate=x_sample, y_pos_integrate=y_sample)

    result_signal = monte_carlo_integration_momentum(f=rate_integrand_signal, dq=dqi, num_samples=num_samples) # TODO expand to integrate over signal
#    result_signal = monte_carlo_integration_position(f=rate_integrand_signal, dq=dqi, dr=dr) # TODO expand to integrate over signal

    return result_signal #TODO expand to include type II and also return idler


def calculate_pair_generation_rate(x_pos, y_pos, thetap, omegai, omegas, dr, simulation_parameters):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param x_pos: Location of signal (idler) photon in the x direction a distance z away from the crystal
    :param y_pos: Location of signal (idler) photon in the y direction a distance z away from the crystal
    :param dr: One half the area of real space centered around the origin which the signal and idler
        will be integrated over.
    """
    momentum_span = simulation_parameters.get("momentum_span")
    dqi = (omegai / C) * momentum_span

    rate_integrand = get_rate_integrand(x_pos, y_pos, thetap, omegai, omegas, simulation_parameters)
    rate_integrand_signal = functools.partial(rate_integrand, integrate_over="idler")
#    rate_integrand_idler = functools.partial(rate_integrand, integrate_over="signal")

    result_signal = grid_integration_position(rate_integrand_signal, dqi, dr)

    return result_signal #TODO expand to include type II and also return idler


def simulate_ring_momentum(simulation_parameters):
    """
    Simulate and plot the ring for a plane in momentum space, given fixed (x, y) for signal and fixed (x, y) for idler.

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.    
    """
    num_plot_qx_points = simulation_parameters["num_plot_qx_points"]
    num_plot_qy_points = simulation_parameters["num_plot_qy_points"]

    thetap = simulation_parameters["thetap"] # Incident pump angle, in Radians
    omegap = simulation_parameters["omegap"] # Pump frequency (Radians / sec)
    omegai = simulation_parameters["omegai"] # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"] # Signal frequency (Radians / sec)
    momentum_span = simulation_parameters["momentum_span"]
    signal_x_pos = simulation_parameters["signal_x_pos"]
    signal_y_pos = simulation_parameters["signal_y_pos"]
    idler_x_pos = simulation_parameters["idler_x_pos"]
    idler_y_pos = simulation_parameters["idler_y_pos"]
    z_pos = simulation_parameters["z_pos"]
 
    save_directory = simulation_parameters["save_directory"]

    dqix = (omegai / C) * momentum_span
    dqiy = (omegai / C) * momentum_span
    dqsx = (omegas / C) * momentum_span
    dqsy = (omegas / C) * momentum_span

    rate_integrand = get_rate_integrand(signal_x_pos, signal_y_pos, thetap, omegai, omegas, simulation_parameters)

    x = np.linspace(-dqix, dqix, num_plot_qx_points)
    y = np.linspace(-dqiy, dqiy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)
    Z = np.real(rate_integrand(X, Y, X, Y, idler_x_pos, idler_y_pos, "signal"))
    plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    plt.xlabel("$q_x$ ($q_{xi} = q_{xs}$)")
    plt.ylabel("$q_y$ ($q_{yi} = q_{ys}$)")

    # Get current time for file name
    time_str = get_current_time()

    plt.savefig(f"{save_directory}/{time_str}_momentum.png", dpi=300)

    # Save parameters and data
    with open(f"{save_directory}/{time_str}_momentum.pkl", "wb") as file:
        pickle.dump(Z, file)

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_momentum_params.pkl", "wb") as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_momentum_params.txt", 'w') as file:
        file.write(json.dumps(simulation_parameters))


def simulate_ring_slice(simulation_parameters):
    """
    Simulate and plot a slice of the conditional probabilities of detecting the signal photon, 
    given the idler photon is fixed (with location swept across the x-axis).

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    TODO use kwargs instead of a dict!
    """
    start_time = time.time()

    num_plot_x_points = simulation_parameters["num_plot_x_points"] #.get
    x_span = simulation_parameters["x_span"] # Span in the x-direction to plot conditional probability of signal over, in meters
    thetap = simulation_parameters["thetap"] # Incident pump angle, in Radians
    omegap = simulation_parameters["omegap"] # Pump frequency (Radians / sec)
    omegai = simulation_parameters["omegai"] # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"] # Signal frequency (Radians / sec)
    idler_x_span = simulation_parameters["idler_x_span"] # Span in the x-direction to fix idler at
    idler_x_increment = simulation_parameters["idler_x_increment"] # Increment size to change idler by in the x-direction
    signal_y_pos = simulation_parameters["signal_y_pos"]

    save_directory = simulation_parameters["save_directory"]
    seed = simulation_parameters["random_seed"]

    np.random.seed(seed)

    x = np.linspace(-x_span, x_span, num_plot_x_points)
    plt.figure(figsize=(8, 6))
    sweep_points = np.arange(-idler_x_span, idler_x_span, idler_x_increment) #TODO pass in

    probs = np.zeros([len(sweep_points), len(x)]) # initialize array to save data (check todo)
    for i, idler_x_pos in enumerate(sweep_points):
        calculate_conditional_probability_vec = np.vectorize(calculate_conditional_probability)
        z1 = calculate_conditional_probability_vec(x, signal_y_pos, thetap=thetap, omegai=omegai, omegas=omegas, dr=idler_x_pos, simulation_parameters=simulation_parameters) # TODO use a diff function
        probs[i] = z1
        plt.plot(x, z1, label=idler_x_pos)
    plt.title( "Conditional probability of signal given idler at different locations on x-axis" )
    plt.legend()
 
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    plt.savefig(f"{save_directory}/{time_str}_rings_slice.png", dpi=300)

    #plt.show()

    # Save parameters and data
    with open(f"{save_directory}/{time_str}_ring_slice.pkl", "wb") as file:
        pickle.dump(probs, file)

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_ring_slice_params.pkl", "wb") as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_ring_slice_params.txt", 'w') as file:
        file.write(json.dumps(simulation_parameters))

    # Save time to a text file
    time_info = {"Time Elapsed in seconds" : end_time - start_time}
    with open(f"{save_directory}/{time_str}_ring_slice_{num_plot_x_points}_time.txt", 'w') as file:
        file.write(json.dumps(time_info))


def simulate_rings(simulation_parameters):
    """
    Simulate and plot entangled pair rings by integrating the conditional probability of detecting the signal photon given detecting the 
    idler photon, integrating over the possible positions of the idler photon. 

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    """
    start_time = time.time()

    seed = simulation_parameters["random_seed"]

    np.random.seed(seed)

    num_plot_x_points = simulation_parameters["num_plot_x_points"] #.get
    num_plot_y_points = simulation_parameters["num_plot_y_points"]
    x_span = simulation_parameters["x_span"] # Span in the x-direction to plot over, in meters
    y_span = simulation_parameters["y_span"] # Span in the y-direction to plot over, in meters
    thetap = simulation_parameters["thetap"] # Incident pump angle, in Radians
    omegap = simulation_parameters["omegap"] # Pump frequency (Radians / sec)
    omegai = simulation_parameters["omegai"] # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"] # Signal frequency (Radians / sec)
    momentum_span = simulation_parameters["momentum_span"] # Extent of span to integrate in momentum space over across x axis and in y axis for both signal and idler (fraction of omega / C)
    num_momentum_integration_points = simulation_parameters["num_momentum_integration_points"]  # Number of points to integrate over in momentum space
    grid_integration_size = simulation_parameters["grid_integration_size"] # Size of square root of grid for integration in real space (todo improve name)
    pump_waist_size = simulation_parameters["pump_waist_size"] # Size of pump beam waist
    pump_waist_distance = simulation_parameters["pump_waist_distance"] # Distance of pump waist from crystal (meters)
    z_pos = simulation_parameters["z_pos"] # View location in the z direction, from crystal (meters)
    crystal_length = simulation_parameters["crystal_length"] # Length of the crystal, in meters

    num_cores = simulation_parameters["simulation_cores"] # Number of cores to use in the simulation
    save_directory = simulation_parameters["save_directory"]

    # Create a grid of x and y values
    x = np.linspace(-x_span, x_span, num_plot_x_points)
    y = np.linspace(-y_span, y_span, num_plot_y_points)
    X, Y = np.meshgrid(x, y)

    calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)

    # Run calculate_pair_generation_rate in parallel
    parallel_calculate = functools.partial(calculate_pair_generation_rate_vec, thetap=thetap, omegai=omegai, omegas=omegas, dr=grid_integration_size, simulation_parameters=simulation_parameters)
    Z1 = Parallel(n_jobs=num_cores)(delayed(parallel_calculate)(xi, yi) for xi in x for yi in y)
    Z = np.reshape(np.array(Z1), [num_plot_y_points, num_plot_x_points]).T #TODO test this

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(Z), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title( "BBO crystal entangled photons rates" )
    plt.savefig(f"{save_directory}/{time_str}_rings.png", dpi=300)    

    # Save data to a pickled file
    with open(f"{save_directory}/{time_str}_rings.pkl", "wb") as file:
        pickle.dump(Z, file)

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_rings_params.pkl", "wb") as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_rings_params.txt", 'w') as file:
        file.write(json.dumps(simulation_parameters))

    # Save time to a text file
    time_info = {"Time Elapsed in seconds" : end_time - start_time}
    with open(f"{save_directory}/{time_str}_rings_time.txt", 'w') as file:
        file.write(json.dumps(time_info))

    #plt.show()


# Todo, momentum space plot

# todo, type 2 noncollinear

# Plot total output power as a function of theta_p and other params
# TODO


def main():
    """ main function """
    print("Hello world")
    dir_string = create_directory(data_directory_path="plots")

    pump_wavelength = 405e-9# 405.9e-9 # Pump wavelength in meters
    down_conversion_wavelength = 810e-9# 811.8e-9 # Wavelength of down-converted photons in meters
    thetap = 28.95 * np.pi / 180
#    thetap = 0 * np.pi / 180

    w0 = 388e-6 # beam waist in meters, page 8
    d = 107.8e-2 # pg 15
    z_pos = 35e-3 # 35 millimeters, page 15
    crystal_length = 0.002  # Length of the nonlinear crystal in meters

    simulation_parameters = {
        "num_plot_qx_points": 1000,
        "num_plot_qy_points": 1000,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "signal_x_pos": 0,
        "signal_y_pos": 0,
        "idler_x_pos": 0,
        "idler_y_pos": 0,
        "momentum_span": 0.0014,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "save_directory": dir_string,
    }

    simulate_ring_momentum(simulation_parameters=simulation_parameters)


    simulation_parameters = {
        "num_plot_x_points": 100,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "x_span": 3e-3,
        "idler_x_span": 0.0012,
        "idler_x_increment": 0.0001,
        "momentum_span": 0.014,
        "num_momentum_integration_points": 20000,
        "idler_y_pos": 0,
        "signal_y_pos": 0,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "save_directory": dir_string,
        "random_seed": 1
    }

    simulate_ring_slice(simulation_parameters=simulation_parameters)


    simulation_parameters = {
        "num_plot_x_points": 6,
        "num_plot_y_points": 6,
        "thetap": thetap,
        "omegap": (2 * np.pi * C) / pump_wavelength,
        "omegai": (2 * np.pi * C) / down_conversion_wavelength,
        "omegas": (2 * np.pi * C) / down_conversion_wavelength,
        "x_span": 2e-3,
        "y_span": 2e-3,
        "momentum_span": 0.0014,
        "num_momentum_integration_points": 20000,
        "grid_integration_size": 6,
        "pump_waist_size": w0,
        "pump_waist_distance": d,
        "z_pos": z_pos,
        "crystal_length": crystal_length,
        "simulation_cores": 4,
        "save_directory": dir_string,
        "random_seed": 1
    }

    simulate_rings(simulation_parameters=simulation_parameters)
    # Todo, create folder in specified directory with date, etc. For now just save.


if __name__=="__main__": 
    main()
