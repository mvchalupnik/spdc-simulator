import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import functools
import itertools
import pickle
import json
from file_utils import get_current_time

# Constants
C = 2.99792e8  # Speed of light, in meters per second


def grid_integration_momentum(f, dqix, dqiy, dqx, dqy, num_samples_wide, num_samples_narrow, num_cores):
    """
    Integrate along momentum dimensions.
    """
    dqix_samples = np.linspace(-dqix, dqix, num_samples_wide)
    dqiy_samples = np.linspace(-dqiy, dqiy, num_samples_wide)

    # From conservation of momentum, dqix should be close to -dqsx and dqiy should be close to -dqsy
    # so we can write dqix = -dqsx + dqx and dqiy = -dqsy + dqy 
    # and then can use fewer points of integration along the dqx, dqy dimensions.
    dqx_samples = np.linspace(-dqx, dqx, num_samples_narrow)
    dqy_samples = np.linspace(-dqy, dqy, num_samples_narrow)

    # Generate the coordinate grid using meshgrid
    dqix_grid, dqiy_grid, dqx_grid, dqy_grid = np.meshgrid(dqix_samples, dqiy_samples, dqx_samples, dqy_samples,
                                                           indexing='ij')

    # Flatten the grids
    dqix_flat = dqix_grid.ravel()
    dqiy_flat = dqiy_grid.ravel()
    dqx_flat = dqx_grid.ravel()
    dqy_flat = dqy_grid.ravel()
    time1 = time.time()

    # Vectorized evaluation of the function f over the flattened grids
    func_grid = f(dqix_flat, dqiy_flat, dqx_flat, dqy_flat)
#    func_grid = Parallel(n_jobs=num_cores)(delayed(f)(dqix_flat[i], dqiy_flat[i], dqx_flat[i], dqy_flat[i]) for i in range(len(dqix_flat)))

    time2 = time.time()

    # Reshape grid
    reshaped_func_grid = np.reshape(func_grid, [len(dqix_samples), len(dqiy_samples), len(dqx_samples), len(dqy_samples)])

    # To do: do a Fourier transform on four axes
    ft_func_grid = np.fft.fftn(reshaped_func_grid)
    ft_func_grid_shifted = np.fft.fftshift(ft_func_grid)
    # Return the absolute value of this grid squared
    time3 = time.time()
    print(time2-time1)
    print(time3-time2)

    return np.abs(ft_func_grid_shifted)**2

def n_o(wavelength):
    """
    Ordinary refractive index for BBO crystal, from Sellmeier equations for BBO.
    
    :param wavelength: Wavelength of light entering the crystal.
    """
    lambda_sq_in_microns = (wavelength*10**6)**2
    return np.sqrt(np.abs(2.7405 + 0.0184 / (lambda_sq_in_microns - 0.0179) - 0.0155 * lambda_sq_in_microns))

def n_e(wavelength):
    """
    Extraordinary refractive index for BBO crystal, from Sellmeier equations for BBO.

    :param wavelength: Wavelength of light entering the crystal.
    """
    lambda_sq_in_microns = (wavelength*10**6)**2
    return np.sqrt(np.abs(2.3730 + 0.0128 / (lambda_sq_in_microns - 0.0156) - 0.0044 * lambda_sq_in_microns))

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

def get_rate_integrand(thetap, omegai, omegas, simulation_parameters):
    """
    Return the integrand used to calculate rates and conditional probabilities
    """
    # Also can multiply by detector efficiencies, and a constant dependent on epsilon_0 and chi_2

    # z is distance away from crystal along pump propagation direction
    ks = omegas / C
    ki = omegai / C
    kpz = (omegas + omegai) / C # This is on page 8 in the bottom paragraph on the left column
    omegap = omegas + omegai # This is on page 8 in the bottom paragraph on the left column
    z_pos = simulation_parameters["z_pos"]
    w0 = simulation_parameters["pump_waist_size"]
    d = simulation_parameters["pump_waist_distance"]
    crystal_length = simulation_parameters["crystal_length"]

    def rate_integrand(qix, qiy, delta_qx, delta_qy):
        qsx = -qix + delta_qx
        qsy = -qiy + delta_qy

        qs_abs = np.sqrt(qsx**2 + qsy**2)
        qi_abs = np.sqrt(qix**2 + qiy**2)

        # The exp(1j * ((qsx * xs_pos + qsy * ys_pos) + (qix * xi_pos + qiy * yi_pos))) portion of the integrand makes it a Fourier transform,
        # avoiding the necessity of actually doing the four-dimensional momentum integral later.
        integrand = np.exp(1j * (ks + ki) * z_pos) * pump_function(qix + qsx, qiy + qsy, kpz, omegap, w0, d) * \
            phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * \
            np.exp(1j * (- qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

        return integrand

    return rate_integrand


def calculate_conditional_probability(xs_pos, ys_pos, xi_pos, yi_pos, thetap, omegai, omegas, simulation_parameters):
    """
    Return the conditional probability of detecting the signal at (xs_pos, ys_pos) given the idler is detected at (xi_pos, yi_pos). Equation 84.

    :param xs_pos: Location of signal photon in the x direction a distance z away from the crystal
    :param ys_pos: Location of signal photon in the y direction a distance z away from the crystal
    :param xi_pos: Location of idler photon in the x direction a distance z away from the crystal
    :param yi_pos: Location of idler photon in the y direction a distance z away from the crystal
    """
    momentum_span_wide = simulation_parameters.get("momentum_span_wide")
    momentum_span_narrow = simulation_parameters.get("momentum_span_narrow")
    num_cores = simulation_parameters.get("num_cores")

    dqix = (omegai / C) * momentum_span_wide
    dqiy = (omegai / C) * momentum_span_wide
    dqx = (omegas / C) * momentum_span_narrow
    dqy = (omegas / C) * momentum_span_narrow

    num_samples_momentum_wide = simulation_parameters["num_samples_momentum_wide"]
    num_samples_momentum_narrow = simulation_parameters["num_samples_momentum_narrow"]

    rate_integrand = get_rate_integrand(thetap, omegai, omegas, simulation_parameters)
    result_grid = grid_integration_momentum(f=rate_integrand, dqix=dqix, dqiy=dqiy,
                                            dqx=dqx, dqy=dqy, num_samples_wide=num_samples_momentum_wide,
                                            num_samples_narrow=num_samples_momentum_narrow, num_cores=num_cores)

    # Select out the signal by index closest to input point
    xi_samples = 1 / np.linspace(-dqix, dqix, num_samples_momentum_wide) # TODO, unclear HOW to label the Fourier transform axes after FT
    index_xi = (np.abs(xi_pos - xi_samples)).argmin()
    yi_samples = 1 / np.linspace(-dqiy, dqiy, num_samples_momentum_wide) # TODO, unclear HOW to label the Fourier transform axes after FT
    index_yi = (np.abs(yi_pos - yi_samples)).argmin()
    print(xi_pos)
    print(xi_samples)

    dx_samples = 1 / np.linspace(-dqx, dqx, num_samples_momentum_narrow)
    dx = xs_pos - xi_pos
    index_dx = (np.abs(dx - dx_samples)).argmin()

    dy_samples = 1 / np.linspace(-dqy, dqy, num_samples_momentum_narrow)
    dy = ys_pos - yi_pos
    index_dy = (np.abs(dy - dy_samples)).argmin()

    result = result_grid[index_xi][index_yi][index_dx][index_dy]

    return result #TODO expand to include type II
    # Also todo, return the actual point used


def calculate_rings(thetap, omegai, omegas, simulation_parameters):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param dx: One half the area of real space centered around the origin along the x direction which the idler (signal) will be integrated over
    :param dy: One half the area of real space centered around the origin along the x direction which the idler (signal) will be integrated over
    """
    momentum_span_wide = simulation_parameters.get("momentum_span_wide")
    momentum_span_narrow = simulation_parameters.get("momentum_span_narrow")
    num_cores = simulation_parameters.get("num_cores")

    dqix = (omegai / C) * momentum_span_wide
    dqiy = (omegai / C) * momentum_span_wide
    dqx = (omegas / C) * momentum_span_narrow
    dqy = (omegas / C) * momentum_span_narrow
    grid_integration_size = simulation_parameters["grid_integration_size"]
    num_samples_momentum_wide = simulation_parameters["num_samples_momentum_wide"]
    num_samples_momentum_narrow = simulation_parameters["num_samples_momentum_narrow"]

    rate_integrand = get_rate_integrand(thetap, omegai, omegas, simulation_parameters)
    result_grid = grid_integration_momentum(f=rate_integrand, dqix=dqix, dqiy=dqiy, dqx=dqx, dqy=dqy,
                                            num_samples_wide=num_samples_momentum_wide,
                                            num_samples_narrow=num_samples_momentum_narrow, num_cores=num_cores)

    # Sum result over two dimensions (integrate) TODO multiply by volume also
    result_grid_sum_dx = np.sum(result_grid, axis=3)
    result_grid_sum_dy = np.sum(result_grid_sum_dx, axis=2)

    return result_grid_sum_dy


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
    signal_x_pos = simulation_parameters["signal_x_pos"] #change to xs_pos TODO
    signal_y_pos = simulation_parameters["signal_y_pos"]
    idler_x_pos = simulation_parameters["idler_x_pos"]
    idler_y_pos = simulation_parameters["idler_y_pos"]
    z_pos = simulation_parameters["z_pos"]
 
    save_directory = simulation_parameters["save_directory"]

    dqix = (omegai / C) * momentum_span
    dqiy = (omegai / C) * momentum_span
    dqsx = (omegas / C) * momentum_span
    dqsy = (omegas / C) * momentum_span

    rate_integrand = get_rate_integrand(thetap, omegai, omegas, simulation_parameters)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    x = np.linspace(-dqix, dqix, num_plot_qx_points)
    y = np.linspace(-dqiy, dqiy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)
    Z = rate_integrand(X, Y, 2*X, 2*Y) #TODO check

    im1 = ax1.imshow(np.abs(Z), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    ax1.set_title("Abs(Integrand)")
    ax1.set_xlabel("$q_x$ ($q_{xi} = q_{xs}$) (Rad/m)")
    ax1.set_ylabel("$q_y$ ($q_{yi} = q_{ys}$) (Rad/m)")
    ax1.tick_params(axis='both', labelsize=4)
    cb1 = fig.colorbar(im1, ax=ax1, location='right', shrink=0.6)
    cb1.ax.tick_params(labelsize=4)
    cb1.ax.yaxis.offsetText.set_fontsize(4)

    im2 = ax2.imshow(np.real(Z), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='jet')
    ax2.set_title("Re(Integrand)")
    ax2.set_xlabel("$q_x$ ($q_{xi} = q_{xs}$) (Rad/m)")
    ax2.set_ylabel("$q_y$ ($q_{yi} = q_{ys}$) (Rad/m)")
    ax2.tick_params(axis='both', labelsize=4)
    cb2 = fig.colorbar(im2, ax=ax2, location='right', shrink=0.6)
    cb2.ax.tick_params(labelsize=4)
    cb2.ax.yaxis.offsetText.set_fontsize(4)

    x = np.linspace(-dqix, dqix, num_plot_qx_points)
    y = np.linspace(-dqiy, dqiy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)
    Z = rate_integrand(X, Y, 0, 0)
    im3 = ax3.imshow(np.abs(Z), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    ax3.set_xlabel("$q_x$ ($q_{xi} = -q_{xs}$) (Rad/m)")
    ax3.set_ylabel("$q_y$ ($q_{yi} = -q_{ys}$) (Rad/m)")
    ax3.tick_params(axis='both', labelsize=4)
    cb3 = fig.colorbar(im3, ax=ax3, location='right', shrink=0.6)
    cb3.ax.tick_params(labelsize=4)
    cb3.ax.yaxis.offsetText.set_fontsize(4)

    im4 = ax4.imshow(np.real(Z), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='jet')
    ax4.set_xlabel("$q_x$ ($q_{xi} = -q_{xs}$) (Rad/m)")
    ax4.set_ylabel("$q_y$ ($q_{yi} = -q_{ys}$) (Rad/m)")
    ax4.tick_params(axis='both', labelsize=4)
    cb4 = fig.colorbar(im4, ax=ax4, location='right', shrink=0.6)
    cb4.ax.tick_params(labelsize=4)
    cb4.ax.yaxis.offsetText.set_fontsize(4)

    plt.tight_layout()

    # Get current time for file name
    time_str = get_current_time()

    plt.savefig(f"{save_directory}/{time_str}_momentum.png", dpi=300)
    plt.close()

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_momentum_params.pkl", "wb") as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_momentum_params.txt", 'w') as file:
        file.write(json.dumps(simulation_parameters))

def simulate_ring_slice(simulation_parameters):
    """
    Simulate and plot a slice of the conditional probabilities of detecting the signal photon, 
    given the idler photon is detected on the x-axis.

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    TODO use kwargs instead of a dict!
    """
    start_time = time.time()

    thetap = simulation_parameters["thetap"] # Incident pump angle, in Radians
    omegap = simulation_parameters["omegap"] # Pump frequency (Radians / sec)
    omegai = simulation_parameters["omegai"] # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"] # Signal frequency (Radians / sec)

    x_signal_span = simulation_parameters["signal_x_span"] # Span in the x-direction to plot conditional probability of signal over, in meters
    signal_y_pos = simulation_parameters["signal_y_pos"]
    idler_x_span = simulation_parameters["idler_x_span"] # Span in the x-direction to fix idler at
    idler_x_increment = simulation_parameters["idler_x_increment"] # Increment size to change idler by in the x-direction
    idler_y_pos = simulation_parameters["idler_y_pos"]

    save_directory = simulation_parameters["save_directory"]
    num_cores = simulation_parameters["simulation_cores"]

    x_signal = np.linspace(-x_signal_span, x_signal_span, num_plot_x_points) #TODO standardize x_signal or signal_x
    sweep_points = np.arange(-idler_x_span, idler_x_span, idler_x_increment)

    calculate_conditional_probability_vec = np.vectorize(calculate_conditional_probability)
    parallel_calc_conditional_prob = functools.partial(calculate_conditional_probability_vec, yi_pos=idler_y_pos,
                                                       thetap=thetap, omegai=omegai, omegas=omegas, simulation_parameters=simulation_parameters)

    # Inefficient; you don't have to call this each time (below)
    z1 = [parallel_calc_conditional_prob(x_s, signal_y_pos, sweep_points) for x_s in x_signal]

    probs = np.array(z1)

    plt.figure(figsize=(8, 6))
    plt.plot(x_signal, probs, label=sweep_points)
    plt.legend()

    plt.title( "Conditional probability of signal given idler at different locations on x-axis" )

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    plt.savefig(f"{save_directory}/{time_str}_rings_slice.png", dpi=300)
    plt.close()

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
    with open(f"{save_directory}/{time_str}_ring_slice_time.txt", 'w') as file:
        file.write(json.dumps(time_info))

def simulate_rings(simulation_parameters):
    """
    Simulate and plot entangled pair rings by integrating the conditional probability of detecting the signal photon given detecting the 
    idler photon, integrating over the possible positions of the idler photon. 

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    """
    start_time = time.time()

    thetap = simulation_parameters["thetap"] # Incident pump angle, in Radians
    omegap = simulation_parameters["omegap"] # Pump frequency (Radians / sec)
    omegai = simulation_parameters["omegai"] # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"] # Signal frequency (Radians / sec)
    grid_integration_size = simulation_parameters["grid_integration_size"] # Size of square root of grid for integration in real space (todo improve name)
    pump_waist_size = simulation_parameters["pump_waist_size"] # Size of pump beam waist
    pump_waist_distance = simulation_parameters["pump_waist_distance"] # Distance of pump waist from crystal (meters)
    z_pos = simulation_parameters["z_pos"] # View location in the z direction, from crystal (meters)
    crystal_length = simulation_parameters["crystal_length"] # Length of the crystal, in meters

    num_cores = simulation_parameters["simulation_cores"] # Number of cores to use in the simulation
    save_directory = simulation_parameters["save_directory"]

    # Run calculate_pair_generation_rate in parallel
    Z1 = calculate_rings(thetap=thetap, omegai=omegai, omegas=omegas,
                         simulation_parameters=simulation_parameters)

    # Next, fix idler and integrate over signal
    # TODO, only relevant for type II

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    # Plot results
    plt.figure(figsize=(8, 6))
#    plt.imshow(np.abs(Z), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    plt.imshow(np.abs(Z1), origin='lower', cmap='gray')

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title( "BBO crystal entangled photons rates" )
    plt.savefig(f"{save_directory}/{time_str}_rings.png", dpi=300)    
    plt.close()

    # Save data to a pickled file
    with open(f"{save_directory}/{time_str}_rings.pkl", "wb") as file:
        pickle.dump(Z1, file)

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


# todo, type 2 noncollinear

# Plot total output power as a function of theta_p and other params
# TODO
