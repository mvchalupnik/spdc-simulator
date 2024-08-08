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

def monte_carlo_integration_momentum(f, dqsx, dqsy, dqix, dqiy, num_samples):
    """
    Integrate function `f` using the Monte Carlo method along four dimensions of momentum,
    qx, qy for both the signal and idler photons.
    """
  #  np.random.seed(3) #this can smooth the result
    ## Generate random samples within the bounds [-dq, dq] for each variable
#    MAX_BATCH_SIZE = 200000
    MAX_BATCH_SIZE = 20000000

    batch_size = np.min([MAX_BATCH_SIZE, num_samples])

    mytime = time.time()

    average = None
    # TODO clean up and simplify
    for j in range(num_samples // batch_size):
        qix_samples = np.random.uniform(-dqix, dqix, batch_size)
        qiy_samples = 0
#        qiy_samples = np.random.uniform(-dqiy, dqiy, batch_size)
        qsx_samples = np.random.uniform(-dqsx, dqsx, batch_size)
#        qsy_samples = np.random.uniform(-dqsy, dqsy, batch_size)
        qsy_samples = 0

        func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples)
        average = np.mean(func_values) if j == 0 else np.mean([average, np.mean(func_values)]) # Make this into a list for parallelization


    batch_remainder = num_samples % batch_size
    if batch_remainder != 0:
        qix_samples = np.random.uniform(-dqix, dqix, batch_remainder)
        qiy_samples = 0
    #    qiy_samples = np.random.uniform(-dqiy, dqiy, batch_remainder)
        qsx_samples = np.random.uniform(-dqsx, dqsx, batch_remainder)
    #    qsy_samples = np.random.uniform(-dqsy, dqsy, batch_remainder)
        qsy_samples = 0

        func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples)
        average = np.mean([average, np.mean(func_values)])
    mytime2 = time.time()

    # Evaluate the function at each sample point
 #   print(f"Time increment1: {mytime2 - mytime}")
#    mytime3 = time.time()
#    func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples)
#    mytime4 = time.time()
#    print(f"Time increment2: {mytime4 - mytime3}")

#    # Calculate the average value of the function
#    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dqix) * (2 * dqiy) * (2 * dqsx) * (2 * dqsy)
    
    # Estimate the integral as the average value times the volume
    integral_estimate = average * volume

    # Square the integral at the end
    integral_estimate_sq = np.abs(integral_estimate)**2

    return integral_estimate_sq

# def fdsfdsmonte_carlo_integration_momentum(f, dqsx, dqsy, dqix, dqiy, num_samples):
#     """
#     Integrate function `f` using the Monte Carlo method along four dimensions of momentum,
#     qx, qy for both the signal and idler photons.
#     """
#   #  np.random.seed(3) #this can smooth the result
#     ## Generate random samples within the bounds [-dq, dq] for each variable
#     mytime = time.time()
#     qix_samples = np.random.uniform(-dqix, dqix, num_samples)
#     mytime2 = time.time()
#     qiy_samples = 0
# #    qiy_samples = np.random.uniform(-dqiy, dqiy, num_samples)
#     qsx_samples = np.random.uniform(-dqsx, dqsx, num_samples)
# #    qsy_samples = np.random.uniform(-dqsy, dqsy, num_samples)
#     qsy_samples = 0

#     # Evaluate the function at each sample point
#     print(f"Time increment1: {mytime2 - mytime}")
#     mytime3 = time.time()
#     func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples)
#     mytime4 = time.time()
#     print(f"Time increment2: {mytime4 - mytime3}")

#     # Calculate the average value of the function
#     avg_value = np.mean(func_values)
    
#     # The volume of the integration region
#     volume = (2 * dqix) * (2 * dqiy) * (2 * dqsx) * (2 * dqsy)
    
#     # Estimate the integral as the average value times the volume
#     integral_estimate = avg_value * volume

#     # Square the integral at the end
#     integral_estimate_sq = np.abs(integral_estimate)**2

#     return integral_estimate_sq

def grid_integration_position(f, dqsx, dqsy, dqix, dqiy, dx, dy, num_samples_position, num_samples_momentum):
    """
    Integrate along x and y. First pass function to be integrated along four dimensions
    of momentum (signal qx and qy, idler qx and qy).
    """
    # Generate from a grid samples within the bounds [-dx, dx] and [-dy, dy] for each variable
    x_samples = np.linspace(-dx, dx, num_samples_position)
    y_samples = np.linspace(-dy, dy, num_samples_position)
    coord_pairs = list(itertools.product(x_samples, y_samples))

    # Evaluate the function at each sample point
    func_values = np.zeros(len(coord_pairs), dtype='complex128') # Technically won't be complex here
    for n in range(len(coord_pairs)):
        x_sample, y_sample = coord_pairs[n]
        g = functools.partial(f, x_pos_integrate=x_sample, y_pos_integrate=y_sample)
        func_values[n] = monte_carlo_integration_momentum(g, dqsx, dqsy, dqix, dqiy, num_samples_momentum)
#        func_values[n] = grid_integration_momentum(g, dqsx, dqsy, dqix, dqiy, num_samples_momentum)

    # Calculate the average value of the function
    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dx) * (2 * dy)
    
    # Estimate the integral as the average value times the volume
    integral_estimate = avg_value * volume
    
    return integral_estimate

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
            raise ValueError(f"Expecting to integrate over signal or idler and received {integrate_over}.")

        qs_dot_rhos = (qsx * xs_pos + qsy * ys_pos)
        qi_dot_rhoi = (qix * xi_pos + qiy * yi_pos)
        qs_abs = np.sqrt(qsx**2 + qsy**2)
        qi_abs = np.sqrt(qix**2 + qiy**2)

        integrand = np.exp(1j * (ks + ki) * z_pos) * pump_function(qix + qsx, qiy + qsy, kpz, omegap, w0, d) * \
            phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * \
            np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

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
    momentum_span = simulation_parameters.get("momentum_span")
    num_samples = simulation_parameters["num_momentum_integration_points"]
    dqix = (omegai / C) * momentum_span
    dqiy = (omegai / C) * momentum_span
    dqsx = (omegas / C) * momentum_span
    dqsy = (omegas / C) * momentum_span

    rate_integrand = get_rate_integrand(xs_pos, ys_pos, thetap, omegai, omegas, simulation_parameters)
    rate_integrand_signal = functools.partial(rate_integrand, integrate_over="idler", x_pos_integrate=xi_pos, y_pos_integrate=yi_pos)
    result_signal = monte_carlo_integration_momentum(f=rate_integrand_signal, dqsx=dqsx, dqsy=dqsy, dqix=dqix, dqiy=dqiy, num_samples=num_samples)
#    result_signal = grid_integration_momentum(f=rate_integrand_signal, dqsx=dqsx, dqsy=dqsy, dqix=dqix, dqiy=dqiy, num_samples=num_samples)

    # TODO right now result_idler will equal result_signal; I think this  is always true but need to check, also given different omegas, omegai
    # rate_integrand_idler = functools.partial(rate_integrand, integrate_over="signal", x_pos_integrate=xs_pos, y_pos_integrate=ys_pos)
    # result_idler = monte_carlo_integration_momentum(f=rate_integrand_idler, dqsx=dqsx, dqsy=dqsy, dqix=dqix, dqiy=dqiy, num_samples=num_samples)

    return result_signal #TODO expand to include type II and also return idler


def calculate_pair_generation_rate(x_pos, y_pos, thetap, omegai, omegas, dx, dy, integrate_over, simulation_parameters):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param x_pos: Location of signal (idler) photon in the x direction a distance z away from the crystal
    :param y_pos: Location of signal (idler) photon in the y direction a distance z away from the crystal
    :param dx: One half the area of real space centered around the origin along the x direction which the idler (signal) will be integrated over
    :param dy: One half the area of real space centered around the origin along the x direction which the idler (signal) will be integrated over
    :param integrate_over: If "signal", integrate over signal. If "idler", integrate over idler.
    """
    momentum_span = simulation_parameters.get("momentum_span")
    dqix = (omegai / C) * momentum_span
    dqiy = (omegai / C) * momentum_span
    dqsx = (omegas / C) * momentum_span
    dqsy = (omegas / C) * momentum_span
    num_samples_monte_carlo = simulation_parameters["num_momentum_integration_points"]
    grid_integration_size = simulation_parameters["grid_integration_size"]

    rate_integrand = get_rate_integrand(x_pos, y_pos, thetap, omegai, omegas, simulation_parameters)
    rate_integrand_wrt = functools.partial(rate_integrand, integrate_over=integrate_over)
    result = grid_integration_position(f=rate_integrand_wrt, dqsx=dqsx, dqsy=dqsy, dqix=dqix, dqiy=dqiy, dx=dx, dy=dy,
                                       num_samples_position=grid_integration_size, num_samples_momentum=num_samples_monte_carlo)

    return result


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

    rate_integrand = get_rate_integrand(signal_x_pos, signal_y_pos, thetap, omegai, omegas, simulation_parameters)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    x = np.linspace(-dqix, dqix, num_plot_qx_points)
    y = np.linspace(-dqiy, dqiy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)
    Z = rate_integrand(X, Y, X, Y, idler_x_pos, idler_y_pos, "signal")

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
    Z = rate_integrand(X, Y, -X, -Y, idler_x_pos, idler_y_pos, "signal")
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

    num_plot_x_points = simulation_parameters["num_plot_x_points"]
    x_signal_span = simulation_parameters["signal_x_span"] # Span in the x-direction to plot conditional probability of signal over, in meters
    signal_y_pos = simulation_parameters["signal_y_pos"]
    idler_x_span = simulation_parameters["idler_x_span"] # Span in the x-direction to fix idler at
    idler_x_increment = simulation_parameters["idler_x_increment"] # Increment size to change idler by in the x-direction
    idler_y_pos = simulation_parameters["idler_y_pos"]

    save_directory = simulation_parameters["save_directory"]
    seed = simulation_parameters["random_seed"]
    num_cores = simulation_parameters["simulation_cores"]

    # Seed for reproducibility
    np.random.seed(seed)

    x_signal = np.linspace(-x_signal_span, x_signal_span, num_plot_x_points) #TODO standardize x_signal or signal_x
    sweep_points = np.arange(-idler_x_span, idler_x_span, idler_x_increment)

    calculate_conditional_probability_vec = np.vectorize(calculate_conditional_probability)
    parallel_calc_conditional_prob = functools.partial(calculate_conditional_probability_vec, yi_pos=idler_y_pos,
                                                       thetap=thetap, omegai=omegai, omegas=omegas, simulation_parameters=simulation_parameters)

    z1 = Parallel(n_jobs=num_cores)(delayed(parallel_calc_conditional_prob)(x_signal, signal_y_pos, idler_x_pos) for idler_x_pos in sweep_points)
    probs = np.transpose(np.array(z1))

    plt.figure(figsize=(8, 6))
    plt.plot(x_signal, probs)

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

    # Seed for reproducibility
    seed = simulation_parameters["random_seed"]
    np.random.seed(seed)

    num_plot_x_points = simulation_parameters["num_plot_x_points"]
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

    # First, fix signal and integrate over idler:
    # Create a grid of x and y values
    x = np.linspace(-x_span, x_span, num_plot_x_points)
    y = np.linspace(-y_span, y_span, num_plot_y_points)

    calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)

    # Run calculate_pair_generation_rate in parallel
    parallel_calculate_pair_generation_rate = functools.partial(calculate_pair_generation_rate_vec, thetap=thetap, omegai=omegai, omegas=omegas,
                                                                dx=x_span, dy=y_span,
                                                                integrate_over="idler", simulation_parameters=simulation_parameters)
    Z1 = Parallel(n_jobs=num_cores)(delayed(parallel_calculate_pair_generation_rate)(xs, ys) for xs in x for ys in y)
    Z = np.reshape(np.array(Z1), [num_plot_y_points, num_plot_x_points]).T

    # Next, fix idler and integrate over signal
    # TODO, only relevant for type II

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
    plt.close()

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


# todo, type 2 noncollinear

# Plot total output power as a function of theta_p and other params
# TODO

