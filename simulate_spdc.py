import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import functools
import itertools
import pickle
import json
from file_utils import get_current_time
from memory_profiler import profile
import gc

# Constants
C = 2.99792e8  # Speed of light, in meters per second

@profile
def grid_integration_momentum(f, dqix, dqiy, dqx, dqy, num_samples_wide_x, num_samples_wide_y, num_samples_narrow_x,
                              num_samples_narrow_y, num_cores):
    """
    Integrate along momentum dimensions.
    """
    qix_array = np.linspace(-dqix, dqix, num_samples_wide_x, dtype=np.float32)
    qiy_array = np.linspace(-dqiy, dqiy, num_samples_wide_y, dtype=np.float32)

    # From conservation of momentum, dqix should be close to -dqsx and dqiy should be close to -dqsy
    # so we can write dqix = -dqsx + dqx and dqiy = -dqsy + dqy 
    # and then can use fewer points of integration along the dqx, dqy dimensions.
    dqx_array = np.linspace(-dqx, dqx, num_samples_narrow_x, dtype=np.float32)
    dqy_array = np.linspace(-dqy, dqy, num_samples_narrow_y, dtype=np.float32)

    # Generate the coordinate grid using meshgrid
    qix_grid, qiy_grid, dqx_grid, dqy_grid = np.meshgrid(qix_array, qiy_array, dqx_array, dqy_array,
                                                           indexing='ij')

    # Flatten the grids
    qix_flat = qix_grid.ravel()
    qiy_flat = qiy_grid.ravel()
    dqx_flat = dqx_grid.ravel()
    dqy_flat = dqy_grid.ravel()
    time1 = time.time()

    # Vectorized evaluation of the function f over the flattened grids
#    result_grid = f(qix_flat, qiy_flat, dqx_flat, dqy_flat)

    # Comment this out for debug
    qix_jobs = np.array_split(qix_flat, num_cores)
    qiy_jobs = np.array_split(qiy_flat, num_cores)
    dqx_jobs = np.array_split(dqx_flat, num_cores)
    dqy_jobs = np.array_split(dqy_flat, num_cores)

    result_grids = Parallel(n_jobs=num_cores)(delayed(f)(qix_jobs[i], qiy_jobs[i], dqx_jobs[i], dqy_jobs[i]) for i in range(num_cores))

    # Manually clean up large objects
    del qix_grid, qiy_grid, dqx_grid, dqy_grid, qix_flat, qiy_flat, dqx_flat, dqy_flat, qix_jobs, qiy_jobs, dqx_jobs, dqy_jobs

    result_grid = np.concatenate(result_grids)

    del result_grids

    time2 = time.time()
    print(time2-time1)

    # Reshape grid
    reshaped_result_grid = np.reshape(result_grid, [len(qix_array), len(qiy_array), len(dqx_array), len(dqy_array)])

    del result_grid

    # N-dimensional Fourier transform across all four axes
    ft_result_grid = np.fft.fftn(reshaped_result_grid)
    del reshaped_result_grid

    ft_result_grid_shifted = np.fft.fftshift(ft_result_grid)

    # Manually clean up large objects
    del ft_result_grid

    # Return the absolute value of this grid squared
    time3 = time.time()
    print(time3-time2)

    # Find the four Fourier transformed axes
    def get_fourier_transformed_axis(q_increment, num_points):
        if num_points % 2 == 0:
            ft_axis = np.arange(-num_points/2, num_points/2-1)*1/(q_increment*num_points)
        else:
            ft_axis = np.arange(-(num_points-1)/2, (num_points-1)/2)*1/(q_increment*num_points)
        return ft_axis
        #2pi factor? todo
    xi_array = get_fourier_transformed_axis(q_increment=(2*dqix)/(2*np.pi*num_samples_wide_x), num_points=num_samples_wide_x)
    yi_array = get_fourier_transformed_axis(q_increment=(2*dqiy)/(2*np.pi*num_samples_wide_y), num_points=num_samples_wide_y)
    dx_array = get_fourier_transformed_axis(q_increment=(2*dqx)/(2*np.pi*num_samples_narrow_x), num_points=num_samples_narrow_x)
    dy_array = get_fourier_transformed_axis(q_increment=(2*dqy)/(2*np.pi*num_samples_narrow_y), num_points=num_samples_narrow_y)

    result_squared = np.abs(ft_result_grid_shifted)**2
    del ft_result_grid_shifted
    gc.collect()

    # Return the absolute value of the Fourier transformed grid squared, as well as the four new axes
    return result_squared, xi_array, yi_array, dx_array, dy_array

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

def alpha(thetap, lambd):
    """ Return the alpha coefficient as a function of pump incidence tilt angle `thetap` and 
    wavelength `lambd`.

    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param lambd: Wavelength, in meters
    """
    alpha = ((n_o(lambd)**2 - n_e(lambd)**2) * np.sin(thetap) * np.cos(thetap)) / \
    (n_o(lambd)**2 * np.sin(thetap)**2 + n_e(lambd)**2 * np.cos(thetap)**2)
    return alpha

def beta(thetap, lambd):
    """ Return the `beta` coefficient as a function of pump incidence tilt angle `thetap` and 
    pump wavelength `lambd`.

    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param lambd: Wavelength, in meters
    """
    beta = (n_o(lambd) * n_e(lambd)) / \
    (n_o(lambd)**2 * np.sin(thetap)**2 + n_e(lambd)**2 * np.cos(thetap)**2)
    return beta

def gamma(thetap, lambd):
    """ Return the gamma coefficient as a function of pump incidence tilt angle `thetap` and 
    pump wavelength `lambd`.

    :param thetap: Angle `thetap` along which pump beam enters BBO crystal (about y-axis)
    :param lambd: Wavelength, in meters
    """
    gamma = n_o(lambd) / \
    np.sqrt((n_o(lambd)**2 * np.sin(thetap)**2 + n_e(lambd)**2 * np.cos(thetap)**2))
    return gamma

def eta(thetap, lambd):
    """
    Return the eta coefficient as a function of pump incidence tilt angle `thetap` and 
    wavelength `lambd`.
    
    :param thetap: Angle `thetap` along which pump beam enters BBO crystal (about y-axis)
    :param lambd: Wavelength, in meters
    """
    eta = (n_o(lambd) * n_e(lambd)) / \
    np.sqrt((n_o(lambd)**2 * np.sin(thetap)**2 + n_e(lambd)**2 * np.cos(thetap)**2))
    return eta

def delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegas, omegai):
    """
    Return delta_k for type I phase matching, for BBO crystal.
    
    :param qsx: k-vector in the x direction for signal
    :param qix: k-vector in the x direction for idler
    :param qsy: k-vector in the y direction for signal
    :param qiy: k-vector in the y direction for idler
    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param omegap: Angular frequency of the pump beam (TODO, overdefined)
    :param omegas: Angular frequency of the signal beam
    :param omegai: Angular frequency of the idler beam
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

def delta_k_type_2(q1x, q2x, q1y, q2y, thetap, omegap, omega1, omega2):
    """ Return delta_k for type II, case 1 phase matching, for BBO crystal.
    
    :param qsx: k-vector in the x direction for signal
    :param qix: k-vector in the x direction for idler
    :param qsy: k-vector in the y direction for signal
    :param qiy: k-vector in the y direction for idler
    :param thetap: Angle theta along which pump beam enters BBO crystal (about y-axis)
    :param omegap: Angular frequency of the pump beam (TODO, overdefined)
    :param omegai: Angular frequency of the idler beam
    :param omegas: Angular frequency of the signal beam
    """
    lambda1 = (2 * np.pi * C) / omega1
    lambda2 = (2 * np.pi * C) / omega2
    lambdap = (2 * np.pi * C) / omegap

    qpx = q1x + q2x # Conservation of momentum
    qpy = q1y + q2y # Conservation of momentum
    q1_abs = np.sqrt(q1x**2 + q1y**2)
    q2_abs = np.sqrt(q2x**2 + q2y**2)

    delta_k = -alpha(thetap, lambda1)*q1x + eta(thetap, lambda1)*omega1/C - \
        C/(2*eta(thetap, lambda1)*omega1)*(beta(thetap, lambda1)**2 * q1x**2 + gamma(thetap, lambda1)**2 * q1y**2) + \
        n_o(lambda2) * omega2 / C - C * q2_abs**2 / (2*n_o(lambda2) * omega2) + \
        alpha(thetap, lambdap) * (q1x + q2x) - eta(thetap, lambdap) * omegap / C + \
        C / (2 * eta(thetap, lambdap) * omegap) * \
        (beta(thetap, lambdap)**2 * (q1x + q2x)**2 + gamma(thetap, lambdap)**2 * (q1y + q2y)**2)

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

def get_rate_integrand(thetap, omegai, omegas, simulation_parameters, phase_matching_case):
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

        # Calculate delta_k based on the type of phase-matching
        if phase_matching_case == "1": # TODO change to Enum
            delta_k_term = delta_k_type_1(qsx=qsx, qix=qix, qsy=qsy, qiy=qiy, thetap=thetap,
                                          omegap=omegap, omegai=omegai, omegas=omegas)
        elif phase_matching_case == "2s":
            delta_k_term = delta_k_type_2(q1x=qsx, q2x=qix, q1y=qsy, q2y=qiy, thetap=thetap,
                                          omegap=omegap, omega1=omegas, omega2=omegai)
        elif phase_matching_case == "2i":
            delta_k_term = delta_k_type_2(q1x=qix, q2x=qsx, q1y=qiy, q2y=qsy, thetap=thetap,
                                          omegap=omegap, omega1=omegai, omega2=omegas)
        else:
            raise TypeError(f"Error, unknown phase matching case {phase_matching_case}.")

        # The exp(1j * ((qsx * xs_pos + qsy * ys_pos) + (qix * xi_pos + qiy * yi_pos))) portion of the integrand makes it a Fourier transform,
        # avoiding the necessity of actually doing the four-dimensional momentum integral later.
        integrand = np.exp(1j * (ks + ki) * z_pos) * pump_function(qix + qsx, qiy + qsy, kpz, omegap, w0, d) * \
            phase_matching(delta_k_term, crystal_length) * \
            np.exp(1j * (-qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

        return integrand

    return rate_integrand


def calculate_conditional_probability(xi_pos, yi_pos, thetap, omegai, omegas, simulation_parameters):
    """
    Return the conditional probability of detecting the idler at any x pos and at yi_pos given the signal is detected
    at (xs_pos, ys_pos). Equation 84.

    :param ys_pos: Location of signal photon in the y direction a distance z away from the crystal
    :param xi_pos: Location of idler photon in the x direction a distance z away from the crystal
    :param yi_pos: Location of idler photon in the y direction a distance z away from the crystal
    TODO reduce duplicate code with calculate_rings
    """
    momentum_span_wide = simulation_parameters.get("momentum_span_wide")
    momentum_span_narrow = simulation_parameters.get("momentum_span_narrow")
    num_samples_momentum_wide = simulation_parameters["num_samples_momentum_wide"]
    num_samples_momentum_narrow = simulation_parameters["num_samples_momentum_narrow"]
    num_cores = simulation_parameters.get("simulation_cores")
    phase_matching_type = simulation_parameters.get("phase_matching_type")

    dqix = (omegai / C) * momentum_span_wide
    dqiy = (omegai / C) * momentum_span_wide
    dqx = (omegas / C) * momentum_span_narrow
    dqy = (omegas / C) * momentum_span_narrow

    def get_integral_grid(phase_matching_case):
        rate_integrand = get_rate_integrand(thetap, omegai, omegas, simulation_parameters, phase_matching_case)
        # TODO make below more general? wrt 1 and 2 not s and i
        result_grid, xis, yis, dxs, dys = grid_integration_momentum(f=rate_integrand, dqix=dqix, dqiy=dqiy, dqx=dqx, dqy=dqy,
                                                num_samples_wide=num_samples_momentum_wide,
                                                num_samples_narrow=num_samples_momentum_narrow, num_cores=num_cores)

        # Choose conditional probability grid of one photon
        xi_pos = 2e-3 #TODO pass in
        yi_pos = 0 # TODO pass in

        index_xi = (np.abs(xi_pos - xis)).argmin()
        index_yi = (np.abs(yi_pos - yis)).argmin()

        result_grid_probability = result_grid[index_xi, index_yi, :, :]

        # # Transform axes
        xss = dxs - xi_pos
        yss = dys - yi_pos

        return result_grid_probability, xss, yss

    if phase_matching_type == 1:
        result, xs, ys = get_integral_grid(phase_matching_case="1")
    elif phase_matching_type == 2:
        result1, _, _ = get_integral_grid(phase_matching_case="2s") #TODO ENUMS!
        result2, xs, ys = get_integral_grid(phase_matching_case="2i")
        result = result1 + result2
    else:
        raise ValueError(f"Unknown phase_matching_type {phase_matching_type}.")

    return result, xs, ys
    #TODO expand to include type II
    # Also could return signal probability
    # Also todo, return the actual points used

def calculate_rings(thetap, omegai, omegas, simulation_parameters):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param dx: One half the area of real space centered around the origin along the x direction which the idler (signal) will be integrated over
    :param dy: One half the area of real space centered around the origin along the x direction which the idler (signal) will be integrated over
    """
    momentum_span_wide_x = simulation_parameters.get("momentum_span_wide_x")
    momentum_span_wide_y = simulation_parameters.get("momentum_span_wide_y")

    momentum_span_narrow_x = simulation_parameters.get("momentum_span_narrow_x")
    momentum_span_narrow_y = simulation_parameters.get("momentum_span_narrow_y")

    num_samples_momentum_wide_x = simulation_parameters["num_samples_momentum_wide_x"]
    num_samples_momentum_wide_y = simulation_parameters["num_samples_momentum_wide_y"]

    num_samples_momentum_narrow_x = simulation_parameters["num_samples_momentum_narrow_x"]
    num_samples_momentum_narrow_y = simulation_parameters["num_samples_momentum_narrow_y"]

    num_cores = simulation_parameters.get("simulation_cores") #get rid of .get
    phase_matching_type = simulation_parameters.get("phase_matching_type")

    dqix = (omegai / C) * momentum_span_wide_x
    dqiy = (omegai / C) * momentum_span_wide_y
    dqx = (omegas / C) * momentum_span_narrow_x
    dqy = (omegas / C) * momentum_span_narrow_y

    def get_integral_grid(phase_matching_case):
        rate_integrand = get_rate_integrand(thetap, omegai, omegas, simulation_parameters, phase_matching_case)
        # TODO make below more general? wrt 1 and 2 not s and i
        result_grid, xis, yis, dxs, dys = grid_integration_momentum(f=rate_integrand, dqix=dqix, dqiy=dqiy, dqx=dqx, dqy=dqy,
                                                num_samples_wide_x=num_samples_momentum_wide_x,
                                                num_samples_wide_y=num_samples_momentum_wide_y,
                                                num_samples_narrow_x=num_samples_momentum_narrow_x, 
                                                num_samples_narrow_y=num_samples_momentum_narrow_y, 
                                                num_cores=num_cores)

        # Sum result over two dimensions (integrate) TODO multiply by volume also
        result_grid_sum_over_dx = np.sum(result_grid, axis=3)
        return np.sum(result_grid_sum_over_dx, axis=2), xis, yis

    if phase_matching_type == 1:
        result, xs, ys = get_integral_grid(phase_matching_case="1")
    elif phase_matching_type == 2:
        result1, _, _ = get_integral_grid(phase_matching_case="2s") #TODO ENUMS!
        result2, xs, ys = get_integral_grid(phase_matching_case="2i")
        result = result1 + result2
    else:
        raise ValueError(f"Unknown phase_matching_type {phase_matching_type}.")

    return result, xs, ys

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
    phase_matching_type = simulation_parameters["phase_matching_type"] # Type I or Type II phase-matching

    momentum_span_x = simulation_parameters["momentum_span_x"]
    momentum_span_y = simulation_parameters["momentum_span_y"]
    signal_x_pos = simulation_parameters["signal_x_pos"] #change to xs_pos TODO
    signal_y_pos = simulation_parameters["signal_y_pos"]
    idler_x_pos = simulation_parameters["idler_x_pos"]
    idler_y_pos = simulation_parameters["idler_y_pos"]
    z_pos = simulation_parameters["z_pos"]
 
    save_directory = simulation_parameters["save_directory"]

    dqix = (omegai / C) * momentum_span_x
    dqiy = (omegai / C) * momentum_span_y
    dqsx = (omegas / C) * momentum_span_x
    dqsy = (omegas / C) * momentum_span_y

    x = np.linspace(-dqix, dqix, num_plot_qx_points)
    y = np.linspace(-dqiy, dqiy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)

    if phase_matching_type == 1:
        rate_integrand = get_rate_integrand(thetap, omegai, omegas, simulation_parameters, "1")
        Z1 = rate_integrand(X, Y, 2*X, 2*Y)
        Z2 = rate_integrand(X, Y, 0, 0)
    elif phase_matching_type == 2:
        rate_integrand_2s = get_rate_integrand(thetap, omegai, omegas, simulation_parameters, "2s") #TODO ENUMS!
        rate_integrand_2i = get_rate_integrand(thetap, omegai, omegas, simulation_parameters, "2i")
        Z1 = rate_integrand_2s(X, Y, 2*X, 2*Y) + rate_integrand_2i(X, Y, 2*X, 2*Y)
        Z2 = rate_integrand_2s(X, Y, 0, 0) + rate_integrand_2i(X, Y, 0, 0)
    else:
        raise ValueError(f"Unknown phase_matching_type {phase_matching_type}.")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    im1 = ax1.imshow(np.abs(Z1), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    ax1.set_title("Abs(Integrand)")
    ax1.set_xlabel("$q_x$ ($q_{xi} = q_{xs}$) (Rad/m)")
    ax1.set_ylabel("$q_y$ ($q_{yi} = q_{ys}$) (Rad/m)")
    ax1.tick_params(axis='both', labelsize=4)
    cb1 = fig.colorbar(im1, ax=ax1, location='right', shrink=0.6)
    cb1.ax.tick_params(labelsize=4)
    cb1.ax.yaxis.offsetText.set_fontsize(4)

    im2 = ax2.imshow(np.real(Z1), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='jet')
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
    im3 = ax3.imshow(np.abs(Z2), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    ax3.set_xlabel("$q_x$ ($q_{xi} = -q_{xs}$) (Rad/m)")
    ax3.set_ylabel("$q_y$ ($q_{yi} = -q_{ys}$) (Rad/m)")
    ax3.tick_params(axis='both', labelsize=4)
    cb3 = fig.colorbar(im3, ax=ax3, location='right', shrink=0.6)
    cb3.ax.tick_params(labelsize=4)
    cb3.ax.yaxis.offsetText.set_fontsize(4)

    im4 = ax4.imshow(np.real(Z2), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='jet')
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

def simulate_conditional_probability(simulation_parameters):
    """
    Simulate and plot a slice of the conditional probabilities of detecting the idler photon, 
    given the signal photon is detected on the x-axis.

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    TODO use kwargs instead of a dict!
    """
    start_time = time.time()

    thetap = simulation_parameters["thetap"] # Incident pump angle, in Radians
    omegap = simulation_parameters["omegap"] # Pump frequency (Radians / sec)
    omegai = simulation_parameters["omegai"] # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"] # Signal frequency (Radians / sec)
    phase_matching_type = simulation_parameters["phase_matching_type"]

    num_plot_x_points = simulation_parameters["num_plot_x_points"]
    idler_x_span  = simulation_parameters["idler_x_span"]
    idler_y_pos = simulation_parameters["idler_y_pos"]
    signal_x_pos = simulation_parameters["signal_x_pos"] # Span in the x-direction to fix signal at
    signal_y_pos = simulation_parameters["signal_y_pos"]

    save_directory = simulation_parameters["save_directory"]
    num_cores = simulation_parameters["simulation_cores"]

    # x_idler = np.linspace(-idler_x_span, idler_x_span, num_plot_x_points)

    # calculate_conditional_probability_vec = np.vectorize(calculate_conditional_probability)
    # parallel_calc_conditional_prob = functools.partial(calculate_conditional_probability_vec, yi_pos=idler_y_pos, xs_pos=signal_x_pos, ys_pos=signal_y_pos,
    #                                                    thetap=thetap, omegai=omegai, omegas=omegas, simulation_parameters=simulation_parameters)

    # # Inefficient; you don't have to call this each time (below)
    # z1 = [parallel_calc_conditional_prob(x_i) for x_i in x_idler] # You should also return the actual point that got plotted

    # Run calculate_pair_generation_rate
  ##  xs_pos, ys_pos, xi_pos, yi_pos, thetap, omegai, omegas, simulation_parameters
    Z1, xis, yis = calculate_conditional_probability(xi_pos=0, yi_pos=idler_y_pos,
                                                  thetap=thetap, omegai=omegai, omegas=omegas,
                                                  simulation_parameters=simulation_parameters)

    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(Z1), extent=(xis.min(), xis.max(), yis.min(), yis.max()), origin='lower', cmap='gray')

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title( "Conditional probability of signal given idler at TODO" )

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    plt.savefig(f"{save_directory}/{time_str}_rings_conditional_probability.png", dpi=300)
    plt.close()

    # Save parameters and data
    with open(f"{save_directory}/{time_str}_ring_conditional_probability.pkl", "wb") as file:
        pickle.dump(Z1, file)

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_ring_conditional_probability_params.pkl", "wb") as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_ring_conditional_probability_params.txt", 'w') as file:
        file.write(json.dumps(simulation_parameters))

    # Save time to a text file
    time_info = {"Time Elapsed in seconds" : end_time - start_time}
    with open(f"{save_directory}/{time_str}_ring_conditional_probability_time.txt", 'w') as file:
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
    pump_waist_size = simulation_parameters["pump_waist_size"] # Size of pump beam waist
    pump_waist_distance = simulation_parameters["pump_waist_distance"] # Distance of pump waist from crystal (meters)
    z_pos = simulation_parameters["z_pos"] # View location in the z direction, from crystal (meters)
    crystal_length = simulation_parameters["crystal_length"] # Length of the crystal, in meters
    phase_matching_type = simulation_parameters["phase_matching_type"]

    num_cores = simulation_parameters["simulation_cores"] # Number of cores to use in the simulation
    save_directory = simulation_parameters["save_directory"]

    # Run calculate_pair_generation_rate in parallel
    Z1, xis, yis = calculate_rings(thetap=thetap, omegai=omegai, omegas=omegas,
                         simulation_parameters=simulation_parameters)

    # Next, fix idler and integrate over signal
    # TODO, only relevant for type II

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(Z1), extent=(xis.min(), xis.max(), yis.min(), yis.max()), origin='lower', cmap='gray')

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



# Plot total output power as a function of theta_p and other params
# TODO
