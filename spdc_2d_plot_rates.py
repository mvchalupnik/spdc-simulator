import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy
from joblib import Parallel, delayed
from scipy.stats import qmc
import functools
from scipy.stats import norm

# Constants
crystal_length = 0.002    # Length of the nonlinear crystal in meters
C = 2.99792e8 # Speed of light, in meters per second
np.random.seed(0)

# def monte_carlo_integration(f, s, num_samples=100000):

#     np.random.seed(8)

#     # Generate random samples from the importance distribution g
#     samples = np.random.normal(scale=s, size=(num_samples, 4))
#     x1_samples, y1_samples, x2_samples, y2_samples = samples.T

#     def g(x1, x2, y1, y2):
#         np.exp(-((x1**2 + y1**2 + x2**2 + y2**2) / (2 * s**2)))

#     # Evaluate the function and the importance distribution at the sample points
#     f_values = f(x1_samples, y1_samples, x2_samples, y2_samples)
#     g_values = g(x1_samples, y1_samples, x2_samples, y2_samples)

#     # Compute the weighted average
#     import pdb; pdb.set_trace()
#     avg_value_real = np.mean(np.real(f_values) / np.real(g_values))
#     avg_value_imag = np.mean(np.imag(f_values) / np.imag(g_values))
#     avg_value = avg_value_real + 1j * avg_value_imag

#     # The volume of the integration region
#     volume = (2 * s)**4

#     # Estimate the integral as the weighted average times the volume
#     integral_estimate = avg_value * volume

#     return integral_estimate

def monte_carlo_integration_momentum(f, dq, num_samples=300000):
    ## Generate random samples within the bounds [-dq, dq] for each variable
    qix_samples = np.random.uniform(-dq, dq, num_samples)
    qiy_samples = np.random.uniform(-dq, dq, num_samples)
    qsx_samples = np.random.uniform(-dq, dq, num_samples)
    qsy_samples = np.random.uniform(-dq, dq, num_samples)


    # Evaluate the function at each sample point
    func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples)

    # Square the absolute value of the result
    func_values_sq = np.abs(func_values)**2 

    # Calculate the average value of the function
    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dq)**4
    
    # Estimate the integral as the average value times the volume
    integral_estimate = avg_value * volume
    
    return integral_estimate


# def monte_carlo_integration_position(f, dq, dr, num_samples=1):
#   #  np.random.seed(0) #Use the same positions each time this is called
#     # Generate random samples within the bounds [-dr, dr] for each variable
#     x_samples = np.random.uniform(-dr, dr, num_samples)
#     y_samples = np.random.uniform(-dr, dr, num_samples)
#     print(x_samples)
#     print(y_samples)

#     # Evaluate the function at each sample point
#     #func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples, x_samples, y_samples)
#     func_values = np.zeros(num_samples, dtype='complex128') # Technically won't be complex here
#     for n in range(num_samples): # can simplify?
#         x_sample = -0.00075
#         y_sample = 0
# #        x_sample = x_samples[n]
# #        y_sample = y_samples[n]
#         g = functools.partial(f, x_pos_integrate=x_sample, y_pos_integrate=y_sample)
#         func_values[n] = monte_carlo_integration_momentum(g, dq)

#     # Calculate the average value of the function
#     avg_value = np.mean(func_values)
    
#     # The volume of the integration region
#     volume = (2 * dr)**2
    
#     # Estimate the integral as the average value times the volume
#     integral_estimate = avg_value * volume
    
#     return integral_estimate


# def monte_carlo_integration_momentum(f, dq, num_samples=200000):
#     ## Generate random samples within the bounds [-dq, dq] for each variable
#     # qix_samples = np.random.uniform(-dq, dq, num_samples)
#     # qiy_samples = np.random.uniform(-dq, dq, num_samples)
#     # qsx_samples = np.random.uniform(-dq, dq, num_samples)
#     # qsy_samples = np.random.uniform(-dq, dq, num_samples)

#     # Sample more heavily from point opposite expected point
#     # qix_samples = np.random.normal(loc=-0.00075, scale=dq/5, size=num_samples) * np.random.choice([-1, 1], size=num_samples)
#     # qiy_samples = np.random.normal(loc=0, scale=dq/5, size=num_samples) * np.random.choice([-1, 1], size=num_samples)
#     # qsx_samples = np.random.normal(loc=-0.00075, scale=dq/5, size=num_samples) * np.random.choice([-1, 1], size=num_samples)
#     # qsy_samples = np.random.normal(loc=0, scale=dq/5, size=num_samples) * np.random.choice([-1, 1], size=num_samples)

#     qix_dist = norm(loc=-0.00075, scale=dq/5)
#     qiy_dist = norm(loc=0, scale=dq/5)
#     qsx_dist = norm(loc=-0.00075, scale=dq/5)
#     qsy_dist = norm(loc=0, scale=dq/5)

#     qix_samples = qix_dist.rvs(num_samples) * np.random.choice([-1, 1], size=num_samples)
#     qiy_samples = qiy_dist.rvs(num_samples) * np.random.choice([-1, 1], size=num_samples)
#     qsx_samples = qsx_dist.rvs(num_samples) * np.random.choice([-1, 1], size=num_samples)
#     qsy_samples = qsy_dist.rvs(num_samples) * np.random.choice([-1, 1], size=num_samples)

#     # Evaluate the function at each sample point
#     func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples) / (2**4 * qix_dist.pdf(qix_samples) * qiy_dist.pdf(qiy_samples) * qsx_dist.pdf(qsx_samples) * qsy_dist.pdf(qsy_samples))

#     # Square the absolute value of the result
#     func_values_sq = np.abs(func_values)**2 

#     # Calculate the average value of the function
#     avg_value = np.mean(func_values)
    
#     # The volume of the integration region
#     volume = (2 * dq)**4
    
#     # Estimate the integral as the average value times the volume
#     integral_estimate = avg_value * volume
    
#     return integral_estimate


def monte_carlo_integration_position(f, dq, dr, num_samples=1):
  #  np.random.seed(0) #Use the same positions each time this is called
    # Generate random samples within the bounds [-dr, dr] for each variable
    x_samples = np.random.uniform(-dr, dr, num_samples)
    y_samples = np.random.uniform(-dr, dr, num_samples)
    print(x_samples)
    print(y_samples)

    # Evaluate the function at each sample point
    #func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples, x_samples, y_samples)
    func_values = np.zeros(num_samples, dtype='complex128') # Technically won't be complex here
    for n in range(num_samples): # can simplify?
        x_sample = 0.00075
        y_sample = 0
#        x_sample = x_samples[n]
#        y_sample = y_samples[n]
        g = functools.partial(f, x_pos_integrate=x_sample, y_pos_integrate=y_sample)
        func_values[n] = monte_carlo_integration_momentum(g, dq)

    # Calculate the average value of the function
    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dr)**2
    
    # Estimate the integral as the average value times the volume
    integral_estimate = avg_value * volume
    
    return integral_estimate





def complex_quadrature(func, lims, **kwargs):
    def real_func(x1, x2, x3, x4):
        return np.real(func(x1, x2, x3, x4))
    def imag_func(x1, x2, x3, x4):
        return np.imag(func(x1, x2, x3, x4))
    real_integral = integrate.nquad(real_func, lims, **kwargs)
    imag_integral = integrate.nquad(imag_func, lims, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1] + 1j * imag_integral[1])

def complex_quadrature2var(func, lims, **kwargs):
    def real_func(x1, x2):
        return np.real(func(x1, x2))
    def imag_func(x1, x2):
        return np.imag(func(x1, x2))
    real_integral = integrate.nquad(real_func, lims, **kwargs)
    imag_integral = integrate.nquad(imag_func, lims, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1] + 1j * imag_integral[1])


# Sellmeier equations for BBO
def n_o(wavelength):
    """
    Ordinary refractive index for BBO crystal.
    
    :param wavelength: Wavelength of light entering the crystal.
    """
    lambda_sq = wavelength**2
    return np.sqrt(np.abs(2.7405 + 0.0184 / (lambda_sq - 0.0179) - 0.0155 * lambda_sq))

def n_e(wavelength):
    """
    Extraordinary refractive index for BBO crystal.

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
    :param omegap: Angular frequency of the pump beam
    :param omegai: Angular frequency of the idler beam
    :param omegas: Angular frequency of the signal beam
    """
    lambdas = (2 * np.pi * C) / omegas # ?
    lambdai = (2 * np.pi * C) / omegai # ?
    lambdap = (2 * np.pi * C) / omegap # ?

    qpx = qsx + qix # Conservation of momentum?
    qpy = qsy + qiy # Conservation of momentum?
    qs_abs = np.sqrt(qsx**2 + qsy**2)
    qi_abs = np.sqrt(qix**2 + qiy**2)

    delta_k = n_o(lambdas) * omegas / C + n_o(lambdai) * omegai / C - eta(thetap, lambdap) * omegap / C + \
    C / (2 * eta(thetap, lambdap) * omegap) * (beta(thetap, lambdap)**2 * qpx**2 + gamma(thetap, lambdap)**2 * qpy**2) + \
    alpha(thetap, lambdap) * (qsx + qix) - C / (2 * n_o(lambdas) * omegas) * qs_abs**2 - \
    C / (2 * n_o(lambdai) * omegai) * qi_abs**2

    return delta_k

def pump_function(qpx, qpy, kp, omega):
    """ Function for the Gaussian pump beam. (equation 31)

    :param qpx: k-vector in the x direction for pump
    :param qpy: k-vector in the y direction for pump
    :param kpz: k-vector for pump (?? z?)
    :param omega: Pump frequency
    :param d: Distance behind the crystal where we are looking #??? 
    """
    qp_abs = np.sqrt(qpx**2 + qpy**2)
    d = 107.8e-2 # pg 15
    w0 = 388e-6 # beam waist in meters, page 8
    V = np.exp(-qp_abs**2 * w0**2 / 4) * np.exp(-1j * qp_abs**2 * d / (2 * kp)) # times a phase
    return V

def calculate_pair_generation_rate(x_pos, y_pos, thetap, omegap, omegai, omegas, dr):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param x_pos: Location of signal (idler) photon in the x direction a distance z away from the crystal
    :param y_pos: Location of signal (idler) photon in the y direction a distance z away from the crystal
    :param dr: One half the area of real space to integrate the signal and idler over (integrate over the origin) (Maybe don't have to pass?)
    """
    # Note: The integral could also be handled by doing an FFT

    # Multiply by detector efficiencies, and a constant dependent on epsilon_0 and chi_2

    # z is distance away from crystal along pump propagation direction
    z_pos = 35e-3 # 35 millimeters, page 15
    ks = omegas / C # ? Guess? divide by n?
    ki = omegai / C # ? Guess? divide by n?
    kpz = (omegas + omegai) / C # This is on page 8 in the bottom paragraph on the left column
    omegap = omegas + omegai # This is on page 8 in the bottom paragraph on the left column [NOTE this is also passed in]


    def rate_integrand(qix, qiy, qsx, qsy, x_pos_integrate, y_pos_integrate, integrate_over):
        if integrate_over == "signal":
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
            # Raise error
            print("error")

        qs_dot_rhos = (qsx * xs_pos + qsy * ys_pos)
        qi_dot_rhoi = (qix * xi_pos + qiy * yi_pos)
        qs_abs = np.sqrt(qsx**2 + qsy**2)
        qi_abs = np.sqrt(qix**2 + qiy**2)

        # qix + qsx ? pump_function
        integrand = np.exp(1j * (ks + ki) * z_pos) * pump_function(qix + qsx, qiy + qsy, kpz, omegap) * phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * \
        np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

        # integrand = pump_function(qix + qsx, qiy + qsy, kpz, omegap) * phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * \
        # np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

        # DEBUG
        #integrand = pump_function(qix + qsx, qiy + qsy, kpz, omegap)
#        integrand = phase_matching(qix, crystal_length)
      #  integrand = phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * pump_function(qix + qsx, qiy + qsy, kpz, omegap)
        #integrand = delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas)
     #   integrand = np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

   #     import pdb; pdb.set_trace()
        return integrand
    dqix = (omegai / C)*0.0034 # ?enclose circle in momentum space
    dqiy = (omegai / C)*0.0034 # 0.0014
    dqsx = (omegas / C)*0.0034 # Circle not enclosed
    dqsy = (omegas / C)*0.0034 # ? Guess


#     x = np.linspace(-dqix, dqix, 1000)
#     y = np.linspace(-dqiy, dqiy, 1000)
#     X, Y = np.meshgrid(x, y)
#     ##Momentum must be conserved, so qix = -qsx and qiy = -qiy?
#     ##(Assume qpx and qpy negligible? Though they appear in the expression for the pump beam)
# #    Z = np.abs(rate_integrand(X, Y, X, Y))
#     # print("test1")
#     # print(np.abs(rate_integrand(X, Y, -X, -Y)))
#     # print("test2")
#     # print(np.abs(rate_integrand(X, Y, X, Y)))
#     Z = np.abs(rate_integrand(X, Y, -X, -Y, 0, 0, "signal"))

#     # Z = np.abs(rate_integrand(X, Y))
#     plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
#     plt.xlabel("qx")
#     plt.ylabel("qy")
#     import pdb; pdb.set_trace()


    #### Hack the integral
    # Bring rate_integrand outside to calculate only once
    # real_part = np.sum(np.sum(np.sum(np.sum(np.real(rate_integrand(X, Y))))))
    # imag_part = np.sum(np.sum(np.sum(np.sum(np.imag(rate_integrand(X, Y))))))
    # result = real_part + 1j * imag_part

#    result = np.sum(np.sum(np.sum(np.sum(rate_integrand(X, Y)))))
    rate_integrand_signal = functools.partial(rate_integrand, integrate_over="idler")
    rate_integrand_idler = functools.partial(rate_integrand, integrate_over="signal")

    result_signal = monte_carlo_integration_position(rate_integrand_signal, dqix, dr)
    result_idler = result_signal # FOR debug
#    result_idler = monte_carlo_integration(rate_integrand_idler, dqix, dr)
#    print(f"result_signal: {result_signal}")
#    print(f"result_idler: {result_idler}")
#    result = np.abs(result_idler)**2 
    # #### Hack the integral
    # real_part = np.sum(np.sum(np.real(rate_integrand(X, Y, 0, 0))))
    # imag_part = np.sum(np.sum(np.imag(rate_integrand(X, Y, 0, 0))))
    # result = real_part + 1j * imag_part

    # error_estimate = 0

    # # #### Hack the integral
    # real_part = np.sum(np.sum(np.real(rate_integrand(0, 0, 0, 0))))
    # imag_part = np.sum(np.sum(np.imag(rate_integrand(0, 0, 0, 0))))
    # result = real_part + 1j * imag_part
    # error_estimate = 0


    #### Use Scipy for the integral
#    opts = {"limit": 2}
    opts = {}
#    result, error_estimate = complex_quadrature(rate_integrand, [[-dqix, dqix], [-dqiy, dqiy], [-dqsx, dqsx], [-dqsy, dqsy]], opts=opts)
#    result, error_estimate = complex_quadrature2var(rate_integrand, [[-dqix, dqix], [-dqiy, dqiy]], opts=opts)

 #   print(f"Integral result: {result}")
 #   print(f"Error estimate: {error_estimate}")
    ####
    return result_signal, result_idler

def plot_rings():
    """ Plot entangled pair rings. """
    start_time = time.time()

    pump_wavelength = 405e-9 # Pump wavelength in meters
    nominal_pump_frequency = (2 * np.pi * C) / pump_wavelength
    down_conversion_frequency = nominal_pump_frequency / 2 # 0.9995
#    signal_frequency = np.linspace(down_conversion_frequency*1.0005, down_conversion_frequency*0.995, 2) # Downcoverted photon wavelength in meters
#    idler_frequency = np.linspace(down_conversion_frequency*0.995, down_conversion_frequency*1.0005, 2) # Downcoverted photon wavelength in meters
    #idler_frequency = pump_frequency - signal_frequency # Downcoverted photon wavelength in meters
    signal_frequency = np.random.uniform(down_conversion_frequency*1.000, down_conversion_frequency*1.000, 1)
    idler_frequency = np.random.uniform(down_conversion_frequency*1.000, down_conversion_frequency*1.000, 1)
    #idler_frequency = nominal_pump_frequency - signal_frequency # Downcoverted photon wavelength in meters

#    signal_frequency = np.random.normal(down_conversion_frequency, down_conversion_frequency*0.0009, 1)
#    idler_frequency = np.random.normal(down_conversion_frequency, down_conversion_frequency*0.0009, 1)

    
    # Set parameters
    thetap = 28.95 * np.pi / 180

    # omegap = (2 * np.pi * C) / pump_wavelength # ?
    # omegai = (2 * np.pi * C) / down_conversion_wavelength # ?
    # omegas = (2 * np.pi * C) / down_conversion_wavelength # ?

    # Plot total output power as a function of theta_p


    # Plot beam in real space
    span = 100e-6 #??
    span = 3e-3#3e-4 #??
    span = 1e-3
    num_points = 500
    x = np.linspace(-span, span, num_points)

    E_signal_total = np.zeros(num_points, dtype='complex128')
    E_idler_total = np.zeros(num_points, dtype='complex128')
    for s, i in zip(signal_frequency, idler_frequency):
        print((C * np.pi * 2) /s)
        print((C * np.pi * 2) /i)

        calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)
        E_signal, E_idler = calculate_pair_generation_rate_vec(x, 0, thetap, s + i, i, s, span)
        # E_signal_total += np.abs(E_signal)**2
        # E_idler_total += np.abs(E_idler)**2
        E_signal_total += E_signal
        E_idler_total += E_idler

        # plt.figure(figsize=(8, 6))
        # plt.plot(x, np.abs(E_signal)**2)
        # plt.plot(x, np.abs(E_idler)**2)
        # import pdb; pdb.set_trace()

    z1 = E_signal_total
    z2 = E_idler_total

    plt.figure(figsize=(8, 6))
    plt.plot(x, z1)
    plt.plot(x, z2)
#    plt.plot(x, np.abs(z1 + z2)**2)
  
    plt.title( "BBO crystal entangled photons" ) 
    plt.show() 
 
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")


    import pdb; pdb.set_trace()

    ##Create a grid of x and y values
   ##span = 100e-6 #??
    span = 1e-3
    x = np.linspace(-span, span, 50)
    y = np.linspace(-span, span, 50)
    X, Y = np.meshgrid(x, y)

    calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)
    Z1, Z2 = calculate_pair_generation_rate_vec(X, Y, thetap, omegap, omegai, omegas, span)
    Z = np.abs(Z1 + Z2)**2
#    Z_idler = calculate_pair_generation_rate_vec(X, Y, thetap, omegap, omegai, omegas)

    # parallel_calculate = functools.partial(calculate_pair_generation_rate_vec, thetap=thetap, omegap=omegap, omegai=omegai, omegas=omegas, dr=span)
    # Z = Parallel(n_jobs=4)(delayed(parallel_calculate)(xi, yi) for xi in x for yi in y)
    # Z = np.reshape(np.array(Z), [50, 50]).T

#    Z = calculate_pair_generation_rate(x=4e-6, y=0, thetap=thetap, omegap=omegap, omegai=omegai, omegas=omegas)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title( "BBO crystal entangled photons" ) 
    plt.show()

    # Todo, plot conditional probability
    # Todo, momentum space plot

def main():
    """ main function """
    print("Hello world")
    plot_rings()


if __name__=="__main__": 
    main() 