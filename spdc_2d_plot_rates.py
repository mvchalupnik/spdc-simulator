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

def monte_carlo_integration_momentum(f, dq, num_samples=200000):
    ## Generate random samples within the bounds [-dq, dq] for each variable
   # np.random.seed(1)
    qix_samples = np.random.uniform(-dq, dq, num_samples)
    qiy_samples = np.random.uniform(-dq, dq, num_samples)
    qsx_samples = np.random.uniform(-dq, dq, num_samples)
    qsy_samples = np.random.uniform(-dq, dq, num_samples)

    # qix_samples = np.random.uniform(-dq, -dq*0.6, num_samples)* np.random.choice([-1, 1], size=num_samples)
    # qiy_samples = np.random.uniform(-dq, -dq*0.6, num_samples) * np.random.choice([-1, 1], size=num_samples)
    # qsx_samples = np.random.uniform(-dq, -dq*0.6, num_samples)* np.random.choice([-1, 1], size=num_samples)
    # qsy_samples = np.random.uniform(-dq, -dq*0.6, num_samples)* np.random.choice([-1, 1], size=num_samples)


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


def monte_carlo_integration_position(f, dq, dr, num_samples=1):
  #  np.random.seed(0) #Use the same positions each time this is called
    # Generate random samples within the bounds [-dr, dr] for each variable
    x_samples = np.random.uniform(-dr, dr, num_samples)
    y_samples = np.random.uniform(-dr, dr, num_samples)

    # Evaluate the function at each sample point
    #func_values = f(qix_samples, qiy_samples, qsx_samples, qsy_samples, x_samples, y_samples)
    func_values = np.zeros(num_samples, dtype='complex128') # Technically won't be complex here
    for n in range(num_samples): # can simplify?
        x_sample = dr# x_samples[n] 0.00025 when integrating to 1 mm
        y_sample = 0#y_samples[n]
        g = functools.partial(f, x_pos_integrate=x_sample, y_pos_integrate=y_sample)
        func_values[n] = monte_carlo_integration_momentum(g, dq)

    # Calculate the average value of the function
    avg_value = np.mean(func_values)
    
    # The volume of the integration region
    volume = (2 * dr)**2
    
    # Estimate the integral as the average value times the volume
    integral_estimate = avg_value * volume
    
    return integral_estimate


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
    #    integrand = pump_function(qix + qsx, qiy + qsy, kpz, omegap)
#        integrand = phase_matching(qix, crystal_length)
        #integrand = phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * pump_function(qix + qsx, qiy + qsy, kpz, omegap)
        #integrand = delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas)
    #    integrand = np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z_pos / (2 * ks) - qi_abs**2 * z_pos / (2 * ki)))

   #     import pdb; pdb.set_trace()
        return integrand
    dqix = (omegai / C)*0.0014 # ?enclose circle in momentum space
    dqiy = (omegai / C)*0.0014 # 0.014 to enclose, 0.003 to run
    dqsx = (omegas / C)*0.0014 # 
    dqsy = (omegas / C)*0.0014 # ? Guess


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


    rate_integrand_signal = functools.partial(rate_integrand, integrate_over="idler")
    rate_integrand_idler = functools.partial(rate_integrand, integrate_over="signal")

    result_signal = monte_carlo_integration_position(rate_integrand_signal, dqix, dr)
    result_idler = result_signal
#    result_idler = monte_carlo_integration_position(rate_integrand_idler, dqix, dr) # comment to debug

    return result_signal, result_idler

def plot_rings():
    """ Plot entangled pair rings. """
    start_time = time.time()

    pump_wavelength = 405e-9 # Pump wavelength in meters
    down_conversion_wavelength = 810e-9

    # Set parameters
    thetap = 28.95 * np.pi / 180

    omegap = (2 * np.pi * C) / pump_wavelength # ?
    omegai = (2 * np.pi * C) / down_conversion_wavelength # ?
    omegas = (2 * np.pi * C) / down_conversion_wavelength # ?

    # Plot total output power as a function of theta_p
    # TODO

    # Plot beam in real space
    span = 100e-6 #??
    span = 3e-3#3e-4 #??

#    span = 1e-3

#    span = 2.8e-3
    num_points = 100
    x = np.linspace(-span, span, num_points)
    plt.figure(figsize=(8, 6))
    for a in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002]:
        calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)
        R_signal, R_idler = calculate_pair_generation_rate_vec(x, 0, thetap, omegas + omegai, omegai, omegas, a)
        print(f"R_signal: {R_signal}")

        z1 = R_signal
        z2 = R_idler

        plt.plot(x, z1, label=a)
   # plt.plot(x, z2)
#    plt.plot(x, np.abs(z1 + z2)**2)
    plt.legend()
    plt.title( "BBO crystal entangled photons" ) 
    plt.show() 
 
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")


    import pdb; pdb.set_trace()

    ##Create a grid of x and y values
   ##span = 100e-6 #??
    span = 2e-3
    x = np.linspace(-span, span, 50)
    y = np.linspace(-span, span, 50)
    X, Y = np.meshgrid(x, y)

    calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)
    Z1, Z2 = calculate_pair_generation_rate_vec(X, Y, thetap, omegap, omegai, omegas, span)

#    Z_idler = calculate_pair_generation_rate_vec(X, Y, thetap, omegap, omegai, omegas)

    # parallel_calculate = functools.partial(calculate_pair_generation_rate_vec, thetap=thetap, omegap=omegap, omegai=omegai, omegas=omegas, dr=span)
    # Z = Parallel(n_jobs=4)(delayed(parallel_calculate)(xi, yi) for xi in x for yi in y)
    # Z = np.reshape(np.array(Z), [50, 50]).T

#    Z = calculate_pair_generation_rate(x=4e-6, y=0, thetap=thetap, omegap=omegap, omegai=omegai, omegas=omegas)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(Z1), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
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