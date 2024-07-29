import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy

# Constants
pump_wavelength = 405.0e-9  # Pump wavelength in meters
down_conversion_wavelength = 810e-9 # Downcoverted photon wavelength in meters
crystal_length = 0.002    # Length of the nonlinear crystal in meters
C = 2.99792e8 # Speed of light, in meters per second

def complex_quadrature(func, lims, **kwargs):
    def real_func(x1, x2, x3, x4):
        return np.real(func(x1, x2, x3, x4))
    def imag_func(x1, x2, x3, x4):
        return np.imag(func(x1, x2, x3, x4))
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
    lambdas = C / (2 * np.pi * omegas) # ?
    lambdai = C / (2 * np.pi * omegai) # ?
    lambdap = C / (2 * np.pi * omegap) # ?

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

def calculate_pair_generation_rate(x, y, thetap, omegap, omegai, omegas):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param x: Location in the x direction a distance z away from the crystal
    :param y: Location in the y direction a distance z away from the crystal
    """
    # Multiply by detector efficiencies, and a constant dependent on epsilon_0 and chi_2
    # V is pump shape, nominally dependent on qp.
    V = 1 # ? Guess

    # z is distance away from crystal along pump propagation direction
    z = 35e-3 # 35 millimeters
    ks = omegas / C # ? Guess? divide by n?
    ki = omegai / C # ? Guess? divide by n?
    kpz = (omegas + omegai) / C # This is on page 8 in the bottom paragraph on the left column
    omegap = omegas + omegai # This is on page 8 in the bottom paragraph on the left column

    def rate_integrand(qix, qiy, qsx, qsy):
        qs_dot_rhos = (qsx * x + qsy * y)
        qi_dot_rhoi = (qix * x + qiy * y)
        qs_abs = np.sqrt(qsx**2 + qsy**2)
        qi_abs = np.sqrt(qix**2 + qiy**2)

        rate = np.exp(1j * (ks + ki) * z) * pump_function(qix + qsx, qiy + qsy, kpz, omegap) * phase_matching(delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas), crystal_length) * \
        np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z / (2 * ks) - qi_abs**2 * z / (2 * ki)))

        # DEBUG
        # rate = pump_function(qix + qsx, qiy + qsy, kpz, omegap)
        # rate = delta_k_type_1(qsx, qix, qsy, qiy, thetap, omegap, omegai, omegas)
        # rate = np.exp(1j * (qs_dot_rhos + qi_dot_rhoi - qs_abs**2 * z / (2 * ks) - qi_abs**2 * z / (2 * ki)))
        return rate
    dqix = (omegai / C)*0.01 # ?
    dqiy = (omegai / C)*0.01 # 
    dqsx = (omegas / C)*0.01 # 
    dqsy = (omegas / C)*0.01 # ? Guess

    print(rate_integrand(dqix, dqix, dqix, dqix))    
    print(np.abs(rate_integrand(x, 0, 0, 0)))
    plt.plot(x, np.abs(rate_integrand(x, 0, 0, 0))**2) # Plot the pump function
    x = np.linspace(-dqix, dqix, 1000)
    y = np.linspace(-dqix, dqix, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.abs(rate_integrand(X, Y, 0, 0))
    plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='gray')
    plt.xlabel("qx")
    plt.ylabel("qy")

    import pdb; pdb.set_trace()
    opts = {"limit": 2}
    result, error_estimate = complex_quadrature(rate_integrand, [[-dqix, dqix], [-dqiy, dqiy], [-dqsx, dqsx], [-dqsy, dqsy]], opts=opts)
    print(f"Integral result: {result}")
    print(f"Error estimate: {error_estimate}")
    return np.abs(result)**2 

def plot_rings():
    """ Plot entangled pair rings. """
    # Set parameters
    thetap = 28.95 * np.pi / 180
    omegap = (2 * np.pi * C) / pump_wavelength # ?
    omegai = (2 * np.pi * C) / down_conversion_wavelength # ?
    omegas = (2 * np.pi * C) / down_conversion_wavelength # ?

    # Plot total output power as a function of theta_p
    # Sum (technically integrate) over real space






    # Plot beam in real space
    span = 100e-6 #??
    x = np.linspace(-span, span, 4)

    calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)
    z = calculate_pair_generation_rate_vec(x, 0, thetap, omegap, omegai, omegas)
    plt.figure(figsize=(8, 6))
    plt.plot(x, z)
  
    plt.title( "BBO crystal entangled photons" ) 
    plt.show() 




    import pdb; pdb.set_trace()

    # Create a grid of x and y values
    span = 100e-6 #??
    x = np.linspace(-span, span, 2)
    y = np.linspace(-span, span, 2)
    X, Y = np.meshgrid(x, y)

#    Z = calculate_pair_generation_rate(X, Y, thetap, omegap, omegai, omegas)
    calculate_pair_generation_rate_vec = np.vectorize(calculate_pair_generation_rate)
    Z = calculate_pair_generation_rate_vec(X, Y, thetap, omegap, omegai, omegas)
#    Z = calculate_pair_generation_rate(x=4e-6, y=0, thetap=thetap, omegap=omegap, omegai=omegai, omegas=omegas)
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin='lower', cmap='gray')
  
    plt.title( "BBO crystal entangled photons" ) 
    plt.show() 

def main():
    """ main function """
    print("Hello world")
    plot_rings()


if __name__=="__main__": 
    main() 