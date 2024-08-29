import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import json
from file_utils import get_current_time
import gc
from enum import Enum, auto
from typing import Callable

# Constants
C = 2.99792e8  # Speed of light, in meters per second.


# Create an Enum for the phase-matching type.
class PhaseMatchingCase(Enum):
    TYPE_ONE = 0
    TYPE_TWO_SIGNAL = auto()
    TYPE_TWO_IDLER = auto()


def grid_integration_momentum(
    f: Callable[[float, float, float, float], np.complex128],
    qx: float,
    qy: float,
    dqx: float,
    dqy: float,
    num_samples_wide_x: int,
    num_samples_wide_y: int,
    num_samples_narrow_x: int,
    num_samples_narrow_y: int,
    num_jobs: int,
):
    """
    Integrate a function f (equation 82-84) along four dimensions of momentum: qx (k-vector along x) of idler (signal),
    qy (k-vector along y) of idler (signal), dqx = qix + qsx, and dqy = qiy + qsy. A 4-D Fourier transform is used
    to simplify the integration. Integrating this function will provide the count rate for
    entangled photon pairs from a bulk BBO crystal at a given coordinate in real space.

    :param f: The function to integrate over.
    :param qx: One half of the interval of k-vector along x for the idler (signal), to integrate over.
    :param qy: One half of the interval of k-vector along y for the idler (signal), to integrate over.
    :param dqx: One half of the interval of the difference in k-vectors along x for the signal and idler,
        (qsx = -qix + dx).
    :param dqy: One half of the interval of the difference in k-vectors along y for the signal and idler,
        (qsy = -qiy + dy).
    :param num_samples_wide_x: The number of samples to integrate over along x for the [-dqix, dqix] interval.
    :param num_samples_wide_y: The number of samples to integrate over along y for the [-dqiy, dqiy] interval.
    :param num_samples_narrow_x: The number of samples to integrate over along y for the [-dqx, dqx] interval.
    :param num_samples_narrow_y: The number of samples to integrate over along y for the [-dqy, dqy] interval.
    :param num_jobs: The number of jobs to parallelize the batched function evaluation over.
    """
    qx_array = np.linspace(-qx, qx, num_samples_wide_x, dtype=np.float32,)
    qy_array = np.linspace(-qy, qy, num_samples_wide_y, dtype=np.float32,)

    dqx_array = np.linspace(-dqx, dqx, num_samples_narrow_x, dtype=np.float32,)
    dqy_array = np.linspace(-dqy, dqy, num_samples_narrow_y, dtype=np.float32,)

    # Generate the coordinate grid using meshgrid
    (qx_grid, qy_grid, dqx_grid, dqy_grid,) = np.meshgrid(
        qx_array, qy_array, dqx_array, dqy_array, indexing="ij",
    )

    # Flatten the grids
    qx_flat = qx_grid.ravel()
    qy_flat = qy_grid.ravel()
    dqx_flat = dqx_grid.ravel()
    dqy_flat = dqy_grid.ravel()
    time1 = time.time()

    qx_jobs = np.array_split(qx_flat, num_jobs)
    qy_jobs = np.array_split(qy_flat, num_jobs)
    dqx_jobs = np.array_split(dqx_flat, num_jobs)
    dqy_jobs = np.array_split(dqy_flat, num_jobs)

    result_grids = Parallel(n_jobs=num_jobs)(
        delayed(f)(qx_jobs[i], qy_jobs[i], dqx_jobs[i], dqy_jobs[i],)
        for i in range(num_jobs)
    )

    # Manually clean up large objects
    del (
        qx_grid,
        qy_grid,
        dqx_grid,
        dqy_grid,
        qx_flat,
        qy_flat,
        dqx_flat,
        dqy_flat,
        qx_jobs,
        qy_jobs,
        dqx_jobs,
        dqy_jobs,
    )
    gc.collect()

    result_grid = np.concatenate(result_grids)

    time2 = time.time()
    print(f"Seconds for integration, part 1: {time2-time1}")

    # Reshape grid
    reshaped_result_grid = np.reshape(
        result_grid, [len(qx_array), len(qy_array), len(dqx_array), len(dqy_array)]
    )

    del result_grids
    del result_grid
    gc.collect()

    # N-dimensional Fourier transform across all four axes
    ft_result_grid_shifted = np.fft.fftshift(np.fft.fftn(reshaped_result_grid))

    # Manually clean up large objects
    del reshaped_result_grid
    gc.collect()

    # Return the absolute value of this grid squared
    time3 = time.time()
    print(f"Seconds for integration, part 2: {time3-time2}")

    # Find the four Fourier transformed axes
    def get_fourier_transformed_axis(q_increment, num_points):
        if num_points % 2 == 0:
            ft_axis = (
                np.arange(-num_points / 2, num_points / 2 - 1, dtype=np.float32,)
                * 1
                / (q_increment * num_points)
            )
        else:
            ft_axis = (
                np.arange(
                    -(num_points - 1) / 2, (num_points - 1) / 2, dtype=np.float32,
                )
                * 1
                / (q_increment * num_points)
            )
        return ft_axis

    x_array = get_fourier_transformed_axis(
        q_increment=(2 * qx) / (2 * np.pi * num_samples_wide_x),
        num_points=num_samples_wide_x,
    )
    y_array = get_fourier_transformed_axis(
        q_increment=(2 * qy) / (2 * np.pi * num_samples_wide_y),
        num_points=num_samples_wide_y,
    )
    dx_array = get_fourier_transformed_axis(
        q_increment=(2 * dqx) / (2 * np.pi * num_samples_narrow_x),
        num_points=num_samples_narrow_x,
    )
    dy_array = get_fourier_transformed_axis(
        q_increment=(2 * dqy) / (2 * np.pi * num_samples_narrow_y),
        num_points=num_samples_narrow_y,
    )

    squared_result = np.asarray(np.abs(ft_result_grid_shifted) ** 2, dtype=np.float32,)
    del ft_result_grid_shifted

    # Return the absolute value of the Fourier transformed grid squared, as well as the four new axes
    return (
        squared_result,
        x_array,
        y_array,
        dx_array,
        dy_array,
    )


def n_o(wavelength: float):
    """
    Ordinary refractive index for BBO crystal, from the Sellmeier equations for BBO.

    :param wavelength: Wavelength of light entering the crystal, in microns.
    """
    lambda_sq_in_microns = (wavelength * 10 ** 6) ** 2
    return np.sqrt(
        np.abs(
            2.7405
            + 0.0184 / (lambda_sq_in_microns - 0.0179)
            - 0.0155 * lambda_sq_in_microns
        )
    )


def n_e(wavelength: float):
    """
    Extraordinary refractive index for BBO crystal, from the Sellmeier equations for BBO.

    :param wavelength: Wavelength of light entering the crystal, in microns.
    """
    lambda_sq_in_microns = (wavelength * 10 ** 6) ** 2
    return np.sqrt(
        np.abs(
            2.3730
            + 0.0128 / (lambda_sq_in_microns - 0.0156)
            - 0.0044 * lambda_sq_in_microns
        )
    )


def phase_matching(delta_k: float, L: float):
    """
    Return the phase matching function given a delta k-vector `delta_k` and length `L`.

    :param delta_k: Change in wave vector k, in Radians per meter.
    :param L: Length of crystal in meters.
    """
    return L * np.sinc(delta_k * L / 2) * np.exp(1j * delta_k * L / 2)


def alpha(thetap: float, lambd: float):
    """Return the alpha coefficient as a function of pump incidence tilt angle `thetap` and
    wavelength `lambd`.

    :param thetap: Angle in Radians along which pump beam enters BBO crystal (about y-axis).
    :param lambd: Wavelength, in meters.
    """
    alpha = ((n_o(lambd) ** 2 - n_e(lambd) ** 2) * np.sin(thetap) * np.cos(thetap)) / (
        n_o(lambd) ** 2 * np.sin(thetap) ** 2 + n_e(lambd) ** 2 * np.cos(thetap) ** 2
    )
    return alpha


def beta(thetap: float, lambd: float):
    """Return the `beta` coefficient as a function of pump incidence tilt angle `thetap` and
    wavelength `lambd`.

    :param thetap: Angle in Radians along which pump beam enters BBO crystal (about y-axis).
    :param lambd: Wavelength, in meters.
    """
    beta = (n_o(lambd) * n_e(lambd)) / (
        n_o(lambd) ** 2 * np.sin(thetap) ** 2 + n_e(lambd) ** 2 * np.cos(thetap) ** 2
    )
    return beta


def gamma(thetap: float, lambd: float):
    """Return the gamma coefficient as a function of pump incidence tilt angle `thetap` and
    wavelength `lambd`.

    :param thetap: Angle in Radians along which pump beam enters BBO crystal (about y-axis).
    :param lambd: Wavelength, in meters.
    """
    gamma = n_o(lambd) / np.sqrt(
        (n_o(lambd) ** 2 * np.sin(thetap) ** 2 + n_e(lambd) ** 2 * np.cos(thetap) ** 2)
    )
    return gamma


def eta(thetap: float, lambd: float):
    """
    Return the eta coefficient as a function of pump incidence tilt angle `thetap` and
    wavelength `lambd`.

    :param thetap: Angle in Radians along which pump beam enters BBO crystal (about y-axis).
    :param lambd: Wavelength, in meters.
    """
    eta = (n_o(lambd) * n_e(lambd)) / np.sqrt(
        (n_o(lambd) ** 2 * np.sin(thetap) ** 2 + n_e(lambd) ** 2 * np.cos(thetap) ** 2)
    )
    return eta


def delta_k_type_1(
    qsx: float, qix: float, qsy: float, qiy: float, thetap: float, omegas: float, omegai: float,
):
    """
    Return delta_k for type I phase matching, for BBO crystal.

    :param qsx: k-vector in the x direction for signal.
    :param qix: k-vector in the x direction for idler.
    :param qsy: k-vector in the y direction for signal.
    :param qiy: k-vector in the y direction for idler.
    :param thetap: Angle theta in Radians along which pump photon enters BBO crystal (about y-axis).
    :param omegas: Angular frequency of the signal photon.
    :param omegai: Angular frequency of the idler photon.
    """
    omegap = omegas + omegai

    lambdas = (2 * np.pi * C) / omegas
    lambdai = (2 * np.pi * C) / omegai
    lambdap = (2 * np.pi * C) / omegap

    qpx = qsx + qix  #
    qpy = qsy + qiy  # Conservation of momentum
    qs_abs = np.sqrt(qsx ** 2 + qsy ** 2)
    qi_abs = np.sqrt(qix ** 2 + qiy ** 2)

    delta_k = (
        n_o(lambdas) * omegas / C
        + n_o(lambdai) * omegai / C
        - eta(thetap, lambdap) * omegap / C
        + C
        / (2 * eta(thetap, lambdap) * omegap)
        * (
            beta(thetap, lambdap) ** 2 * qpx ** 2
            + gamma(thetap, lambdap) ** 2 * qpy ** 2
        )
        + alpha(thetap, lambdap) * (qsx + qix)
        - C / (2 * n_o(lambdas) * omegas) * qs_abs ** 2
        - C / (2 * n_o(lambdai) * omegai) * qi_abs ** 2
    )

    return delta_k


def delta_k_type_2(
    q1x: float, q2x: float, q1y: float, q2y: float, thetap: float, omega1: float, omega2: float,
):
    """Return delta_k for type II, case 1 phase matching, for BBO crystal.

    :param q1x: k-vector in the x direction for signal (idler).
    :param q2x: k-vector in the x direction for idler (signal).
    :param q1y: k-vector in the y direction for signal (idler).
    :param q2y: k-vector in the y direction for idler (signal).
    :param thetap: Angle theta in Radians along which pump photon enters the BBO crystal (about y-axis).
    :param omega1: Angular frequency of the signal (idler) photon.
    :param omega2: Angular frequency of the idler (signal) photon.
    """
    omegap = omega1 + omega2

    lambda1 = (2 * np.pi * C) / omega1
    lambda2 = (2 * np.pi * C) / omega2
    lambdap = (2 * np.pi * C) / omegap

    q2_abs = np.sqrt(q2x ** 2 + q2y ** 2)

    delta_k = (
        -alpha(thetap, lambda1) * q1x
        + eta(thetap, lambda1) * omega1 / C
        - C
        / (2 * eta(thetap, lambda1) * omega1)
        * (
            beta(thetap, lambda1) ** 2 * q1x ** 2
            + gamma(thetap, lambda1) ** 2 * q1y ** 2
        )
        + n_o(lambda2) * omega2 / C
        - C * q2_abs ** 2 / (2 * n_o(lambda2) * omega2)
        + alpha(thetap, lambdap) * (q1x + q2x)
        - eta(thetap, lambdap) * omegap / C
        + C
        / (2 * eta(thetap, lambdap) * omegap)
        * (
            beta(thetap, lambdap) ** 2 * (q1x + q2x) ** 2
            + gamma(thetap, lambdap) ** 2 * (q1y + q2y) ** 2
        )
    )

    return delta_k


def pump_function(qpx: float, qpy: float, kp: float, omega: float, w0: float, d: float):
    """Function producing the Gaussian pump beam.

    :param qpx: k-vector in the x direction for pump.
    :param qpy: k-vector in the y direction for pump.
    :param kp: k-vector in the z direction for pump.
    :param omega: Pump frequency.
    :param w0: Size of pump beam waist (meters).
    :param d: Distance of pump waist from crystal (meters).
    """
    qp_abs = np.sqrt(qpx ** 2 + qpy ** 2)
    V = np.exp(-(qp_abs ** 2) * w0 ** 2 / 4) * np.exp(-1j * qp_abs ** 2 * d / (2 * kp))
    return V


def get_rate_integrand(
    thetap: float, omegai: float, omegas: float, z_pos: float, w0: float, d: float, crystal_length: float, phase_matching_case: Enum,
):
    """
    Return the integrand used to calculate entangled photon generation rates.
    This is taken from equation 84 from Suman Karan et al 2020 J. Opt. 22 08350.

    :param thetap: Angle theta in Radians along which pump beam enters BBO crystal (about y-axis).
    :param omegai: The angular frequency of the idler.
    :param omegas: The angular frequency of the signal.
    :param z_pos: The view location in the z direction, from crystal (meters).
    :param w0: Size of pump beam waist (meter).
    :param d: Distance of pump waist from crystal (meters).
    :param crystal_length: The length of the crystal (meters).
    :param phase_matching_case: The phase-matching case: type I, type II idler or type II signal.
    """
    omegap = omegai + omegas

    ks = omegas / C
    ki = omegai / C
    kpz = (
        omegas + omegai
    ) / C  # This is on page 8 in the bottom paragraph on the left column

    def rate_integrand(qix, qiy, delta_qx, delta_qy):
        qsx = -qix + delta_qx
        qsy = -qiy + delta_qy

        qs_abs = np.sqrt(qsx ** 2 + qsy ** 2)
        qi_abs = np.sqrt(qix ** 2 + qiy ** 2)

        # Calculate delta_k based on the type of phase-matching
        if phase_matching_case == PhaseMatchingCase.TYPE_ONE:
            delta_k_term = delta_k_type_1(
                qsx=qsx,
                qix=qix,
                qsy=qsy,
                qiy=qiy,
                thetap=thetap,
                omegai=omegai,
                omegas=omegas,
            )
        elif phase_matching_case == PhaseMatchingCase.TYPE_TWO_SIGNAL:
            delta_k_term = delta_k_type_2(
                q1x=qsx,
                q2x=qix,
                q1y=qsy,
                q2y=qiy,
                thetap=thetap,
                omega1=omegas,
                omega2=omegai,
            )
        elif phase_matching_case == PhaseMatchingCase.TYPE_TWO_IDLER:
            delta_k_term = delta_k_type_2(
                q1x=qix,
                q2x=qsx,
                q1y=qiy,
                q2y=qsy,
                thetap=thetap,
                omega1=omegai,
                omega2=omegas,
            )
        else:
            raise TypeError(
                f"Error, unknown phase matching case {phase_matching_case}."
            )

        # The exp(1j * ((qsx * xs_pos + qsy * ys_pos) + (qix * xi_pos + qiy * yi_pos))) portion of the
        # integrand (in the text) makes it a Fourier transform,
        # avoiding the necessity of actually doing the four-dimensional momentum integral later.
        integrand = (
            np.exp(1j * (ks + ki) * z_pos)
            * pump_function(qix + qsx, qiy + qsy, kpz, omegap, w0, d,)
            * phase_matching(delta_k_term, crystal_length,)
            * np.exp(
                1j
                * (-(qs_abs ** 2) * z_pos / (2 * ks) - qi_abs ** 2 * z_pos / (2 * ki))
            )
        )

        return integrand

    return rate_integrand


def calculate_rings(
    thetap: float,
    omegai: float,
    omegas: float,
    momentum_span_wide_x: float,
    momentum_span_wide_y: float,
    momentum_span_narrow_x: float,
    momentum_span_narrow_y: float,
    num_samples_momentum_wide_x: int,
    num_samples_momentum_wide_y: int,
    num_samples_momentum_narrow_x: int,
    num_samples_momentum_narrow_y: int,
    z_pos: float,
    pump_waist_size: float,
    pump_waist_distance: float,
    crystal_length: float,
    num_jobs: int,
    phase_matching_type: Enum,
):
    """
    Return the entangled pair generation rate at location (x, y, z) from the crystal. Equation 84.

    :param thetap: Angle theta in Radians along which pump photon enters BBO crystal (about y-axis).
    :param omegai: Angular frequency of the idler.
    :param omegas: Angular frequency of the signal.
    :param momentum_span_wide_x: One half of the interval of k-vector along x for the idler, to integrate over.
    :param momentum_span_wide_y: One half of the interval of k-vector along y for the idler, to integrate over.
    :param momentum_span_narrow_x: One half of the interval of the difference in k-vectors along x for the
        signal and idler.
    :param momentum_span_narrow_y: One half of the interval of the difference in k-vectors along y for the
        signal and idler.
    :param num_samples_momentum_wide_x:The number of samples to integrate over along x for the
        momentum_span_wide_x interval.
    :param num_samples_momentum_wide_y: The number of samples to integrate over along y for the
        momentum_span_wide_y interval.
    :param num_samples_momentum_narrow_x: The number of samples to integrate over along y for the
        momentum_span_narrow_x interval.
    :param num_samples_momentum_narrow_y: The number of samples to integrate over along y for the
        momentum_span_narrow_y interval.
    :param z_pos: The view location in the z direction, from crystal (meters).
    :param pump_waist_size: Size of pump beam waist (meter).
    :param pump_waist_distance: Distance of pump waist from crystal (meters).
    :param crystal_length: The length of the crystal (meters).
    :param num_jobs: The number of jobs to parallelize the batched function evaluation over.
    :param phase_matching_type: The type of phase-matching (type I or type II).
    """

    qx = (omegai / C) * momentum_span_wide_x
    qy = (omegai / C) * momentum_span_wide_y
    dqx = (omegas / C) * momentum_span_narrow_x
    dqy = (omegas / C) * momentum_span_narrow_y

    def get_integral_grid(phase_matching_case,):
        rate_integrand = get_rate_integrand(
            thetap=thetap,
            omegai=omegai,
            omegas=omegas,
            z_pos=z_pos,
            w0=pump_waist_size,
            d=pump_waist_distance,
            crystal_length=crystal_length,
            phase_matching_case=phase_matching_case,
        )
        (result_grid, xs, ys, dxs, dys,) = grid_integration_momentum(
            f=rate_integrand,
            qx=qx,
            qy=qy,
            dqx=dqx,
            dqy=dqy,
            num_samples_wide_x=num_samples_momentum_wide_x,
            num_samples_wide_y=num_samples_momentum_wide_y,
            num_samples_narrow_x=num_samples_momentum_narrow_x,
            num_samples_narrow_y=num_samples_momentum_narrow_y,
            num_jobs=num_jobs,
        )

        # Sum result over two dimensions (integrate)
        result_grid_sum_over_dx = np.sum(result_grid, axis=3) * (2 * dqy)
        return (
            np.sum(result_grid_sum_over_dx, axis=2,) * (2 * dqx),
            xs,
            ys,
        )

    if phase_matching_type == 1:
        result, xs, ys = get_integral_grid(
            phase_matching_case=PhaseMatchingCase.TYPE_ONE
        )
    elif phase_matching_type == 2:
        result1, _, _ = get_integral_grid(
            phase_matching_case=PhaseMatchingCase.TYPE_TWO_SIGNAL
        )
        result2, xs, ys = get_integral_grid(
            phase_matching_case=PhaseMatchingCase.TYPE_TWO_IDLER
        )
        result = result1 + result2
    else:
        raise ValueError(f"Unknown phase_matching_type {phase_matching_type}.")

    return result, xs, ys


def simulate_ring_momentum(simulation_parameters,):
    """
    Simulate and plot the ring for a plane in momentum space, given fixed (x, y) for signal and fixed (x, y) for idler.

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    """
    num_plot_qx_points = simulation_parameters["num_plot_qx_points"]
    num_plot_qy_points = simulation_parameters["num_plot_qy_points"]

    thetap = simulation_parameters["thetap"]  # Incident pump angle, in Radians
    omegai = simulation_parameters["omegai"]  # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"]  # Signal frequency (Radians / sec)
    phase_matching_type = simulation_parameters[
        "phase_matching_type"
    ]  # Type I or Type II phase-matching

    momentum_span_x = simulation_parameters["momentum_span_x"]
    momentum_span_y = simulation_parameters["momentum_span_y"]
    signal_x_pos = simulation_parameters["signal_x_pos"]
    signal_y_pos = simulation_parameters["signal_y_pos"]
    idler_x_pos = simulation_parameters["idler_x_pos"]
    idler_y_pos = simulation_parameters["idler_y_pos"]
    z_pos = simulation_parameters["z_pos"]
    w0 = simulation_parameters["pump_waist_size"]
    d = simulation_parameters["pump_waist_distance"]
    crystal_length = simulation_parameters["crystal_length"]

    save_directory = simulation_parameters["save_directory"]

    qx = (omegai / C) * momentum_span_x
    qy = (omegai / C) * momentum_span_y

    x = np.linspace(-qx, qx, num_plot_qx_points)
    y = np.linspace(-qy, qy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)

    if phase_matching_type == 1:
        rate_integrand = get_rate_integrand(
            thetap=thetap,
            omegai=omegai,
            omegas=omegas,
            z_pos=z_pos,
            w0=w0,
            d=d,
            crystal_length=crystal_length,
            phase_matching_case=PhaseMatchingCase.TYPE_ONE,
        )
        Z1 = rate_integrand(X, Y, 2 * X, 2 * Y) * np.exp(
            1j
            * (X * signal_x_pos + Y * signal_y_pos + X * idler_x_pos + Y * idler_y_pos)
        )
        Z2 = rate_integrand(X, Y, 0, 0) * np.exp(
            1j
            * (X * signal_x_pos + Y * signal_y_pos - X * idler_x_pos - Y * idler_y_pos)
        )
    elif phase_matching_type == 2:
        rate_integrand_2s = get_rate_integrand(
            thetap=thetap,
            omegai=omegai,
            omegas=omegas,
            z_pos=z_pos,
            w0=w0,
            d=d,
            crystal_length=crystal_length,
            phase_matching_case=PhaseMatchingCase.TYPE_TWO_SIGNAL,
        )
        rate_integrand_2i = get_rate_integrand(
            thetap=thetap,
            omegai=omegai,
            omegas=omegas,
            z_pos=z_pos,
            w0=w0,
            d=d,
            crystal_length=crystal_length,
            phase_matching_case=PhaseMatchingCase.TYPE_TWO_IDLER,
        )
        Z1 = rate_integrand_2s(X, Y, 2 * X, 2 * Y) * np.exp(
            1j
            * (X * signal_x_pos + Y * signal_y_pos + X * idler_x_pos + Y * idler_y_pos)
        )
        +rate_integrand_2i(X, Y, 2 * X, 2 * Y) * np.exp(
            1j
            * (X * signal_x_pos + Y * signal_y_pos + X * idler_x_pos + Y * idler_y_pos)
        )
        Z2 = rate_integrand_2s(X, Y, 0, 0) * np.exp(
            1j
            * (X * signal_x_pos + Y * signal_y_pos - X * idler_x_pos - Y * idler_y_pos)
        )
        +rate_integrand_2i(X, Y, 0, 0) * np.exp(
            1j
            * (X * signal_x_pos + Y * signal_y_pos - X * idler_x_pos - Y * idler_y_pos)
        )
    else:
        raise ValueError(f"Unknown phase_matching_type {phase_matching_type}.")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    im1 = ax1.imshow(
        np.abs(Z1),
        extent=(x.min(), x.max(), y.min(), y.max(),),
        origin="lower",
        cmap="gray",
    )
    ax1.set_title("Abs(Integrand)")
    ax1.set_xlabel("$q_x$ ($q_{xi} = q_{xs}$) (Rad/m)")
    ax1.set_ylabel("$q_y$ ($q_{yi} = q_{ys}$) (Rad/m)")
    ax1.tick_params(axis="both", labelsize=4)
    cb1 = fig.colorbar(im1, ax=ax1, location="right", shrink=0.6,)
    cb1.ax.tick_params(labelsize=4)
    cb1.ax.yaxis.offsetText.set_fontsize(4)

    im2 = ax2.imshow(
        np.real(Z1),
        extent=(x.min(), x.max(), y.min(), y.max(),),
        origin="lower",
        cmap="jet",
    )
    ax2.set_title("Re(Integrand)")
    ax2.set_xlabel("$q_x$ ($q_{xi} = q_{xs}$) (Rad/m)")
    ax2.set_ylabel("$q_y$ ($q_{yi} = q_{ys}$) (Rad/m)")
    ax2.tick_params(axis="both", labelsize=4)
    cb2 = fig.colorbar(im2, ax=ax2, location="right", shrink=0.6,)
    cb2.ax.tick_params(labelsize=4)
    cb2.ax.yaxis.offsetText.set_fontsize(4)

    x = np.linspace(-qx, qx, num_plot_qx_points)
    y = np.linspace(-qy, qy, num_plot_qy_points)
    X, Y = np.meshgrid(x, y)
    im3 = ax3.imshow(
        np.abs(Z2),
        extent=(x.min(), x.max(), y.min(), y.max(),),
        origin="lower",
        cmap="gray",
    )
    ax3.set_xlabel("$q_x$ ($q_{xi} = -q_{xs}$) (Rad/m)")
    ax3.set_ylabel("$q_y$ ($q_{yi} = -q_{ys}$) (Rad/m)")
    ax3.tick_params(axis="both", labelsize=4)
    cb3 = fig.colorbar(im3, ax=ax3, location="right", shrink=0.6,)
    cb3.ax.tick_params(labelsize=4)
    cb3.ax.yaxis.offsetText.set_fontsize(4)

    im4 = ax4.imshow(
        np.real(Z2),
        extent=(x.min(), x.max(), y.min(), y.max(),),
        origin="lower",
        cmap="jet",
    )
    ax4.set_xlabel("$q_x$ ($q_{xi} = -q_{xs}$) (Rad/m)")
    ax4.set_ylabel("$q_y$ ($q_{yi} = -q_{ys}$) (Rad/m)")
    ax4.tick_params(axis="both", labelsize=4)
    cb4 = fig.colorbar(im4, ax=ax4, location="right", shrink=0.6,)
    cb4.ax.tick_params(labelsize=4)
    cb4.ax.yaxis.offsetText.set_fontsize(4)

    plt.tight_layout()

    # Get current time for file name
    time_str = get_current_time()

    plt.savefig(
        f"{save_directory}/{time_str}_momentum.png", dpi=300,
    )
    plt.close()

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_momentum_params.pkl", "wb",) as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_momentum_params.txt", "w",) as file:
        file.write(json.dumps(simulation_parameters))


def simulate_rings(simulation_parameters,):
    """
    Simulate and plot entangled pair rings by integrating the conditional probability of detecting the
    signal photon given detecting the idler photon, integrating over the possible positions of the idler photon.

    :param simulation_parameters: A dict containing relevant parameters for running the simulation.
    """
    start_time = time.time()

    thetap = simulation_parameters["thetap"]  # Incident pump angle, in Radians
    omegai = simulation_parameters["omegai"]  # Idler frequency (Radians / sec)
    omegas = simulation_parameters["omegas"]  # Signal frequency (Radians / sec)
    momentum_span_wide_x = simulation_parameters["momentum_span_wide_x"]
    momentum_span_wide_y = simulation_parameters["momentum_span_wide_y"]
    momentum_span_narrow_x = simulation_parameters["momentum_span_narrow_x"]
    momentum_span_narrow_y = simulation_parameters["momentum_span_narrow_y"]
    num_samples_momentum_wide_x = simulation_parameters["num_samples_momentum_wide_x"]
    num_samples_momentum_wide_y = simulation_parameters["num_samples_momentum_wide_y"]
    num_samples_momentum_narrow_x = simulation_parameters[
        "num_samples_momentum_narrow_x"
    ]
    num_samples_momentum_narrow_y = simulation_parameters[
        "num_samples_momentum_narrow_y"
    ]
    pump_waist_size = simulation_parameters[
        "pump_waist_size"
    ]  # Size of pump beam waist
    pump_waist_distance = simulation_parameters[
        "pump_waist_distance"
    ]  # Distance of pump waist from crystal (meters)
    z_pos = simulation_parameters[
        "z_pos"
    ]  # View location in the z direction, from crystal (meters)
    crystal_length = simulation_parameters[
        "crystal_length"
    ]  # Length of the crystal, in meters
    phase_matching_type = simulation_parameters["phase_matching_type"]

    num_jobs = simulation_parameters[
        "num_jobs"
    ]  # Number of jobs to use for estimating the integral.
    save_directory = simulation_parameters["save_directory"]

    # Run calculate_pair_generation_rate in parallel
    Z1, xis, yis = calculate_rings(
        thetap=thetap,
        omegai=omegai,
        omegas=omegas,
        momentum_span_wide_x=momentum_span_wide_x,
        momentum_span_wide_y=momentum_span_wide_y,
        momentum_span_narrow_x=momentum_span_narrow_x,
        momentum_span_narrow_y=momentum_span_narrow_y,
        num_samples_momentum_wide_x=num_samples_momentum_wide_x,
        num_samples_momentum_wide_y=num_samples_momentum_wide_y,
        num_samples_momentum_narrow_x=num_samples_momentum_narrow_x,
        num_samples_momentum_narrow_y=num_samples_momentum_narrow_y,
        z_pos=z_pos,
        pump_waist_size=pump_waist_size,
        pump_waist_distance=pump_waist_distance,
        crystal_length=crystal_length,
        num_jobs=num_jobs,
        phase_matching_type=phase_matching_type,
    )

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

    # Get current time for file name
    time_str = get_current_time()

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.abs(Z1),
        extent=(xis.min(), xis.max(), yis.min(), yis.max(),),
        origin="lower",
        cmap="gray",
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("BBO crystal entangled photons rates")
    plt.savefig(
        f"{save_directory}/{time_str}_rings.png", dpi=300,
    )
    plt.close()

    # Save data to a pickled file
    with open(f"{save_directory}/{time_str}_rings.pkl", "wb",) as file:
        pickle.dump(Z1, file)

    # Save parameters to a pickled file
    with open(f"{save_directory}/{time_str}_rings_params.pkl", "wb",) as file:
        pickle.dump(simulation_parameters, file)

    # Save parameters to a text file
    with open(f"{save_directory}/{time_str}_rings_params.txt", "w",) as file:
        file.write(json.dumps(simulation_parameters))

    # Save time to a text file
    time_info = {"Time Elapsed in seconds": end_time - start_time}
    with open(f"{save_directory}/{time_str}_rings_time.txt", "w",) as file:
        file.write(json.dumps(time_info))


# Plot total output power as a function of theta_p and other params
# TODO
