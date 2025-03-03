from typing import Dict, Tuple, Literal

import numpy as np
from pandas import concat, DataFrame
from scipy.integrate import simps
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm, gaussian_kde


def fit_kde(pdf_point_arrays: tuple) -> tuple:
    """
    Fits a Kernel Density Estimation (KDE) to the given implied probability density function (PDF).

    Args:
        pdf_point_arrays (tuple): A tuple containing:
            - A numpy array of price values
            - A numpy array of PDF values

    Returns:
        tuple: (prices, fitted_pdf), where:
            - prices: The original price array
            - fitted_pdf: The KDE-fitted probability density values
    """

    # Unpack tuple
    prices, pdf_values = pdf_point_arrays

    # Normalize PDF to ensure it integrates to 1
    pdf_values /= np.trapz(pdf_values, prices)  # Use trapezoidal rule for normalization

    # Fit KDE using price points weighted by the normalized PDF
    kde = gaussian_kde(prices, weights=pdf_values)

    # Generate KDE-fitted PDF values
    fitted_pdf = kde.pdf(prices)

    return (prices, fitted_pdf)


def calculate_pdf(
    options_data: DataFrame,
    current_price: float,
    days_forward: int,
    risk_free_rate: float,
    solver_method: str,
) -> Tuple[np.ndarray]:
    """The main execution path for the pdf module. Takes a `DataFrame` of
    options data as input and makes a series of function calls to

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at
        risk_free_rate: annual risk free rate in nominal terms

    Returns:
        a tuple containing the price and density values (in numpy arrays)
        of the calculated PDF
    """

    # options_data, min_strike, max_strike = _extrapolate_call_prices(
    #     options_data, current_price
    # )
    min_strike = int(options_data.strike.min())
    max_strike = int(options_data.strike.max())
    options_data = _calculate_last_price(options_data)
    options_data = _calculate_IV(
        options_data, current_price, days_forward, risk_free_rate, solver_method
    )
    denoised_iv = _fit_bspline_IV(options_data)
    pdf = _create_pdf_point_arrays(
        denoised_iv, current_price, days_forward, risk_free_rate
    )
    return _crop_pdf(pdf, min_strike, max_strike)


def calculate_cdf(
    pdf_point_arrays: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the cumulative probability at each price. Takes as input the array
    of pdf and array of prices, and calculates the cumulative probability as the
    numerical integral over the pdf function.

    For simplicity, it assumes that the CDF at the starting price
        = 1 - 0.5*(total area of the pdf)
    and therefore it adds 0.5*(total area of the pdf) to every cdf for the
    remainder of the domain

    Args:
        pdf_point_arrays: a tuple containing arrays representing a PDF

    Returns:
        A tuple containing the price domain and the point values of the CDF
    """
    x_array, pdf_array = pdf_point_arrays
    cdf = []
    n = len(x_array)

    total_area = simps(y=pdf_array[0:n], x=x_array)
    remaining_area = 1 - total_area

    for i in range(n):
        if i == 0:
            integral = 0.0 + remaining_area / 2
        else:
            integral = (
                simps(y=pdf_array[i - 1 : i + 1], x=x_array[i - 1 : i + 1]) + cdf[-1]
            )
        cdf.append(integral)

    return (x_array, cdf)


def calculate_quartiles(
    cdf_point_arrays: Tuple[np.ndarray, np.ndarray],
) -> Dict[float, float]:
    """

    Args:
        cdf_point_arrays: a tuple containing arrays representing a CDF

    Returns:
        a DataFrame containing the quartiles of the given CDF
    """
    cdf_interpolated = interp1d(cdf_point_arrays[0], cdf_point_arrays[1])
    x_start, x_end = cdf_point_arrays[0][0], cdf_point_arrays[0][-1]
    return {
        0.25: brentq(lambda x: cdf_interpolated(x) - 0.25, x_start, x_end),
        0.5: brentq(lambda x: cdf_interpolated(x) - 0.5, x_start, x_end),
        0.75: brentq(lambda x: cdf_interpolated(x) - 0.75, x_start, x_end),
    }


def _extrapolate_call_prices(
    options_data: DataFrame, current_price: float
) -> tuple[DataFrame, int, int]:
    """Extrapolate the price of the call options to strike prices outside
    the range of options_data. Extrapolation is done to zero and twice the
    highest strike price in options_data. Done to give the resulting PDF
    more stability.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        current_price: the current price of the security

    Returns:
        the extended options_data DataFrame
    """
    min_strike = int(options_data.strike.min())
    max_strike = int(options_data.strike.max())
    lower_extrapolation = DataFrame(
        {"strike": p, "last_price": current_price - p} for p in range(0, min_strike)
    )
    upper_extrapolation = DataFrame(
        {
            "strike": p,
            "last_price": 0,
        }
        for p in range(max_strike + 1, max_strike * 2)
    )
    return (
        concat([lower_extrapolation, options_data, upper_extrapolation]),
        min_strike,
        max_strike,
    )


def _calculate_last_price(options_data: DataFrame) -> DataFrame:
    """Take the last-price of the options at each strike price.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']

    Returns:
        the options_data DataFrame, with an additional column for mid-price
    """
    options_data["last_price"] = options_data["last_price"]
    options_data = options_data[options_data.last_price >= 0]
    return options_data


def _calculate_IV(
    options_data: DataFrame,
    current_price: float,
    days_forward: int,
    risk_free_rate: float,
    solver_method: Literal["newton", "brent"],
) -> DataFrame:
    """
    Calculate the implied volatility (IV) of the options in options_data.

    Args:
        options_data (pd.DataFrame): A DataFrame containing option price data with
            columns ['strike', 'last_price'].
        current_price (float): The current price of the security.
        days_forward (int): The number of days in the future to estimate the
            price probability density at.
        risk_free_rate (float, optional): Annual risk-free rate in nominal terms. Defaults to 0.
        solver_method (Literal["newton", "brent"], optional):
            The method used to solve for IV.
            - "newton" (default) uses Newton-Raphson iteration.
            - "brent" uses Brent’s method (more stable).

    Returns:
        DataFrame: The options_data DataFrame with an additional column for implied volatility (IV).
    """
    years_forward = days_forward / 365

    # Choose the IV solver method
    if solver_method == "newton":
        iv_solver = _bs_iv_newton_method
    elif solver_method == "brent":
        iv_solver = _bs_iv_brent_method
    else:
        raise ValueError("Invalid solver_method. Choose either 'newton' or 'brent'.")

    options_data["iv"] = options_data.apply(
        lambda row: iv_solver(
            row.last_price, current_price, row.strike, years_forward, r=risk_free_rate
        ),
        axis=1,
    )

    # Remove rows where IV could not be calculated
    options_data = options_data.dropna()

    return options_data


def _fit_bspline_IV(options_data: DataFrame) -> DataFrame:
    """Fit a bspline function on the IV observations, in effect denoising the IV.
        From this smoothed IV function, generate (x,y) coordinates
        representing observations of the denoised IV

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price', 'iv']

    Returns:
        a tuple containing x-axis values (index 0) and y-axis values (index 1)
        'x' represents the price
        'y' represents the value of the IV
    """
    x = options_data["strike"]
    y = options_data["iv"]

    # fit the bspline using scipy.interpolate.splrep, with k=3
    """
    Bspline Parameters:
        t = the vector of knots
        c = the B-spline coefficients
        k = the degree of the spline
    """
    tck = interpolate.splrep(x, y, s=10, k=3)

    dx = 0.1  # setting dx = 0.1 for numerical differentiation
    domain = int((max(x) - min(x)) / dx)

    # compute (x,y) observations of the denoised IV from the fitted IV function
    x_new = np.linspace(min(x), max(x), domain)
    y_fit = interpolate.BSpline(*tck)(x_new)

    return (x_new, y_fit)


def _create_pdf_point_arrays(
    denoised_iv: tuple, current_price: float, days_forward: int, risk_free_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two arrays containing x- and y-axis values representing a calculated
    price PDF

    Args:
        denoised_iv: (x,y) observations of the denoised IV
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at
        risk_free_rate: the current annual risk free interest rate, nominal terms

    Returns:
        a tuple containing x-axis values (index 0) and y-axis values (index 1)
    """

    # extract the x and y vectors from the denoised IV observations
    x_IV = denoised_iv[0]
    y_IV = denoised_iv[1]

    # convert IV-space to price-space
    # re-values call options using the BS formula, taking in as inputs S, domain, IV, and time to expiry
    years_forward = days_forward / 365
    interpolated = _call_value(current_price, x_IV, y_IV, years_forward, risk_free_rate)
    first_derivative_discrete = np.gradient(interpolated, x_IV)
    second_derivative_discrete = np.gradient(first_derivative_discrete, x_IV)

    # apply coefficient to reflect the time value of money
    pdf = np.exp(risk_free_rate * years_forward) * second_derivative_discrete

    # ensure non-negative pdf values (may occur for far OOM options)
    pdf = np.maximum(pdf, 0)  # Set all negative values to 0

    return (x_IV, pdf)


def _crop_pdf(
    pdf: Tuple[np.ndarray, np.ndarray], min_strike: float, max_strike: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the PDF to the range of the original options data"""
    l, r = 0, len(pdf[0]) - 1
    while pdf[0][l] < min_strike:
        l += 1
    while pdf[0][r] > max_strike:
        r -= 1
    return pdf[0][l : r + 1], pdf[1][l : r + 1]


def _bs_iv_brent_method(price, S, K, t, r):
    """
    Computes the implied volatility (IV) of a European call option using Brent’s method.

    This function finds the implied volatility by solving for sigma (volatility) in the
    Black-Scholes pricing formula. It uses Brent’s root-finding algorithm to find the
    volatility that equates the Black-Scholes model price to the observed market price.

    Args:
        price (float): The observed market price of the option.
        S (float): The current price of the underlying asset.
        K (float): The strike price of the option.
        t (float): Time to expiration in years.
        r (float, optional): The risk-free interest rate (annualized). Defaults to 0.

    Returns:
        float: The implied volatility (IV) if a solution is found.
        np.nan: If the function fails to converge to a solution.

    Raises:
        ValueError: If Brent’s method fails to find a root in the given range.

    Notes:
        - The function searches for IV within the range [1e-6, 5.0] (0.0001% to 500% volatility).
        - If `t <= 0`, the function returns NaN since volatility is undefined for expired options.
        - If the function fails to converge, it returns NaN instead of raising an exception.
    """

    if t <= 0:
        return np.nan  # No volatility if time is zero or negative

    try:
        return brentq(lambda iv: _call_value(S, K, iv, t, r) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan  # Return NaN if no solution is found


def _bs_iv_newton_method(
    price: float,
    S: float,
    K: float,
    t: float,
    r: float,
    precision: float = 1e-4,
    initial_guess: float = None,
    max_iter: int = 1000,
    verbose: bool = False,
) -> float:
    """
    Computes the implied volatility (IV) using Newton-Raphson iteration.

    Args:
        price (float): Observed market price of the option.
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        t (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        precision (float, optional): Convergence tolerance for Newton's method. Defaults to 1e-4.
        initial_guess (float, optional): Initial guess for IV. Defaults to 0.2 for ATM options, 0.5 otherwise.
        max_iter (int, optional): Maximum number of iterations before stopping. Defaults to 1000.
        verbose (bool, optional): If True, prints debugging information. Defaults to False.

    Returns:
        float: The implied volatility if found, otherwise np.nan.
    """

    # Set a dynamic initial guess if none is provided
    if initial_guess is None:
        initial_guess = (
            0.2 if abs(S - K) < 0.1 * S else 0.5
        )  # Lower guess for ATM, higher for OTM

    iv = initial_guess

    for i in range(max_iter):
        # Compute Black-Scholes model price and Vega
        P = _call_value(S, K, iv, t, r)
        diff = price - P

        # Check for convergence
        if abs(diff) < precision:
            return iv

        # Compute Vega (gradient)
        grad = _call_vega(S, K, iv, t, r)

        # Prevent division by near-zero Vega to avoid large jumps
        if abs(grad) < 1e-6:
            if verbose:
                print(f"Iteration {i}: Vega too small (grad={grad:.6f}), stopping.")
            return np.nan

        # Newton-Raphson update
        iv += diff / grad

        # Prevent extreme IV values (e.g., IV > 500%)
        if iv < 1e-6 or iv > 5.0:
            if verbose:
                print(f"Iteration {i}: IV out of bounds (iv={iv:.6f}), stopping.")
            return np.nan

    if verbose:
        print(f"Did not converge after {max_iter} iterations")

    return np.nan  # Return NaN if the method fails to converge


def _call_vega(S, K, sigma, t, r):
    # TODO: refactor this function (style)
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma**2 / 2) * t)
    return np.multiply(S, norm.pdf(d1)) * np.sqrt(t)


def _call_value(S, K, sigma, t, r):
    # TODO: refactor this function (style)
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma**2 / 2) * t)
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1), S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))
