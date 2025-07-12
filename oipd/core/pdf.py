from typing import Dict, Tuple, Literal

import numpy as np
from pandas import concat, DataFrame
from scipy.integrate import simpson
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm, gaussian_kde

from oipd.pricing import get_pricer


class OIPDError(Exception):
    """Base exception for OIPD package"""

    pass


class InvalidInputError(OIPDError):
    """Exception raised for invalid input parameters"""

    pass


class CalculationError(OIPDError):
    """Exception raised when calculations fail"""

    pass


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

    Raises:
        InvalidInputError: If input arrays are empty or have mismatched lengths
    """
    # Validate input
    if not isinstance(pdf_point_arrays, tuple) or len(pdf_point_arrays) != 2:
        raise InvalidInputError("pdf_point_arrays must be a tuple of two arrays")

    # Unpack tuple
    prices, pdf_values = pdf_point_arrays

    if len(prices) == 0 or len(pdf_values) == 0:
        raise InvalidInputError("Input arrays cannot be empty")

    if len(prices) != len(pdf_values):
        raise InvalidInputError(
            f"Price and PDF arrays must have same length. Got {len(prices)} and {len(pdf_values)}"
        )

    # Normalize PDF to ensure it integrates to 1
    pdf_values /= np.trapz(pdf_values, prices)  # Use trapezoidal rule for normalization

    try:
        # Fit KDE using price points weighted by the normalized PDF
        kde = gaussian_kde(prices, weights=pdf_values)

        # Generate KDE-fitted PDF values
        fitted_pdf = kde.pdf(prices)
    except Exception as e:
        raise CalculationError(f"Failed to fit KDE: {str(e)}")

    return (prices, fitted_pdf)


def calculate_pdf(
    options_data: DataFrame,
    current_price: float,
    days_forward: int,
    risk_free_rate: float,
    solver_method: str,
    pricing_engine: str = "bs",
    dividend_yield: float | None = None,
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

    Raises:
        InvalidInputError: If input parameters are invalid
        CalculationError: If PDF calculation fails
    """
    # Validate inputs
    if not isinstance(options_data, DataFrame):
        raise InvalidInputError("options_data must be a pandas DataFrame")

    if options_data.empty:
        raise InvalidInputError("options_data cannot be empty")

    if current_price <= 0:
        raise InvalidInputError(f"current_price must be positive, got {current_price}")

    if days_forward <= 0:
        raise InvalidInputError(f"days_forward must be positive, got {days_forward}")

    if not -1 <= risk_free_rate <= 1:
        raise InvalidInputError(
            f"risk_free_rate seems unrealistic: {risk_free_rate}. Expected value between -1 and 1"
        )

    if solver_method not in ["newton", "brent"]:
        raise InvalidInputError(
            f"solver_method must be 'newton' or 'brent', got '{solver_method}'"
        )

    # options_data, min_strike, max_strike = _extrapolate_call_prices(
    #     options_data, current_price
    # )
    min_strike = int(options_data.strike.min())
    max_strike = int(options_data.strike.max())

    options_data = _calculate_last_price(options_data)

    if options_data.empty:
        raise CalculationError("No valid options data after price calculation")

    options_data = _calculate_IV(
        options_data,
        current_price,
        days_forward,
        risk_free_rate,
        solver_method,
        dividend_yield=dividend_yield,
    )

    if options_data.empty:
        raise CalculationError("Failed to calculate implied volatility for any options")

    denoised_iv = _fit_bspline_IV(options_data)
    pdf = _create_pdf_point_arrays(
        denoised_iv,
        current_price,
        days_forward,
        risk_free_rate,
        dividend_yield,
        pricing_engine,
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

    Raises:
        InvalidInputError: If input is invalid
    """
    if not isinstance(pdf_point_arrays, tuple) or len(pdf_point_arrays) != 2:
        raise InvalidInputError("pdf_point_arrays must be a tuple of two arrays")

    x_array, pdf_array = pdf_point_arrays

    if len(x_array) == 0:
        raise InvalidInputError("Input arrays cannot be empty")

    if len(x_array) != len(pdf_array):
        raise InvalidInputError("Price and PDF arrays must have same length")

    cdf = []
    n = len(x_array)

    total_area = simpson(y=pdf_array[0:n], x=x_array)
    remaining_area = 1 - total_area

    for i in range(n):
        if i == 0:
            integral = 0.0 + remaining_area / 2
        else:
            integral = (
                simpson(y=pdf_array[i - 1 : i + 1], x=x_array[i - 1 : i + 1]) + cdf[-1]
            )
        cdf.append(integral)

    return (x_array, np.array(cdf))


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
    dividend_yield: float | None = None,
) -> DataFrame:
    """
    Vectorised implied volatility solver.
    """
    years_forward = days_forward / 365

    # Choose the IV solver method
    if solver_method == "newton":
        iv_solver_scalar = _bs_iv_newton_method
    elif solver_method == "brent":
        iv_solver_scalar = _bs_iv_brent_method
    else:
        raise ValueError("Invalid solver_method. Choose either 'newton' or 'brent'.")

    # Vectorised wrapper – falls back to scalar solver per strike
    prices_arr = options_data["last_price"].values
    strikes_arr = options_data["strike"].values

    q = dividend_yield or 0.0
    iv_values = np.fromiter(
        (
            iv_solver_scalar(p, current_price, k, years_forward, r=risk_free_rate, q=q)
            for p, k in zip(prices_arr, strikes_arr)
        ),
        dtype=float,
    )

    options_data = options_data.copy()
    options_data["iv"] = iv_values
    # Drop rows where IV could not be calculated (NaN)
    options_data = options_data.dropna(subset=["iv"])
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

    # Check if we have enough data points for B-spline fitting
    if len(x) < 4:
        raise CalculationError(
            f"Insufficient data for B-spline fitting: need at least 4 points, got {len(x)}"
        )

    # fit the bspline using scipy.interpolate.splrep, with k=3
    """
    Bspline Parameters:
        t = the vector of knots
        c = the B-spline coefficients
        k = the degree of the spline
    """
    try:
        tck = interpolate.splrep(x, y, s=10, k=3)
    except Exception as e:
        raise CalculationError(
            f"Failed to fit B-spline to implied volatility data: {str(e)}"
        )

    dx = 0.1  # setting dx = 0.1 for numerical differentiation
    domain = int((max(x) - min(x)) / dx)

    # compute (x,y) observations of the denoised IV from the fitted IV function
    x_new = np.linspace(min(x), max(x), domain)
    y_fit = interpolate.BSpline(*tck)(x_new)

    return (x_new, y_fit)


def _create_pdf_point_arrays(
    denoised_iv: tuple,
    current_price: float,
    days_forward: int,
    risk_free_rate: float,
    dividend_yield: float | None = None,
    pricing_engine: str = "bs",
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
    years_forward = days_forward / 365

    # Use the selected pricing engine (defaults to BS)
    price_fn = get_pricer(pricing_engine)
    q = dividend_yield or 0.0
    interpolated = price_fn(current_price, x_IV, y_IV, years_forward, risk_free_rate, q)

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


def _bs_iv_brent_method(price, S, K, t, r, q=0.0):
    """
    Computes the implied volatility (IV) of a European call option using Brent's method.

    This function finds the implied volatility by solving for sigma (volatility) in the
    Black-Scholes pricing formula. It uses Brent's root-finding algorithm to find the
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
        ValueError: If Brent's method fails to find a root in the given range.

    Notes:
        - The function searches for IV within the range [1e-6, 5.0] (0.0001% to 500% volatility).
        - If `t <= 0`, the function returns NaN since volatility is undefined for expired options.
        - If the function fails to converge, it returns NaN instead of raising an exception.
    """

    if t <= 0:
        return np.nan  # No volatility if time is zero or negative

    try:
        from oipd.pricing.european import european_call_price as _bs_price

        return brentq(lambda iv: _bs_price(S, K, iv, t, r, q) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan  # Return NaN if no solution is found


def _bs_iv_newton_method(
    price: float,
    S: float,
    K: float,
    t: float,
    r: float,
    q: float = 0.0,
    precision: float = 1e-4,
    initial_guess: float | None = None,
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

    from oipd.pricing.european import (
        european_call_price as _bs_price,
        european_call_vega as _bs_vega,
    )

    for i in range(max_iter):
        # Compute model price and Vega
        P = _bs_price(S, K, iv, t, r, q)
        diff = price - P

        # Check for convergence
        if abs(diff) < precision:
            return iv

        grad = _bs_vega(S, K, iv, t, r, q)

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
