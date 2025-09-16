from typing import Dict, Tuple, Literal

import numpy as np
import warnings
from pandas import concat, DataFrame
from scipy.integrate import simpson
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from oipd.pricing import get_pricer
from oipd.pricing.black_scholes import (
    black_scholes_call_price as _bs_price,
    black_scholes_call_vega as _bs_vega,
)
from oipd.pricing.black76 import black76_call_price as _b76_price


class OIPDError(Exception):
    """Base exception for OIPD package"""

    pass


class InvalidInputError(OIPDError):
    """Exception raised for invalid input parameters"""

    pass


class CalculationError(OIPDError):
    """Exception raised when calculations fail"""

    pass


"""
Core routines for computing option-implied PDF/CDF and related helpers.
"""


def calculate_pdf(
    options_data: DataFrame,
    underlying_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    solver_method: str,
    pricing_engine: str,
    dividend_yield: float | None,
    price_method: str,
    forward_price: float | None = None,
) -> Tuple[np.ndarray]:
    """The main execution path for the pdf module. Takes a `DataFrame` of
    options data as input and makes a series of function calls to

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        underlying_price: current price of the instrument — cash spot S for
            Black–Scholes or current futures price F for Black‑76
        days_to_expiry: the number of days in the future to estimate the
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

    if underlying_price <= 0:
        raise InvalidInputError(
            f"underlying_price must be positive, got {underlying_price}"
        )

    if days_to_expiry <= 0:
        raise InvalidInputError(
            f"days_to_expiry must be positive, got {days_to_expiry}"
        )

    if not -1 <= risk_free_rate <= 1:
        raise InvalidInputError(
            f"risk_free_rate seems unrealistic: {risk_free_rate}. Expected value between -1 and 1"
        )

    if solver_method not in ["newton", "brent"]:
        raise InvalidInputError(
            f"solver_method must be 'newton' or 'brent', got '{solver_method}'"
        )

    if pricing_engine == "black76" and forward_price is None:
        raise InvalidInputError(
            "forward_price must be provided when using Black-76 pricing"
        )

    # options_data, min_strike, max_strike = _extrapolate_call_prices(
    #     options_data, underlying_price
    # )
    min_strike = int(options_data.strike.min())
    max_strike = int(options_data.strike.max())

    # this step calculates the mid or last price depending on user argument
    options_data = _calculate_price(options_data, price_method)

    if options_data.empty:
        raise CalculationError("No valid options data after price calculation")

    effective_underlying = (
        forward_price if pricing_engine == "black76" else underlying_price
    )
    options_data = _calculate_IV(
        options_data,
        effective_underlying,
        days_to_expiry,
        risk_free_rate,
        solver_method,
        pricing_engine,
        dividend_yield=dividend_yield,
    )

    if options_data.empty:
        raise CalculationError("Failed to calculate implied volatility for any options")

    denoised_iv = _fit_bspline_IV(options_data)
    underlying_for_pricing = (
        forward_price if pricing_engine == "black76" else underlying_price
    )
    pdf = _create_pdf_point_arrays(
        denoised_iv,
        underlying_for_pricing,
        days_to_expiry,
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
    options_data: DataFrame, underlying_price: float
) -> tuple[DataFrame, int, int]:
    """Extrapolate the price of the call options to strike prices outside
    the range of options_data. Extrapolation is done to zero and twice the
    highest strike price in options_data. Done to give the resulting PDF
    more stability.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        underlying_price: the current price of the instrument

    Returns:
        the extended options_data DataFrame
    """
    min_strike = int(options_data.strike.min())
    max_strike = int(options_data.strike.max())
    lower_extrapolation = DataFrame(
        {"strike": p, "last_price": underlying_price - p} for p in range(0, min_strike)
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


def _calculate_price(options_data: DataFrame, price_method: str) -> DataFrame:
    """Select the appropriate price column based on the specified method.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price', 'mid'] (from put-call parity processing)
        price_method: Method to calculate price - 'last' or 'mid'

    Returns:
        the options_data DataFrame, with a 'price' column containing the selected prices
    """
    options_data = options_data.copy()

    if price_method == "mid":
        if "mid" in options_data.columns:
            options_data["price"] = options_data["mid"]
        elif "bid" in options_data.columns and "ask" in options_data.columns:
            mid = (options_data["bid"] + options_data["ask"]) / 2
            mask = options_data["bid"].notna() & options_data["ask"].notna()
            if mask.any():
                options_data["price"] = np.where(mask, mid, options_data["last_price"])
                if not mask.all():
                    warnings.warn(
                        "Using last_price for rows with missing bid/ask",
                        UserWarning,
                    )
            else:
                warnings.warn(
                    "Requested price_method='mid' but bid/ask data not available. "
                    "Falling back to price_method='last'",
                    UserWarning,
                )
                options_data["price"] = options_data["last_price"]
        else:
            warnings.warn(
                "Requested price_method='mid' but bid/ask data not available. "
                "Falling back to price_method='last'",
                UserWarning,
            )
            options_data["price"] = options_data["last_price"]
    else:  # "last"
        options_data["price"] = options_data["last_price"]

    options_data = options_data[options_data["price"] > 0].copy()
    return options_data


def _calculate_IV(
    options_data: DataFrame,
    underlying_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    solver_method: Literal["newton", "brent"],
    pricing_engine: str,
    dividend_yield: float | None = None,
) -> DataFrame:
    """
    Vectorised implied volatility solver.
    """
    years_to_expiry = days_to_expiry / 365

    prices_arr = options_data["price"].values
    strikes_arr = options_data["strike"].values

    if pricing_engine == "black76":
        iv_values = np.fromiter(
            (
                _black76_iv_brent_method(
                    p, underlying_price, k, years_to_expiry, r=risk_free_rate
                )
                for p, k in zip(prices_arr, strikes_arr)
            ),
            dtype=float,
        )
    else:
        if solver_method == "newton":
            iv_solver_scalar = _bs_iv_newton_method
        elif solver_method == "brent":
            iv_solver_scalar = _bs_iv_brent_method
        else:
            raise ValueError("Invalid solver_method. Choose either 'newton' or 'brent'.")

        q = dividend_yield
        iv_values = np.fromiter(
            (
                iv_solver_scalar(
                    p, underlying_price, k, years_to_expiry, r=risk_free_rate, q=q
                )
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

    TODO: Update with the new put-call parity preprocessing (mid price accepted as well as last price)
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


def finite_diff_second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculate second derivative using 5-point stencil for improved numerical stability.

    This replaces the unstable np.gradient approach with a more robust finite difference
    method that's particularly important for deep out-of-the-money options where
    the Breeden-Litzenberger formula becomes numerically challenging.

    For interior points: f''(x) = [-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)] / (12h²)
    For boundary points: Uses lower-order accurate formulas

    Args:
        y: Function values (option prices)
        x: Grid points (strikes)

    Returns:
        Second derivative values at each grid point

    Raises:
        ValueError: If grid spacing is non-uniform or arrays have insufficient length
    """
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length. Got x: {len(x)}, y: {len(y)}")

    if len(x) < 5:
        raise ValueError(f"Need at least 5 points for 5-point stencil. Got {len(x)}")

    # Check for uniform grid spacing (required for finite differences)
    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-6):
        # For non-uniform grids, fall back to np.gradient but warn user
        import warnings

        warnings.warn(
            "Non-uniform grid detected. Using np.gradient fallback which may be less stable. "
            "Consider interpolating to uniform grid first.",
            UserWarning,
        )
        return np.gradient(np.gradient(y, x), x)

    h = h[0]  # Grid spacing
    d2y = np.zeros_like(y)

    # Interior points using 5-point stencil (4th order accurate)
    for i in range(2, len(y) - 2):
        d2y[i] = (-y[i - 2] + 16 * y[i - 1] - 30 * y[i] + 16 * y[i + 1] - y[i + 2]) / (
            12 * h**2
        )

    # Boundary points using forward/backward differences (2nd order accurate)
    # Left boundary (first two points)
    d2y[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / h**2
    d2y[1] = (y[0] - 2 * y[1] + y[2]) / h**2

    # Right boundary (last two points)
    d2y[-2] = (y[-3] - 2 * y[-2] + y[-1]) / h**2
    d2y[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / h**2

    return d2y


def _create_pdf_point_arrays(
    denoised_iv: tuple,
    underlying_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    dividend_yield: float | None,
    pricing_engine: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two arrays containing x- and y-axis values representing a calculated
    price PDF

    Args:
        denoised_iv: (x,y) observations of the denoised IV
        underlying_price: the current price of the instrument
        days_to_expiry: the number of days in the future to estimate the
            price probability density at
        risk_free_rate: the current annual risk free interest rate, nominal terms

    Returns:
        a tuple containing x-axis values (index 0) and y-axis values (index 1)
    """

    # extract the x and y vectors from the denoised IV observations
    x_IV = denoised_iv[0]
    y_IV = denoised_iv[1]

    # convert IV-space to price-space
    years_to_expiry = days_to_expiry / 365

    # Use the selected pricing engine (defaults to BS)
    price_fn = get_pricer(pricing_engine)
    q = dividend_yield or 0.0
    interpolated = price_fn(
        underlying_price, x_IV, y_IV, years_to_expiry, risk_free_rate, q
    )

    # Use stable finite difference method instead of np.gradient for second derivatives
    # This is critical for numerical stability, especially for deep OTM options
    second_derivative_discrete = finite_diff_second_derivative(interpolated, x_IV)

    # apply coefficient to reflect the time value of money
    pdf = np.exp(risk_free_rate * years_to_expiry) * second_derivative_discrete

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
        return brentq(lambda iv: _bs_price(S, K, iv, t, r, q) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan  # Return NaN if no solution is found


def _black76_iv_brent_method(price, F, K, t, r):
    """Implied volatility for Black-76 using Brent's method with bounds."""
    if t <= 0:
        return np.nan

    df = np.exp(-r * t)
    lower = df * max(F - K, 0.0)
    upper = df * F
    if price < lower or price > upper:
        return np.nan

    try:
        return brentq(lambda iv: _b76_price(F, K, iv, t, r, 0.0) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan


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
