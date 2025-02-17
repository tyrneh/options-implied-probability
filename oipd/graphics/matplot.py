import warnings
from datetime import datetime
from typing import Optional, Tuple, Union

from labellines import labelLine
from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from numpy import linspace, ndarray

from oipd.core import calculate_quartiles

pyplot.rcParams["axes.autolimit_mode"] = "round_numbers"


def generate_pdf_figure(
    density_function: Tuple[ndarray],
    *,
    security_ticker: str,
    expiry_date: datetime,
    current_price: Optional[Union[float, bool]] = False,
) -> Figure:
    fig, ax = pyplot.subplots()
    ax.plot(density_function[0], density_function[1])
    ax.set_title(
        f"Probability Density Function of the price of"
        f"\n{security_ticker} on {expiry_date}"
    )

    # add axis titles
    ax.set_xlabel("Price")
    ax.set_ylabel("Probability")

    # format x-axis
    ax.tick_params(axis="x", which="minor", bottom=False)

    # format y-axis
    ax.set_ylim(bottom=0)

    if current_price:
        label = f"{current_price:.2f}"
        line = ax.axvline(
            x=current_price, color="green", linestyle=":", label=label, linewidth=0.75
        )
        # calculate the offset so that it is centered in the current range
        bottom, top = ax.get_ylim()
        label_y_offset = -0.5 + (top - bottom) * 0.2
        _label_line_no_warnings(line, x=current_price, yoffset=label_y_offset)

    return fig


def generate_cdf_figure(
    density_function: Tuple[ndarray],
    *,
    security_ticker: str,
    expiry_date: datetime,
    current_price: Optional[Union[float, bool]] = False,
    quartiles: Optional[bool] = False,
) -> Figure:
    """Create a Matplotlib Figure object of a CDF

    Useful for drawing a graph using Streamlit

    Returns:
        A Matplotlib Figure object of the generated graph
    """
    fig, ax = pyplot.subplots()
    ax.plot(density_function[0], density_function[1])
    ax.set_title(
        f"Cumulative Density Function of the price of"
        f"\n{security_ticker} on {expiry_date}"
    )

    # add axis titles
    ax.set_xlabel("Price")
    ax.set_ylabel("Probability")

    # format x-axis
    ax.tick_params(axis="x", which="minor", bottom=False)

    # format y-axis
    ax.set_ylim([0, 1])
    ax.set_yticks(linspace(0, 1, 11))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    if current_price:
        label = f"{current_price:.2f}"
        line = ax.axvline(
            x=current_price, color="green", linestyle=":", label=label, linewidth=0.75
        )
        _label_line_no_warnings(line, x=current_price, yoffset=0.35)
    if quartiles:
        quartile_bounds = calculate_quartiles(density_function)
        x_start, x_end = ax.get_xlim()
        y_start, y_end = ax.get_ylim()
        for k, v in quartile_bounds.items():
            xmax = (v - x_start) / (x_end - x_start)
            ymax = (k - y_start) / (y_end - y_start)
            label = f"{v:.2f}"
            line = ax.axvline(
                x=v, ymax=ymax, color="black", linestyle="--", label=label
            )
            _label_line_no_warnings(line, x=v, align=True)
            ax.axhline(y=k, xmax=xmax, color="black", linestyle="--")

    return fig


def _label_line_no_warnings(line, **params) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labelLine(line, **params)
