import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from lmfit import Model
from scipy.optimize import curve_fit
from scipy.special import gamma
import math
from pymittagleffler import mittag_leffler
import matplotlib.patches as patches

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "font.size": 12,
        "errorbar.capsize": 2,
    }
)
magenta = "#D105D1"
lblue = "#007EE3"


def set_size(width=418.25555, fraction=1, subplots=(1, 1), ratio=1 / 1.61803398875):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def turn_off_autoscale(ax):
    xlim = ax.set_xlim()
    ax.set_xlim(xlim)
    ylim = ax.set_ylim()
    ax.set_ylim(ylim)


def arrowplot(
    axes, x, y, nArrs=30, mutateSize=10, color="gray", markerstyle="o", label="Nöthing"
):
    """arrowplot : plots arrows along a path on a set of axes
    axes   :  the axes the path will be plotted on
    x      :  list of x coordinates of points defining path
    y      :  list of y coordinates of points defining path
    nArrs  :  Number of arrows that will be drawn along the path
    mutateSize :  Size parameter for arrows
    color  :  color of the edge and face of the arrow head
    markerStyle : Symbol

    Bugs: If a path is straight vertical, the matplotlab FanceArrowPatch bombs out.
      My kludge is to test for a vertical path, and perturb the second x value
      by 0.1 pixel. The original x & y arrays are not changed

    MHuster 2016, based on code by
    """
    # recast the data into numpy arrays
    x = np.array(x, dtype="f")
    y = np.array(y, dtype="f")
    nPts = len(x)

    # Plot the points first to set up the display coordinates
    axes.plot(x, y, markerstyle, ms=5, color=color, label=label)

    # get inverse coord transform
    inv = axes.transData.inverted()

    # transform x & y into display coordinates
    # Variable with a 'D' at the end are in display coordinates
    xyDisp = np.array(axes.transData.transform(list(zip(x, y))))
    xD = xyDisp[:, 0]
    yD = xyDisp[:, 1]

    # drD is the distance spanned between pairs of points
    # in display coordinates
    dxD = xD[1:] - xD[:-1]
    dyD = yD[1:] - yD[:-1]
    drD = np.sqrt(dxD**2 + dyD**2)

    # Compensating for matplotlib bug
    dxD[np.where(dxD == 0.0)] = 0.1

    # rtotS is the total path length
    rtotD = np.sum(drD)

    # based on nArrs, set the nominal arrow spacing
    arrSpaceD = rtotD / nArrs

    # Loop over the path segments
    iSeg = 0
    while iSeg < nPts - 1:
        # Figure out how many arrows in this segment.
        # Plot at least one.
        nArrSeg = max(1, int(drD[iSeg] / arrSpaceD + 0.5))
        xArr = (dxD[iSeg]) / nArrSeg  # x size of each arrow
        segSlope = dyD[iSeg] / dxD[iSeg]
        # Get display coordinates of first arrow in segment
        xBeg = xD[iSeg]
        xEnd = xBeg + xArr
        yBeg = yD[iSeg]
        yEnd = yBeg + segSlope * xArr
        # Now loop over the arrows in this segment
        for iArr in range(nArrSeg):
            # Transform the oints back to data coordinates
            xyData = inv.transform(((xBeg, yBeg), (xEnd, yEnd)))
            # Use a patch to draw the arrow
            # I draw the arrows with an alpha of 0.5
            p = patches.FancyArrowPatch(
                xyData[0],
                xyData[1],
                arrowstyle="simple",
                mutation_scale=mutateSize,
                # color=color,
                facecolor=color,
                edgecolor="black",
                alpha=0.5,
            )
            axes.add_patch(p)
            # Increment to the next arrow
            xBeg = xEnd
            xEnd += xArr
            yBeg = yEnd
            yEnd += segSlope * xArr
        # Increment segment number
        iSeg += 1


def powerlaw(x, alpha, amp):
    return -(alpha + 1) * x + amp


def lin_exp(x, k, x0, amp):
    return amp - k * np.abs(x - x0)


def exp(x, k, amp):
    return amp * np.exp(-k * np.abs(x))


def lin_powerlaw_exp(x, k, amp, alpha):
    return amp - (1 + alpha) * np.log(x) - k * x


def gaus(x, amp, sigma):
    return amp * np.exp(-((x) ** 2) / (2 * sigma**2))


def lin_gaus(x, amp, x0, sigma):
    return amp / np.sqrt(2 * np.pi * sigma**2) - (x - x0) ** 2 / (2 * sigma**2)


def lin_halfgaus(x, sigma):
    return 1 / np.sqrt(np.pi * sigma**2 / 2) - (x) ** 2 / (2 * sigma**2)


def double_gaus(x, amp1, amp2, sigma1, sigma2):
    return amp1 / np.sqrt(2 * np.pi * sigma1**2) * np.exp(
        -((x) ** 2) / (2 * sigma1**2)
    ) + amp2 / np.sqrt(2 * np.pi * sigma2**2) * np.exp(-((x) ** 2) / (2 * sigma2**2))


def ml(x, alpha, amp):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = np.abs(mittag_leffler(x[i], alpha, 1))
    return y


def phi_series(xi, alpha):
    threshold = (1e-10,)
    if xi == 0 or alpha == 0:
        raise ValueError("xi and alpha must be non-zero.")
    gamma_1_alpha = gamma(1 + alpha)
    series_sum = 0.0
    n = 1
    while True:
        try:
            term_num = (-1) ** n * (gamma_1_alpha / xi) ** n
            term_den = math.factorial(n) * gamma(-alpha * n)
            term = term_num / term_den
        except (OverflowError, ZeroDivisionError, ValueError):
            print(alpha)
            print(f"Term {n}={term} caused an error. Stopping early.")
            break

        series_sum += term

        if abs(term) < threshold:
            break
        n += 1
    return (1 / (alpha * xi)) * series_sum


def phi_alpha(x, alpha):
    phi = np.zeros_like(x)
    for i, xi in enumerate(x):
        phi[i] = phi_series(xi, alpha)
    return phi


def phi_alpha_fit(x, y, ax=None, color="black", alpha0=0.5):
    model = Model(phi_alpha)
    model.set_param_hint("alpha", min=0, max=1)
    params = model.make_params()
    result = model.fit(y, params=params, x=x, alpha=alpha0)
    alpha = 0
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = result.eval(x=x_eval)
        alpha = result.best_values["alpha"]
        ax.plot(x_eval, y_eval, color=color, label=r"$\alpha=$" + str(round(alpha, 9)))
    return alpha


def ml_fit(x, y, ax=None, color="black", alpha0=1):
    model = Model(ml)
    model.set_param_hint("alpha", min=0, max=100)
    params = model.make_params()
    result = model.fit(y, params=params, x=x, alpha=alpha0, amp=1.0)
    alpha = 0
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = result.eval(x=x_eval)
        alpha = result.best_values["alpha"]
        ax.plot(x_eval, y_eval, color=color, label=r"$\alpha=$" + str(round(alpha, 9)))
    return alpha


def powerlaw_fit(
    x,
    y,
    ax=None,
    color="tab:red",
    ls="dashed",
    error=False,
    alphacomp=None,
    variable="lagtime",
):
    param, pcov = curve_fit(
        powerlaw, np.log10(x), np.log10(y)
    )  # ,sigma=np.log10(np.sqrt(y)))
    xplot = 10 ** np.linspace(np.log10(x.min()), np.log10(x.max()))
    if ax != None and variable == "lagtime":
        ax.plot(
            xplot,
            10 ** param[1] * xplot ** (-1 - param[0]),
            color=color,
            ls=ls,
            label=r"$\propto\Delta" + "^{" + f"{-round(1 + param[0], 2)}" + "}$",
        )
    elif variable == "time":
        ax.plot(
            xplot,
            10 ** param[1] * xplot ** (-1 - param[0]),
            color=color,
            ls=ls,
            label=r"$\propto\tau" + "^{" + f"{-round(1 + param[0], 2)}" + "}$",
        )
    elif alphacomp != None:
        A = (1 ** (-param[0]) - 5 ** (-param[0])) / param[0]
        ax.plot(
            xplot,
            1 / A * xplot ** (-1 - param[0]),
            color=color,
            ls="dashed",
            label=r"$\alpha=" + str(round(param[0], 2)) + "$",
        )
        A = (1 ** (-alphacomp) - 5 ** (-alphacomp)) / alphacomp
        ax.plot(
            xplot,
            1 / A * xplot ** (-1 - alphacomp),
            color=color,
            ls="dotted",
            label=r"$\alpha=" + str(alphacomp) + "$",
        )
    if error:
        ax.plot(
            xplot,
            10 ** param[1] * xplot ** (-1 - param[0] - pcov[0][0]),
            color=color,
            ls="dashed",
        )
        ax.plot(
            xplot,
            10 ** param[1] * xplot ** (-1 - param[0] + pcov[0][0]),
            color=color,
            ls="dashed",
        )
    return param, pcov


def double_gaus_fit(x, y, ax=None):
    model = Model(double_gaus)
    params = model.make_params()
    result = model.fit(
        y, params=params, x=x, amp1=np.max(y), amp2=np.min(y), sigma1=1, sigma2=10
    )
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = result.eval(x=x_eval)
        ax.plot(x_eval, y_eval, color="tab:red", ls="dashed")
    return result


def lin_powerlaw_exp_fit(x, y, color="tab:red", ax=None):
    model = Model(lin_powerlaw_exp)
    params = model.make_params()
    result = model.fit(
        np.log(y), params=params, x=x, amp=np.log(np.max(y)), alpha=0.5, k=1
    )
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = np.exp(result.eval(x=x_eval))
        alpha = result.best_values["alpha"]
        k = result.best_values["k"]
        ax.plot(
            x_eval,
            y_eval,
            color=color,
            ls="dashed",
            label=r"$\propto\tau"
            + "^{"
            + f"{-round(1 + alpha, 2)}"
            + "}$"
            + "e$^{-"
            + f"{round(k, 2)}"
            + r"\tau}$",
        )
    return result


def exp_fit(x, y, ax=None):
    model = Model(exp)
    params = model.make_params()
    result = model.fit(y, params=params, x=x, amp=np.max(y), k=1)
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = result.eval(x=x_eval)
        ax.plot(x_eval, y_eval, color="tab:red")
    return result


def lin_exp_fit(x, y, ax=None):
    model = Model(lin_exp)
    params = model.make_params()
    result = model.fit(
        np.log(y), params=params, x=x, amp=np.log(np.max(y)), x0=0.1, k=1
    )
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = np.exp(result.eval(x=x_eval))
        k = result.best_values["k"]
        x0 = result.best_values["x0"]
        ax.plot(
            x_eval,
            y_eval,
            color="tab:red",
            label=r"$\propto$e$^{-"
            + f"{round(k, 2)}("
            + r"x-"
            + str(round(x0, 2))
            + ")}$",
        )
    return result


def gaus_fit(x, y, ax=None):
    model = Model(gaus)
    params = model.make_params()
    result = model.fit(y, params=params, x=x, amp=np.max(y), sigma=1)
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = result.eval(x=x_eval)
        ax.plot(x_eval, y_eval, color="tab:red")
    return result


def lin_gaus_fit(x, y, ax=None):
    model = Model(lin_gaus)
    params = model.make_params()
    result = model.fit(np.log(y), params=params, x=x, amp=np.max(y), x0=0.1, sigma=1)
    if ax != None:
        x_eval = np.logspace(np.log10(x.min()), np.log10(x.max()))
        y_eval = np.exp(result.eval(x=x_eval))
        sigma = result.best_values["sigma"]
        x0 = result.best_values["x0"]
        ax.plot(
            x_eval,
            y_eval,
            color="tab:red",
            ls="dotted",
            label=r"$\propto$ e$^{-(x-"
            + f"{round(x0, 2)}"
            + r")^2/(2\cdot"
            + f"{round(sigma, 2)}"
            + "^2)}$",
        )
    return result


def lin_halfgaus_fit(x, y, ax=None):
    model = Model(lin_halfgaus)
    params = model.make_params()
    result = model.fit(np.log(y), params=params, x=x, sigma=1)
    if ax != None:
        x_eval = np.linspace(x.min(), x.max())
        y_eval = np.exp(result.eval(x=x_eval))
        sigma = result.best_values["sigma"]
        ax.plot(
            x_eval,
            y_eval,
            color="tab:red",
            ls="dotted",
            label=r"$\propto$ e$^{-(\tau)^2/(2\cdot" + f"{round(sigma, 2)}" + "^2)}$",
        )
    return result
