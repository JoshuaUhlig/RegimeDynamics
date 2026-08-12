import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import os
import sys

import matplotlib.cm as cm
from pythonplot import *


main_color = cm.plasma(0.5)
second_color = cm.plasma(0.1)
third_color = cm.plasma(0.9)


def tamsd(arr):
    T = int(arr.shape[0]) - 1
    TAMSD = np.zeros(T)
    for dt in range(1, T + 1):
        TAMSD[dt - 1] = np.sum(
            np.mean((arr[dt:, :] - arr[:-dt, :]) ** 2, axis=0), axis=0
        )
    return (np.arange(1, T + 1)), TAMSD


def tamsd_data(arr):
    T = int(arr.shape[0]) - 1
    TAMSD = np.zeros(T)
    for dt in range(1, T + 1):
        TAMSD[dt - 1] = np.sum(
            np.mean((arr[dt:, 1:] - arr[:-dt, 1:]) ** 2, axis=0), axis=0
        )
    return (np.arange(1, T + 1)), TAMSD


def create_tamsd_plot():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size(subplots=(1, 2), ratio=3 / 4),
    )
    ax, ax_scatter = axes
    ax.tick_params(top=True, right=True, direction="in", which="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(
        r"$\overline{\delta^2(\Delta)}$ and $\left\langle\overline{\delta^2(\Delta)}\right\rangle$"
    )
    ax.set_xlabel(r"Lag time $\Delta$ in years")
    trajectory = np.loadtxt("sim_trajectories/1.dat")
    T = trajectory.shape[0] - 1
    avgtamsd = np.zeros(T)
    counter = 0
    path = "sim_trajectories/"
    for count, file in enumerate(os.listdir(path)):
        filepath = path + file
        trajectory = np.loadtxt(filepath)
        t, msd = tamsd(trajectory)
        avgtamsd = avgtamsd + msd
        counter = counter + 1
        if count % 100 == 0:
            ax.plot(t, msd, alpha=0.1, color="black")
    avgtamsd = avgtamsd / counter
    ax.plot(t, avgtamsd, "s", color="black", label=r"CTRW Model")
    path = "all_trajectories/"
    startyear = 1990
    min_length = 30
    TAMSD = np.zeros((min_length))
    counter = 0
    nr_trajectories = 0
    for filename in os.listdir(path):
        file = path + filename
        if os.path.isfile(file):
            trajectory = np.loadtxt(file)
            if trajectory[0, 0] <= startyear and trajectory[-1, 0] > startyear:
                startindex = np.where(trajectory[:, 0] == startyear)[0][0]
                if trajectory[startindex:, :].shape[0] >= min_length:
                    trajectory = trajectory[startindex : startindex + min_length + 1, :]
                    nr_trajectories = nr_trajectories + 1
                    t, msd = tamsd_data(trajectory)
                    TAMSD = TAMSD + msd
                    ax.plot(t, msd, color=main_color, alpha=0.1)
                    counter = counter + 1
    TAMSD = TAMSD / counter
    ax.plot(
        np.arange(1, min_length + 1),
        TAMSD,
        "o",
        markersize=6,
        color=main_color,
        label=r"V-Dem Data",
    )
    startyear_xi_scatter(
        startyear=startyear,
        min_length=min_length,
        fig=fig,
        ax=ax_scatter,
        sim_path="sim_trajectories/",
        data=True,
        color=False,
    )
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=2,
    )
    plt.savefig("figures/tamsd_data_ctrw.pdf", bbox_inches="tight")
    plt.close()


def startyear_xi_scatter(startyear, min_length, fig, ax, sim_path, data, color):
    # xi

    path = "all_trajectories/"
    avgtamsd = np.zeros(min_length)
    nr_trajectories = 0
    for dt in range(1, min_length + 1):
        dx = []
        dy = []
        for count, filename in enumerate(os.listdir(path)):
            file = path + filename
            if os.path.isfile(file):
                trajectory = np.loadtxt(file)
                if (
                    trajectory[0, 0] <= startyear
                    and trajectory[-1, 0] > startyear
                    and trajectory.shape[0] > dt
                ):
                    startindex = np.where(trajectory[:, 0] == startyear)[0][0]  #
                    if trajectory[startindex:, :].shape[0] >= min_length:
                        if dt == 1:
                            nr_trajectories = nr_trajectories + 1
                        trajectory = trajectory[
                            startindex : startindex + min_length + 1, :
                        ]
                        dx = np.append(
                            dx,
                            ((trajectory[dt:, 1] - trajectory[:-dt, 1]) ** 2).flatten(),
                        )
                        dy = np.append(
                            dy,
                            ((trajectory[dt:, 2] - trajectory[:-dt, 2]) ** 2).flatten(),
                        )
        avgtamsd[dt - 1] = np.mean(dx) + np.mean(dy)
    xi = np.zeros((min_length, nr_trajectories))
    k = 0
    for count, filename in enumerate(os.listdir(path)):
        file = path + filename
        if os.path.isfile(file):
            trajectory = np.loadtxt(file)
            if trajectory[0, 0] <= startyear and trajectory[-1, 0] > startyear:
                startindex = np.where(trajectory[:, 0] == startyear)[0][0]
                if trajectory[startindex:, :].shape[0] >= min_length:
                    t, msd = tamsd_data(
                        trajectory[startindex : startindex + min_length + 1, :]
                    )
                    xi[:, k] = msd
                    k = k + 1
    for j in range(min_length - 1):
        xi[j, :] = xi[j, :] / avgtamsd[j]
    deltas = np.array([1, 5, 10, 15, 20, 30, 40, 50])
    deltas = deltas[deltas < min_length] - 1
    hist, bins = np.histogram(xi[0, :], bins="fd", density=True)
    bin_cent = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    avg_hist = hist
    for delta in np.arange(1, 10):
        hist, _ = np.histogram(xi[delta - 1, :], bins=bins, density=True)
        avg_hist = np.append(avg_hist, hist)
    avg_hist = np.reshape(avg_hist, (10, len(bin_cent)))
    dhist = np.std(avg_hist, axis=0)
    avg_hist = np.mean(avg_hist, axis=0)
    avg_hist_data = avg_hist
    dhist_data = dhist
    # avg hist +
    ax.tick_params(top=True, right=True, direction="in", which="both")
    if data and color:
        ax.plot(bin_cent, avg_hist, "o", mfc="None", color="black")
    elif data:
        ax.errorbar(bin_cent, avg_hist, yerr=dhist, fmt="o", color=main_color)

    # Simulations
    path = sim_path
    avgtamsd = np.zeros(min_length)
    nr_trajectories = 10000
    xi = np.zeros((min_length, nr_trajectories))
    for count, filename in enumerate(os.listdir(path)):
        file = path + filename
        if os.path.isfile(file):
            trajectory = np.loadtxt(file)
            msd = np.zeros(min_length)
            for dt in range(1, min_length + 1):
                msd[dt - 1] = np.sum(
                    np.mean((trajectory[dt:, :] - trajectory[:-dt, :]) ** 2, axis=0),
                    axis=0,
                )
            avgtamsd = avgtamsd + msd
            xi[:, count] = msd
    avgtamsd = avgtamsd / nr_trajectories
    # back to xi
    for j in range(min_length - 1):
        xi[j, :] = xi[j, :] / avgtamsd[j]
    hist, bins = np.histogram(xi[0, :], bins=bins, density=True)
    bin_cent = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    ax.set_xlabel(
        r"$\xi=\overline{\delta^2}/\left\langle\overline{\delta^2}\right\rangle$"
    )
    ax.set_ylabel(r"PDF $\phi(\xi)$")
    ax.set_yscale("log")
    avg_hist = hist
    for delta in np.arange(1, 10):
        hist, _ = np.histogram(xi[delta - 1, :], bins=bins, density=True)
        avg_hist = np.append(avg_hist, hist)
    avg_hist = np.reshape(avg_hist, (10, len(bin_cent)))
    dhist = np.std(avg_hist, axis=0)
    avg_hist = np.mean(avg_hist, axis=0)
    if not color:
        ax.errorbar(
            bin_cent, avg_hist, yerr=dhist, fmt="s", color="black", label="CTRW"
        )
    else:
        ax.plot(bin_cent, avg_hist, color=color)


def create_tamsd_comparison_plot():
    fig, axes = plt.subplots(
        1, 2, figsize=figure_size("aps_single", subplots=(1, 2), ratio=3 / 4)
    )
    startyear = 1990
    min_length = 30
    ax, ax_scatter = axes
    ax.tick_params(top=True, right=True, direction="in", which="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\left\langle\overline{\delta^2(\Delta)}\right\rangle$")
    ax.set_xlabel(r"Lag time $\Delta$ in years")
    trajectory = np.loadtxt("sim_trajectories/1.dat")
    T = trajectory.shape[0] - 1
    avgtamsd = np.zeros(T)
    counter = 0
    paths = ["sim_trajectories_lb/", "sim_trajectories/", "sim_trajectories_ub/"]
    labels = ["PC$_c=0.03$", "PC$_c=0.1$", "PC$_c=0.25$"]
    colors = [cm.plasma(0.1), cm.plasma(0.4), cm.plasma(0.8)]
    plot_data = True
    for i, path in enumerate(paths):
        startyear_xi_scatter(
            startyear=startyear,
            min_length=min_length,
            fig=fig,
            ax=ax_scatter,
            sim_path=path,
            data=plot_data,
            color=colors[i],
        )
        plot_data = False
        avgtamsd = np.zeros(T)
        counter = 0
        for count, file in enumerate(os.listdir(path)):
            filepath = path + file
            trajectory = np.loadtxt(filepath)
            t, msd = tamsd(trajectory)
            avgtamsd = avgtamsd + msd
            counter = counter + 1
        avgtamsd = avgtamsd / counter
        ax.plot(t, avgtamsd, color=colors[i], label=labels[i])
    path = "all_trajectories/"
    startyear = 1990
    min_length = 30
    TAMSD = np.zeros(min_length)
    counter = 0
    nr_trajectories = 0
    for filename in os.listdir(path):
        file = path + filename
        if os.path.isfile(file):
            trajectory = np.loadtxt(file)
            if trajectory[0, 0] <= startyear and trajectory[-1, 0] > startyear:
                startindex = np.where(trajectory[:, 0] == startyear)[0][0]
                if trajectory[startindex:, :].shape[0] >= min_length:
                    trajectory = trajectory[startindex : startindex + min_length + 1, :]
                    nr_trajectories = nr_trajectories + 1
                    t, msd = tamsd_data(trajectory)
                    TAMSD = TAMSD + msd
                    # ax.plot(t,msd,color=main_color,alpha=0.1)
                    counter = counter + 1
    TAMSD = TAMSD / counter
    ax.plot(
        np.arange(1, min_length + 1),
        TAMSD,
        "o",
        mfc="None",
        color="black",
        label=r"V-Dem Data",
    )
    fig.legend(frameon=False, loc="outside lower center", ncols=4)
    fig.tight_layout()
    plt.savefig("figures/tamsd_data_ctrw_cutoffs.pdf", bbox_inches="tight")
    plt.close()


def plot_increment_autocorrelation(
    startyear=1990,
    min_length=30,
    sim_path="sim_trajectories/",
    data_path="all_trajectories/",
):
    """
    Increment autocorrelation C(tau) = <v(t) v(t+tau)> / <v(t)^2>,
    where v(t) = r(t+1) - r(t) are the annual increments.
    Computed per trajectory then ensemble-averaged.
    """
    max_lag = min_length // 2
    lags = np.arange(0, max_lag + 1)

    def traj_autocorr(v):
        """Normalised time-averaged autocorrelation for a single increment series."""
        c0 = np.mean(v**2)
        if c0 == 0:
            return None
        return np.array([np.mean(v[: len(v) - lag] * v[lag:]) / c0 for lag in lags])

    # collect per-trajectory autocorrelations
    data_ac1, data_ac2 = [], []
    for filename in os.listdir(data_path):
        filepath = data_path + filename
        if not os.path.isfile(filepath):
            continue
        traj = np.loadtxt(filepath)
        if traj[0, 0] > startyear or traj[-1, 0] <= startyear:
            continue
        startindex = np.where(traj[:, 0] == startyear)[0][0]
        if traj[startindex:].shape[0] < min_length:
            continue
        traj = traj[startindex : startindex + min_length + 1, :]
        ac1 = traj_autocorr(np.diff(traj[:, 1]))
        ac2 = traj_autocorr(np.diff(traj[:, 2]))
        if ac1 is not None:
            data_ac1.append(ac1)
        if ac2 is not None:
            data_ac2.append(ac2)

    sim_ac1, sim_ac2 = [], []
    for filename in os.listdir(sim_path):
        filepath = sim_path + filename
        if not os.path.isfile(filepath):
            continue
        traj = np.loadtxt(filepath)
        ac1 = traj_autocorr(np.diff(traj[:, 0]))
        ac2 = traj_autocorr(np.diff(traj[:, 1]))
        if ac1 is not None:
            sim_ac1.append(ac1)
        if ac2 is not None:
            sim_ac2.append(ac2)

    data_ac1, data_ac2 = np.array(data_ac1), np.array(data_ac2)
    sim_ac1, sim_ac2 = np.array(sim_ac1), np.array(sim_ac2)

    # plot
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=set_size(ratio=3 / 4, subplots=(1, 2)),
        sharey=True,
    )

    for ax, d_ac, s_ac, xlabel in [
        (ax1, data_ac1, sim_ac1, r"${d\mathrm{PC}1}$"),
        (ax2, data_ac2, sim_ac2, r"${d\mathrm{PC}2}$"),
    ]:
        d_mean = np.mean(d_ac, axis=0)
        d_err = np.std(d_ac, axis=0) / np.sqrt(len(d_ac))
        s_mean = np.mean(s_ac, axis=0)
        s_err = np.std(s_ac, axis=0) / np.sqrt(len(s_ac))

        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
        ax.errorbar(
            lags, s_mean, yerr=s_err, fmt="s", color="black", label="CTRW Model"
        )
        ax.errorbar(
            lags, d_mean, yerr=d_err, fmt="o", color=main_color, label="V-Dem Data"
        )

        ax.set_xlabel(r"Lag $\Delta$ (years)")
        ax1.set_ylabel(r"$C_{\mathrm{PC}1}$" + r"$(\Delta)$")
        ax2.set_ylabel(r"$C_{\mathrm{PC}2}$" + r"$(\Delta)$")
        ax.tick_params(top=True, right=True, direction="in", which="both")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.6),
        ncols=2,
        fontsize=10,
    )
    plt.savefig("figures/increment_autocorrelation.pdf", bbox_inches="tight")
    plt.close()


create_tamsd_plot()
create_tamsd_comparison_plot()
plot_increment_autocorrelation()
