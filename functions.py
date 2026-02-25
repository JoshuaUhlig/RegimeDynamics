import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pythonplot import *
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter


main_color = cm.plasma(0.5)
second_color = cm.plasma(0.1)
third_color = cm.plasma(0.9)


def extract_all_trajectories(df, features, savepath, min_length=10):
    """Gets trajectories from a dataframe and saves them
    as an array in the format year-features as a country_id.dat

    Keyword arguments:
    df -- pandas dataframe
    features -- array of features to create a dataframe of
    """
    features = ["country_id", "year"] + features
    data = df[features].to_numpy()
    countries = data[:, 0]
    for country in np.unique(countries):
        trajectory = data[countries == country]
        stopyear = trajectory[-1, 1]
        stop_index = -1
        startyear = trajectory[0, 1]
        start_index = 0
        counter = 1
        old_stop_index = 0
        for i in range(trajectory.shape[0] - 1):
            if trajectory[i + 1, 1] - trajectory[i, 1] != 1.0:
                print(f"{country}: {trajectory[i, 1]}-{trajectory[i + 1, 1]}")
                stopyear = trajectory[i, 1]
                old_stop_index = stop_index
                stop_index = i
                if stop_index - start_index >= min_length:
                    np.savetxt(
                        savepath + str(int(country)) + "_" + str(counter) + ".dat",
                        trajectory[start_index : stop_index + 1, 1:],
                    )
                    counter = counter + 1
                startyear = trajectory[i + 1, 1]
                start_index = i + 1
        if trajectory.shape[0] - start_index >= min_length:
            np.savetxt(
                f"{savepath}{int(country)}_{counter}.dat", trajectory[start_index:, 1:]
            )


def create_country_table(df, pathname="vdem/country_table.dat"):
    """creates country id-name table from vdem dataframe

    Keyword arguments:
    df -- pandas dataframe
    """
    table = df[["country_id", "country_name"]].drop_duplicates().to_numpy()
    with open(pathname, "w") as f:
        for i in range(table.shape[0]):
            f.write(str(table[i][0]) + "\t" + table[i][1] + "\n")


def read_country_table(pathname="country_table.dat"):
    return pd.read_csv(pathname, sep="\t", header=None).to_numpy()


def tamsd(arr):
    T = int(arr.shape[0]) - 1
    TAMSD = np.zeros(T)
    for dt in range(1, T + 1):
        TAMSD[dt - 1] = np.sum(
            np.mean((arr[dt:, 1:] - arr[:-dt, 1:]) ** 2, axis=0), axis=0
        )
    return (np.arange(1, T + 1)), TAMSD


def create_tamsd_plot_country(path, countries, ax=None, min_trajec_length=10):
    """
    Create TAMSD plot for given countries on the provided axis.
    If ax is None, a new figure+axis will be created and returned.
    """
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=set_size())
        new_fig = True

    ax.tick_params(top=True, right=True, direction="in", which="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Lag time $\Delta$ in years")
    ax.set_ylabel(r"TAMSD $\overline{\delta^2(\Delta)}$")

    country_dict = dict(read_country_table())
    cmap = matplotlib.colormaps.get_cmap("plasma")
    gradient = np.linspace(0, 1, len(countries))
    colors = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]
    symbols = ["o", "s", "^", "v", "D"]

    # background trajectories
    for filename in os.listdir(path):
        file = path + filename
        country_id = int(filename.split("_")[0])
        if os.path.isfile(file):
            trajectory = np.loadtxt(file)
            if trajectory.shape[0] >= min_trajec_length and (
                country_dict[country_id] not in countries
            ):
                t, msd = tamsd(trajectory)
                ax.plot(t, msd, color="black", alpha=0.1)

    # highlighted countries
    for i, country in enumerate(countries):
        country_id = [j for j in country_dict if country_dict[j] == country][0]
        trajectory = np.loadtxt(f"all_trajectories/{country_id}_1.dat")
        t, msd = tamsd(trajectory)
        ax.plot(t, msd, symbols[i], color=colors[i], label=country_dict[country_id])

    ax.plot(t, 4 * t, color="black", ls="dashed")
    ax.plot(t, 8e-3 * t, color="black", ls="dashed")
    ax.annotate(r"$\propto \Delta$", (5, 60))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    if new_fig:
        ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 0.5))
        return fig, ax
    else:
        return ax


def trajectory_plot(countries, comp1, comp2, df=None, features=None):
    fig, ax = plt.subplots(
        1, 1, figsize=set_size(width=595.80026, ratio=418.25555 / 595.80026)
    )
    ax.tick_params(top=True, right=True, direction="in", which="both")
    ax.set_xlabel(f"Democraticness (PC{comp1})", fontsize=16)
    ax.set_ylabel(f"Election Capability - Civil Liberties (PC{comp2})", fontsize=16)

    country_dict = dict(read_country_table())
    cmap = matplotlib.colormaps.get_cmap("plasma")
    gradient = np.linspace(0, 1, len(countries))
    colors = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]
    symbols = ["o", "s", "^", "v", "D"]
    data = df[features].to_numpy()
    ax.scatter(
        data[:, comp1 - 1],
        data[:, comp2 - 1],
        s=8,
        c="black",
        alpha=0.05,
        edgecolors="None",
    )
    for i, country in enumerate(countries):
        country_id = [i for i in country_dict if country_dict[i] == country][0]
        trajectory = np.loadtxt("all_trajectories/" + str(country_id) + "_1.dat")
        c1 = trajectory[:, comp1]
        c2 = trajectory[:, comp2]
        if len(countries) == 1:
            arrowplot(ax, c1, c2, nArrs=1, color=main_color)
            ax.set_title(country)
        else:
            arrowplot(
                ax,
                c1,
                c2,
                markerstyle=symbols[i],
                nArrs=1,
                color=colors[i],
                label=country + f" {int(trajectory[0, 0])}-{int(trajectory[-1, 0])}",
            )
        if country == "Japan":
            index = trajectory[:, 0] == 1920
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
            index = trajectory[:, 0] == 1945
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2020
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
        elif country == "Colombia":
            index = trajectory[:, 0] == 1945
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2]),
                color=colors[i],
            )
            index = trajectory[:, 0] == 1955
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1] - 1.3, trajectory[index, 2]),
                color=colors[i],
            )
            index = trajectory[:, 0] == 1972
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] - 0.35),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2013
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] - 0.35),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2020
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1] - 1.3, trajectory[index, 2]),
                color=colors[i],
            )
        elif country == "Hungary":
            index = trajectory[:, 0] == 1944
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1] - 1.2, trajectory[index, 2] - 0.3),
                color=colors[i],
            )
            index = trajectory[:, 0] == 1960
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
            index = trajectory[:, 0] == 1989
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2013
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2020
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] - 0.3),
                color=colors[i],
            )
        elif country == "USA":
            index = trajectory[:, 0] == 1969
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] - 0.3),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2001
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1] + 0.1, trajectory[index, 2] - 0.3),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2009
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1], trajectory[index, 2] + 0.1),
                color=colors[i],
            )
            index = trajectory[:, 0] == 2019
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1] - 1.3, trajectory[index, 2] - 0.1),
                color=colors[i],
            )
        elif country == "Switzerland":
            index = trajectory[:, 0] == 2020
            ax.plot(
                trajectory[index, 1],
                trajectory[index, 2],
                symbols[i],
                color="black",
                mfc="None",
            )
            ax.annotate(
                r"\textbf{" + str(int(trajectory[index, 0][0])) + "}",
                (trajectory[index, 1] + 0.1, trajectory[index, 2] - 0.3),
                color=colors[i],
            )

    if len(countries) == 1:
        plt.savefig("figures/trajectory_" + str(country) + ".pdf", bbox_inches="tight")
    else:
        ax.legend(frameon=False, loc="lower left")
        plt.savefig("figures/trajectory_countries.pdf", bbox_inches="tight")
    return fig, ax


def trajectory_with_tamsd_inset(
    path,
    countries,
    comp1,
    comp2,
    df,
    features,
    inset_width=0.35,
    inset_height=0.35,
    inset_left=0.15,
    inset_bottom=0.15,
    min_trajec_length=50,
    savepath="figures/trajectory_with_tamsd_inset.pdf",
):
    """
    Create a trajectory plot with a TAMSD inset at a manually specified position.

    """

    # main trajectory plot
    fig, ax = trajectory_plot(countries, comp1, comp2, df=df, features=features)

    # inset position in figure coordinates
    inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])

    # draw TAMSD directly inside inset
    create_tamsd_plot_country(
        path, countries, ax=inset_ax, min_trajec_length=min_trajec_length
    )

    # formatting inset
    inset_ax.set_xlabel(r"Lag time $\Delta$ in years", fontsize=10)
    inset_ax.set_ylabel(r"TAMSD $\overline{\delta^2(\Delta)}$", fontsize=10)
    inset_ax.tick_params(labelsize=8)

    # save combined plot
    plt.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

    return fig, ax, inset_ax


def compute_fpt_2d(trajectory, radius):
    times = trajectory[:, 0]
    positions = trajectory[:, 1:3]
    N = len(trajectory)
    fpt_times = np.full(N, np.nan)

    for i in range(N):
        origin = positions[i]
        t0 = times[i]
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[j] - origin)
            if dist > radius:
                fpt_times[i] = times[j] - t0
                break
    return fpt_times, times


def bin_fpts_2d(
    ax,
    path,
    radius,
    startyear,
    stopyear,
    bin_size=1.0,
    df=None,
    features=None,
    symbsize=10,
    max_fpt=8,
):
    voxel_dict = defaultdict(list)
    N = 0
    for count, filename in enumerate(os.listdir(path)):
        file = os.path.join(path, filename)
        startindex = 0
        stopindex = -1
        if os.path.isfile(file):
            traj = np.loadtxt(file)
            if (
                np.any(traj[:, 0] >= startyear)
                and traj[-1, 0] > startyear
                and np.any(traj[:, 0] <= stopyear)
            ):
                startindex = np.where(traj[:, 0] >= startyear)[0][0]
                traj = traj[startindex:, :]
                if traj[-1, 0] >= stopyear:
                    stopindex = np.where(traj[:, 0] == stopyear)[0][0]
                if stopindex != -1:
                    traj = traj[: stopindex + 1, :]
                fpt, times = compute_fpt_2d(traj, radius)
                positions = traj[:, 1:3]
                N = N + 1
                for pos, f in zip(positions, fpt):
                    if not np.isnan(f):
                        voxel = tuple(np.floor(pos / bin_size).astype(int))
                        voxel_dict[voxel].append(f)

    voxel_means = {k: np.mean(v) for k, v in voxel_dict.items()}
    voxels = np.array([k for k in voxel_means.keys()])
    mean_fpts = np.array([v for v in voxel_means.values()])

    data = df[features].to_numpy()
    ax.scatter(data[:, 0], data[:, 1], s=3, c="black", alpha=0.01, edgecolors="None")
    true_positions = (voxels + 0.5) * bin_size
    sc = ax.scatter(
        true_positions[:, 0],
        true_positions[:, 1],
        vmin=1,
        vmax=max_fpt,
        c=mean_fpts,
        cmap="plasma",
        s=symbsize,
    )
    print(f"max fpt={max(mean_fpts)}")
    if startyear == 1960 or startyear == 1990:
        ax.set_xlabel("PC1")
    if startyear == 1960 or startyear == 1900:
        ax.set_ylabel("PC2")
    ax.set_title(f"{startyear}-{stopyear}")

    return sc  # return scatter for shared colorbar


def plot_fpt_years(df, bin_size, symbsize, max_fpt):
    intervals = [(1900, 1930), (1930, 1960), (1960, 1990), (1990, 2020)]
    fig, axes = plt.subplots(2, 2, figsize=set_size(), sharex=True, sharey=True)

    colorbar_ref = None
    for ax, (i, j) in zip(axes.flatten(), intervals):
        ax.tick_params(direction="in")
        sc = bin_fpts_2d(
            ax=ax,
            path="all_trajectories/",
            radius=0.1,
            startyear=i,
            stopyear=j,
            bin_size=bin_size,
            df=df,
            features=["pc1", "pc2"],
            symbsize=symbsize,
            max_fpt=max_fpt,
        )
        colorbar_ref = sc  # keep last scatter for colorbar
    # fig.tight_layout()
    # add single shared colorbar
    if colorbar_ref is not None:
        cbar = fig.colorbar(colorbar_ref, ax=axes, orientation="vertical")
        cbar.set_label("Mean FPT", fontsize=12)
    plt.savefig("figures/mean_fpt_all.pdf", bbox_inches="tight")
    plt.close()


def jump_wt_distr(path, eps=0, min_trajec_length=0):
    jumps = []
    wt = []
    count = 0
    for filename in os.listdir(path):
        file = path + filename
        country_id = int(filename.split("_")[0])
        if os.path.isfile(file):
            trajectory = np.loadtxt(file)
            if trajectory.shape[0] >= min_trajec_length:
                pjumps = np.zeros((trajectory.shape[0] - 1, trajectory.shape[1]))
                # calculate jump distance
                for i in range(trajectory.shape[0] - 1):
                    pjumps[i, :] = trajectory[i + 1, :] - trajectory[i, :]
                # calculate waiting times from zero distance jumps
                tau = 0
                for i in range(trajectory.shape[0] - 1):
                    if np.sum(np.abs(pjumps[i, 1:])) <= eps:
                        tau = tau + 1
                    else:
                        tau = tau + 1
                        wt = np.append(wt, tau)
                        tau = 0
                # remove 0 length jumps
                pjumps = pjumps[np.sum(np.abs(pjumps), axis=1) > 1.0 + eps]
                pjumps[:, 0] = country_id * pjumps[:, 0]
                if count == 0:
                    jumps = pjumps
                else:
                    jumps = np.append(jumps, pjumps, axis=0)
                count = count + 1
    return jumps, wt


def MLE_discrete_powerlaw(x, x0):
    n = len(x)
    return n / (np.sum(np.log(x / (x0 - 0.5))))


def MLE_powerlaw(x, x0):
    n = len(x)
    return n / (np.sum(np.log(x / x0)))


def MLE_bounded_powerlaw(x, xmin, xmax):
    n = len(x)
    alphas = np.linspace(0.01, 0.999, num=10 ^ 20)
    L = np.zeros_like(alphas)
    for i, alpha in enumerate(alphas):
        L[i] = -alpha * np.sum(np.log(x)) + n * np.log(
            (1 - alpha) / (xmax ** (1 - alpha) - xmin ** (1 - alpha))
        )
    return alphas[np.argmax(L)] - 1


def test_step_distr(path="all_trajectories/"):
    jumps, wt = jump_wt_distr(path, eps=0)
    jumps = np.abs(jumps)
    cmap = matplotlib.colormaps.get_cmap("plasma")
    gradient = np.linspace(0, 1, 3)
    colors = list(reversed([matplotlib.colors.to_hex(cmap(i)) for i in gradient]))
    xyt = 10 ** (-1)
    # vary cutoff
    print(f"max Jump PC1={max(jumps[:, 1])}")
    print(f"max Jump PC2={max(jumps[:, 2])}")
    cutoffs = np.logspace(-2, np.log10(0.4), 100)
    fig, ax = plt.subplots(1, 1, figsize=set_size())
    ax.set_xscale("log")
    ax.set_ylabel(r"$\mu_i$")
    ax.set_xlabel(r"Crossover Point $\mathrm{PC}_c$")
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    tstart, tstop = 3 * 10 ** (-2), 0.25
    ax.axvline(tstart, color="gray", ls="dashed")
    ax.axvline(tstop, color="gray", ls="dashed")
    ax.axhline(1.6, color="gray", ls="dashed")
    ax.axhline(2.4, color="gray", ls="dashed")
    # likelihood
    mu1, mu2 = [], []
    for cutoff in cutoffs:
        mu1 = np.append(
            mu1, 1 + MLE_powerlaw(jumps[:, 1][jumps[:, 1] > cutoff], cutoff)
        )
        mu2 = np.append(
            mu2, 1 + MLE_powerlaw(jumps[:, 2][jumps[:, 2] > cutoff], cutoff)
        )
    ax.plot(cutoffs, mu1, "o", mfc="None", color=second_color, label=f"MLE PC1")
    ax.plot(cutoffs, mu2, "s", mfc="None", color=main_color, label=f"MLE PC2")
    ax.vlines(xyt, 1.6, 2.4, color="black")

    ax.legend(frameon=False)
    plt.savefig("figures/PC_likelihood_cutoffs.pdf", bbox_inches="tight")
    plt.close()


def create_hist_one_fig(
    path, eps=0, inset_width=0.4, inset_height=0.4, inset_left=0.05, inset_bottom=0.05
):
    jumps, wt = jump_wt_distr(path, eps=eps)
    jumps = np.abs(jumps)
    # step size PDF
    fig, ax = plt.subplots(1, 1, figsize=set_size())
    symbols = ["o", "s"]
    colors = [second_color, main_color]
    ax.set_ylabel(r"PDF $\lambda(\mathrm{PC})$")
    xyl = 10 ** (-6)
    xyu = 10 ** (1)
    B, A = 0, 0
    xyt = 0.1
    x1, x2 = 0, 0
    tstart, tstop = 3 * 10 ** (-2), 0.25
    ax.axvline(tstart, color="gray", ls="dashed")
    ax.axvline(tstop, color="gray", ls="dashed")
    ax.axvline(xyt, color="black")
    for i in (1, 2):
        ax.tick_params(top=True, right=True, direction="in", which="both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _, bins = np.histogram(np.log10(jumps[:, i]), bins="fd", density=True)
        hist, bin_edges = np.histogram(jumps[:, i], bins=10**bins, density=True)
        bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
        bin_centers = bin_centers[hist > 0]
        hist = hist[hist > 0]
        ax.plot(
            bin_centers,
            hist,
            symbols[i - 1],
            mfc="None",
            color=colors[i - 1],
            label=f"PC{i}",
        )

        if i == 1:
            mu1 = MLE_powerlaw(jumps[:, 1][jumps[:, 1] > xyt], xyt)
            dmu1 = mu1 / np.sqrt(len(jumps[:, 1][jumps[:, 1] > xyt]))
            nu1 = MLE_bounded_powerlaw(jumps[:, 1][jumps[:, 1] <= xyt], xyl, xyt)
            dnu1 = nu1 / np.sqrt(len(jumps[:, 1][jumps[:, 1] <= xyt]))
            digits = 3
            ax.annotate(
                r"$\nu_1="
                + str(round(1 + nu1, digits))
                + r"\pm"
                + f"{round(np.abs(dnu1), digits)}"
                + "$",
                (5 * 10 ** (-4), 1.5),
                color=colors[i - 1],
                fontsize=10,
            )
            ax.annotate(
                r"$\mu_1="
                + str(round(1 + mu1, digits))
                + r"\pm"
                + f"{round(np.abs(dmu1), digits)}"
                + "$",
                (0.25, 6),
                color=colors[i - 1],
                fontsize=10,
            )
            B = 1 / (
                xyt ** (nu1 - mu1) * (xyl ** (-nu1) - xyt ** (-nu1)) / nu1
                + (xyt ** (-mu1) - xyu ** (-mu1)) / mu1
            )
            A = xyt ** (nu1 - mu1) * B
            x1 = np.logspace(np.log10(xyl), np.log10(xyt))
            x2 = np.logspace(np.log10(xyt), np.log10(xyu))
            ax.plot(x1, A * x1 ** (-nu1 - 1), ls="dashed", color=colors[i - 1])
            ax.plot(x2, B * x2 ** (-mu1 - 1), ls="dashed", color=colors[i - 1])
        elif i == 2:
            mu2 = MLE_powerlaw(jumps[:, 2][jumps[:, 2] > xyt], xyt)
            dmu2 = mu2 / np.sqrt(len(jumps[:, 2][jumps[:, 2] > xyt]))
            nu2 = MLE_bounded_powerlaw(jumps[:, 2][jumps[:, 2] <= xyt], xyl, xyt)
            dnu2 = nu2 / np.sqrt(len(jumps[:, 2][jumps[:, 2] <= xyt]))
            digits = 3
            ax.annotate(
                r"$\nu_2="
                + str(round(1 + nu2, digits))
                + r"\pm"
                + f"{round(np.abs(dnu2), digits)}"
                + "$",
                (5 * 10 ** (-4), 41),
                color=colors[i - 1],
                fontsize=10,
            )
            ax.annotate(
                r"$\mu_2="
                + str(round(1 + mu2, digits))
                + r"\pm"
                + f"{round(np.abs(dmu2), digits)}"
                + "$",
                (0.25, 2),
                color=colors[i - 1],
                fontsize=10,
            )
            B = 1 / (
                xyt ** (nu2 - mu2) * (xyl ** (-nu2) - xyt ** (-nu2)) / nu2
                + (xyt ** (-mu2) - xyu ** (-mu2)) / mu2
            )
            A = xyt ** (nu2 - mu2) * B
            x1 = np.logspace(np.log10(xyl), np.log10(xyt))
            x2 = np.logspace(np.log10(xyt), np.log10(xyu))
            ax.plot(x1, A * x1 ** (-nu2 - 1), ls="dashed", color=colors[i - 1])
            ax.plot(x2, B * x2 ** (-mu2 - 1), ls="dashed", color=colors[i - 1])

    ax.legend(frameon=False)
    ax.set_xlabel("Step Size $|$PC$|$")
    fig.tight_layout()
    inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    ### waiting time PDF
    inset_ax.tick_params(
        top=True, right=True, direction="in", which="both", labelsize=8
    )
    unique, counts = np.unique(wt, return_counts=True)
    counts = counts / np.sum(counts)

    inset_ax.set_xscale("log")
    inset_ax.set_yscale("log")
    inset_ax.plot(unique, counts, "^", mfc="None", color=third_color)

    inset_ax.set_ylabel(r"PDF $\psi(\tau)$", fontsize=8)
    inset_ax.xaxis.tick_top()
    inset_ax.yaxis.tick_right()
    inset_ax.xaxis.set_label_position("top")
    inset_ax.yaxis.set_label_position("right")
    inset_ax.set_xlabel(r"Sojourn Time $\tau$", fontsize=8)
    alpha = MLE_discrete_powerlaw(wt, 1)
    inset_ax.plot(
        unique,
        alpha * 0.5**alpha / (1 - (1 / 5) ** alpha) * unique ** (-(alpha + 1)),
        color=third_color,
        ls="dashed",
        label=r"$\alpha="
        + str(round(alpha, 3))
        + r"\pm"
        + str(round(alpha / np.sqrt(len(wt)), 3))
        + "$",
    )
    inset_ax.legend(frameon=False, fontsize=8)
    inset_ax.set_xticks([1, 2, 3, 4, 5])
    inset_ax.set_yticks([0.1])
    inset_ax.set_yticklabels([str(0.1)])
    inset_ax.set_xticklabels([str(i) for i in [1, 2, 3, 4, 5]])

    plt.savefig("figures/hist_all.pdf", bbox_inches="tight")
    plt.close()


def _soft_class_background(
    ax,
    class_points,
    class_colors,
    grid_res=300,
    bw_scale=1.0,
    max_alpha=0.6,
    density_clip=0.02,
):
    """
    class_points: list of (N_i, 2) arrays for each class (x,y)
    class_colors: list of RGBA tuples in [0,1]
    """
    # Bounds from current data limits (with small padding)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin
    xmin, xmax = xmin - 0.02 * xr, xmax + 0.02 * xr
    ymin, ymax = ymin - 0.02 * yr, ymax + 0.02 * yr

    # Grid
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(xs, ys)
    XY = np.vstack([X.ravel(), Y.ravel()])  # shape (2, M)

    # Per-class KDEs -> densities on grid
    densities = []
    for pts in class_points:
        if pts.shape[0] == 0:
            densities.append(np.zeros(XY.shape[1]))
            continue
        # gaussian_kde chooses bandwidth via Scott by default; we can scale it
        kde = gaussian_kde(pts.T)
        kde.set_bandwidth(kde.factor * bw_scale)
        densities.append(kde(XY))

    D = np.vstack(densities)  # (C, M)
    Dsum = D.sum(axis=0) + 1e-12  # avoid /0
    W = D / Dsum  # normalized weights per class
    Dtot = Dsum.reshape(Y.shape)  # total density (for masking/alpha)

    # Build RGBA image by blending the class colors with weights
    rgb = np.zeros((XY.shape[1], 3))
    for i, col in enumerate(class_colors):
        rgb += W[i][:, None] * np.array(col[:3])[None, :]

    rgb = rgb.reshape(Y.shape + (3,))
    # alpha based on total density; fade out low-density regions
    Dmax = Dtot.max() if np.isfinite(Dtot.max()) and Dtot.max() > 0 else 1.0
    alpha = (Dtot / Dmax) ** 0.5  # gentle rolloff
    alpha[Dtot < density_clip * Dmax] = 0.0  # hide very low-density tails
    alpha = np.clip(alpha * max_alpha, 0, 1)

    rgba = np.dstack([rgb, alpha])
    ax.imshow(
        rgba,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        interpolation="bilinear",
        zorder=-5,
        aspect="auto",
    )


def plot_space_examples(df, Auto, Hybrid, Demo):
    path = "all_trajectories/"
    country_dict = dict(read_country_table())
    reverse_country_dict = {v: k for k, v in country_dict.items()}
    features = ["pc1", "pc2"]
    data = df[features].to_numpy()
    fig, ax = plt.subplots(
        1, 1, figsize=set_size(width=595.80026, ratio=418.25555 / 595.80026)
    )
    ax.tick_params(top=True, right=True, direction="in", which="both")
    ax.set_xlabel("Democraticness (PC1)", fontsize=16)
    ax.set_ylabel("Election Capability - Civil Liberties (PC2)", fontsize=16)

    colors = [cm.plasma(0.1), cm.plasma(0.5), cm.plasma(0.8)]  # Demo, Hybrid, Auto
    text_colors = [cm.plasma(0.15), cm.plasma(0.55), cm.plasma(0.85)]

    # Background scatter (all data)
    ax.scatter(data[:, 0], data[:, 1], s=8, c="black", alpha=0.05, edgecolors="none")

    # Collect class points while plotting labels
    auto_xy, mono_xy, demo_xy = [], [], []
    symbols = ["o", "s", "^", "v", "D"]
    country_used = set()
    k = -1

    def plot_group(group, marker, color, annotate_rules, bucket_xy, text_color):
        nonlocal k
        for idx, (country_name, startyear) in enumerate(group):
            country_id = reverse_country_dict.get(country_name)
            if country_id is None:
                print(f"Country '{country_name}' not found in dictionary.")
                continue

            trajectory = None
            for frag in range(1, 5):
                filename = f"{country_id}_{frag}.dat"
                file_path = os.path.join(path, filename)
                if not os.path.isfile(file_path):
                    continue
                temp_trajectory = np.loadtxt(file_path)
                if (
                    temp_trajectory[0, 0] <= startyear
                    and temp_trajectory[-1, 0] > startyear
                ):
                    trajectory = temp_trajectory
                    break

            if trajectory is None:
                print(
                    f"No trajectory found for {country_name} including year {startyear}"
                )
                continue

            try:
                startindex = np.where(trajectory[:, 0] == startyear)[0][0]
            except IndexError:
                print(
                    f"Start year {startyear} not found exactly in trajectory for {country_name}."
                )
                continue

            if country_name not in country_used:
                country_used.add(country_name)
                k += 1

            x, y = trajectory[startindex, 1], trajectory[startindex, 2]
            bucket_xy.append([x, y])

            lbl = f"{country_name} {int(trajectory[startindex, 0])}"
            ax.plot(x, y, marker, color=color, zorder=5)

            # annotation positioning rules (callable or dict)
            if callable(annotate_rules):
                ha, va, dx, dy = annotate_rules(country_name)
            else:
                ha, va, dx, dy = annotate_rules.get(
                    country_name, ("center", "bottom", 0.0, 0.1)
                )

            ax.annotate(
                r"$\textbf{" + lbl + r"}$",
                (x + dx, y + dy),
                color=text_color,
                fontsize=10,
                ha=ha,
                va=va,
                zorder=6,
            )

    # Per-group label offsets
    def auto_rules(name):
        if name == "Germany":
            return ("right", "top", -0.1, 0.0)
        elif name == "North Korea":
            return ("right", "bottom", -0.1, 0.0)
        else:
            return ("left", "bottom", +0.1, 0.0)

    def hybrid_rules(name):
        if name == "Japan":
            return ("left", "bottom", 0.0, +0.1)
        elif name == "Turkey":
            return ("right", "bottom", 0.0, +0.1)
        elif name == "India":
            return ("center", "bottom", 0.0, +0.1)
        else:
            return ("center", "top", 0.0, -0.1)

    def demo_rules(name):
        if name == "USA":
            return ("center", "top", 0.0, -0.1)
        elif name == "Switzerland":
            return ("right", "bottom", 0.0, +0.1)
        elif name == "Sweden":
            return ("center", "bottom", +0.6, +0.1)
        else:
            return ("center", "bottom", 0.0, +0.1)

    # Draw points + collect coordinates
    plot_group(Auto, symbols[2], colors[2], auto_rules, auto_xy, text_color="black")
    plot_group(Hybrid, symbols[1], colors[1], hybrid_rules, mono_xy, text_color="black")
    plot_group(Demo, symbols[0], colors[0], demo_rules, demo_xy, text_color="black")

    # --- SOFT, BLENDED BACKGROUND ---
    auto_xy = np.array(auto_xy)
    mono_xy = np.array(mono_xy)
    demo_xy = np.array(demo_xy)
    _soft_class_background(
        ax,
        class_points=[demo_xy],  # match colors order
        class_colors=[colors[0]],
        grid_res=400,  # higher -> smoother
        bw_scale=1.0,  # try 0.7..1.5 depending on spread
        max_alpha=0.5,  # overall background strength
        density_clip=0.02,  # hide faint tails
    )
    _soft_class_background(
        ax,
        class_points=[auto_xy],  # match colors order
        class_colors=[colors[2]],
        grid_res=400,  # higher -> smoother
        bw_scale=0.8,  # try 0.7..1.5 depending on spread
        max_alpha=0.5,  # overall background strength
        density_clip=0.03,  # hide faint tails
    )
    _soft_class_background(
        ax,
        class_points=[mono_xy],  # match colors order
        class_colors=[colors[1]],
        grid_res=400,  # higher -> smoother
        bw_scale=2.5,  # try 0.7..1.5 depending on spread
        max_alpha=0.5,  # overall background strength
        density_clip=0.02,  # hide faint tails
    )

    # Category labels
    ax.annotate(
        r"\textbf{Autocracies}",
        (-12, 0),
        color=colors[2],
        fontsize=16,
    )
    # bbox=dict(facecolor='none', edgecolor=colors[2]))
    ax.annotate(
        r"\textbf{Hybrid Regimes}",
        (-4, -4.3),
        color=colors[1],
        fontsize=16,
    )
    # bbox=dict(facecolor='none', edgecolor=colors[1]))
    ax.annotate(
        r"\textbf{Democracies}",
        (3.2, 4),
        color=colors[0],
        fontsize=16,
    )
    # bbox=dict(facecolor='none', edgecolor=colors[0]))

    plt.savefig("figures/PC_space.pdf", bbox_inches="tight")
    plt.close()


def plot_extreme_on_ax(
    ax, df, features, selections, title=None, show_legend=True, color=main_color
):
    """
    Plot selected extreme examples on a provided axis.

    Returns:
    --------
    step_sizes : np.array
        Absolute step sizes in PC1 direction for each selection, in order.
    """
    path = "all_trajectories/"
    country_dict = dict(read_country_table())
    reverse_country_dict = {v: k for k, v in country_dict.items()}
    data = df[features].to_numpy()

    ax.tick_params(top=True, right=True, direction="in", which="both")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if title:
        ax.set_title(title)

    # Background scatter
    ax.scatter(data[:, 0], data[:, 1], s=8, c="black", alpha=0.01, edgecolors="None")

    nr_countries = 0
    country_used = set()
    for idx, (country_name, startyear) in enumerate(selections):
        if country_name not in country_used:
            country_used.add(country_name)
            nr_countries += 1

    symbols = ["o", "s", "^", "D", "v"]

    country_used = set()
    k = -1
    step_sizes = []
    for idx, (country_name, startyear) in enumerate(selections):
        print(country_name)
        country_id = reverse_country_dict.get(country_name)
        if country_id is None:
            print(f"Country '{country_name}' not found in dictionary.")
            continue

        trajectory = None
        for frag in range(1, 5):
            filename = f"{country_id}_{frag}.dat"
            file_path = os.path.join(path, filename)
            if not os.path.isfile(file_path):
                continue
            temp_trajectory = np.loadtxt(file_path)
            if (
                temp_trajectory[0, 0] <= startyear
                and temp_trajectory[-1, 0] > startyear
            ):
                trajectory = temp_trajectory
                break

        if trajectory is None:
            print(f"No trajectory found for {country_name} including year {startyear}")
            continue

        try:
            startindex = np.where(trajectory[:, 0] == startyear)[0][0]
        except IndexError:
            print(
                f"Start year {startyear} not found exactly in trajectory for {country_name}."
            )
            continue

        if country_name not in country_used:
            country_used.add(country_name)
            k += 1

        segment = trajectory[startindex:, :]
        marker = symbols[idx % len(symbols)]
        label = f"{country_name} {int(segment[0, 0])}-{int(segment[1, 0])}"

        # Calculate absolute step size in PC1 direction (index 1)
        step_size = np.abs(segment[1, 1] - segment[0, 1])
        step_sizes.append(step_size)

        arrowplot(
            ax,
            segment[0:2, 1],
            segment[0:2, 2],
            nArrs=1,
            color=color,
            markerstyle=marker,
            label=label,
        )

    if show_legend and (title == "Democratisation" or title == "Autocratisation"):
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    elif show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    return np.array(step_sizes)


def create_extreme_stepsize_composite(
    df,
    features,
    selections_list,
    titles=None,
    savepath="figures/extreme_stepsize_composite.pdf",
):
    """
    Create a composite figure with a 2x2 grid of extreme examples plots.

    Parameters:
    -----------
    df : pandas DataFrame
        The data containing PC1 and PC2 features
    features : list
        Features to plot (e.g., ["pc1", "pc2"])
    selections_list : list of list of tuples
        List of 4 selection lists, each containing (country_name, startyear) tuples
        for each of the 4 subplots (top-left, top-right, bottom-left, bottom-right)
    titles : list of str, optional
        List of 4 titles for each subplot
    savepath : str
        Path to save the figure
    """
    import matplotlib.gridspec as gridspec
    from matplotlib import cm

    if titles is None:
        titles = [None, None, None, None]

    # Sample 4 colors evenly from plasma colormap
    plasma = cm.get_cmap("plasma")
    colors = [plasma(i / 3) for i in range(4)]

    fig = plt.figure(figsize=set_size(width=595.80026, ratio=0.8))

    # Create GridSpec: 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs.update(wspace=0.35, hspace=0.3)

    # Create all 4 subplots
    axes = [
        fig.add_subplot(gs[0, 0]),  # top-left
        fig.add_subplot(gs[0, 1]),  # top-right
        fig.add_subplot(gs[1, 0]),  # bottom-left
        fig.add_subplot(gs[1, 1]),  # bottom-right
    ]
    labels = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}", r"\textbf{(d)}"]

    for ax, selections, title, color, label in zip(
        axes, selections_list, titles, colors, labels
    ):
        plot_extreme_on_ax(
            ax,
            df,
            features,
            selections,
            title=title,
            color=color,
        )
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=12)

    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    return fig
