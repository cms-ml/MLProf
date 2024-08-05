# coding: utf-8

colors = {
    "mpl": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
    # Atlas and cms standards correspond to results in :
    # Color wheel from https://arxiv.org/pdf/2107.02270 Table 1, 10 color palette
    # hexacodes in https://github.com/mpetroff/accessible-color-cycles/blob/0a17e754d9f83161baffd803dcea8bee7d95a549/readme.md#final-results # noqa
    # as implemented in mplhep
    "cms_6": [
        "#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd",
    ],
    "atlas_10": [
        "#3f90da",
        "#ffa90e",
        "#bd1f01",
        "#94a4a2",
        "#832db6",
        "#a96b59",
        "#e76300",
        "#b9ac70",
        "#717581",
        "#92dadd",
    ],
    # "custom_edgecolor": ["#CC4F1B", "#1B2ACC", "#3F7F4C"],
    # "custom_facecolor": ["#FF9848", "#089FFF", "#7EFF99"],
}


def open_csv_file(path, columns):
    import pandas as pd

    return pd.read_csv(path, delimiter=",", names=columns)


def calculate_medians_and_errors(batch_sizes, path):
    """
    Calculate and plot the medians and errors of the runtime per batch size

    Args:
    batch_sizes: list(int). The list of different batch sizes to be plotted
    path: str. The path to the csv file containing the results of the measurement.
    """
    import numpy as np

    # open the csv file
    pd_dataset = open_csv_file(path, ["batch_size", "runtimes"])

    # create the arrays to be plotted with median values and up and down errors
    medians = np.empty(len(batch_sizes))
    err_down = np.empty(len(batch_sizes))
    err_up = np.empty(len(batch_sizes))

    for i, batch_size in enumerate(batch_sizes):
        runtimes = pd_dataset.loc[pd_dataset["batch_size"] == batch_size, "runtimes"].values
        medians[i] = np.percentile(runtimes, 50)
        err_down[i] = medians[i] - np.percentile(runtimes, 16)
        err_up[i] = np.percentile(runtimes, 84) - medians[i]

    return medians, err_down, err_up


def apply_customizations(plot_params, fig, ax):
    """
    Apply the remaining customization parameters from the command line

    Args:
    plot_params: dict. The dictionary containing the customization parameters
    fig, ax: the matplotlib object to handle figure and axis.
    """
    # x axis
    if plot_params.get("x_log"):
        ax.set_xscale("log")

    # y axis
    if plot_params.get("y_log"):
        ax.set_yscale("log")
    if plot_params.get("y_min") is not None:
        y_min = plot_params["y_min"]
        if y_min <= 0 and plot_params.get("y_log"):
            y_min = 1e-3
            print(f"when y_log is set, y_min must be strictly positive, setting to {y_min}")
        ax.set_ylim(bottom=y_min)
    if plot_params.get("y_max") is not None:
        ax.set_ylim(top=plot_params["y_max"])


def fill_plot(x, y, y_down, y_up, error_style, color):
    """
    Fill the plots with the measured values and their errors

    Args:
    x: array(float). x-axis values
    y: array(float). y-axis values
    y_down: array(float). error down on the y-axis
    y_up: array(float). error up on the y-axis
    error_style: str. either bars and band
    color: the colors to use for the plotted values
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # TODO: use fig and ax instead
    if error_style == "band":
        p1 = plt.plot(x, y, "-", color=color)
        plt.fill_between(x, y - y_down, y + y_up, alpha=0.5, facecolor=color)
        p2 = plt.fill(np.nan, np.nan, alpha=0.5, color=color)
        legend = (p1[0], p2[0])
    else:  # bars
        p = plt.errorbar(x, y, yerr=(y_down, y_up), capsize=12, marker=".", linestyle="")
        legend = p[0]

    return legend


def plot_batch_size_several_measurements(
    batch_sizes,
    input_paths,
    output_path,
    measurements,
    color_list,
    plot_params,
):
    """
    General plotting function for runtime plots

    Args:
    batch_sizes: list(int). The batch sizes to be used for the x-axis of the plot.
    input_paths: list(str). The paths of the csv files containing the measurement results.
    output_path: str. The path to be used for saving the plot.
    measurements: list(str). The labels of the plot.
    plot_params: dict. The dictionary containing the customization parameters.
    """
    import matplotlib.pyplot as plt
    import mplhep  # type: ignore[import-untyped]
    from cycler import cycler

    if isinstance(measurements[0], str):
        measurements_labels_strs = list(measurements)
    else:
        measurements_labels_strs = [", ".join(measurement) for measurement in measurements]

    # get the values to be plotted, in the same order as the measurements
    plot_data = []
    for input_path in input_paths:
        medians, err_down, err_up = calculate_medians_and_errors(batch_sizes, input_path)
        if plot_params["bs_normalized"]:
            medians = medians / batch_sizes
            err_down = err_down / batch_sizes
            err_up = err_up / batch_sizes
        plot_data.append({"y": medians, "y_down": err_down, "y_up": err_up})

    # set style and add CMS logo
    with plt.style.context(mplhep.style.CMS):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        # create plot with curves using a single color for each value-error pair
        legend_entries = []
        if plot_params.get("custom_colors"):
            # set the color cycle to the custom color cycle
            ax._get_lines.set_prop_cycle(cycler("color", colors[plot_params.get("custom_colors")]))

        for i, data in enumerate(plot_data):
            color_used = color_list[i] if color_list[i] else ax._get_lines.get_next_color()
            entry = fill_plot(
                x=batch_sizes,
                y=data["y"],
                y_down=data["y_down"],
                y_up=data["y_up"],
                error_style=plot_params["error_style"],
                color=color_used,
            )
            legend_entries.append(entry)

        # create legend
        ax.legend(legend_entries, measurements_labels_strs)

        if not plot_params.get("y_log"):
            ax.set_ylim(bottom=0)

        # additional customizations
        apply_customizations(plot_params, fig, ax)

        # x axis
        ax.set_xlabel("Batch size")
        ax.xaxis.set_major_locator(plt.MaxNLocator(len(batch_sizes)))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)

        # y axis
        ax.set_ylabel("Runtime / batch size [ms]" if plot_params["bs_normalized"] else "Runtime [ms]")

        # texts
        mplhep.cms.text(text="Simulation, MLProf", loc=0)
        mplhep.cms.lumitext(text=plot_params["top_right_label"])

        # save plot
        fig.savefig(output_path, bbox_inches="tight")
