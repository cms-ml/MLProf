# coding: utf-8

colors = {"mpl_standard": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
          "custom_edgecolor": ['#CC4F1B', '#1B2ACC', '#3F7F4C'],
          "custom_facecolor": ['#FF9848', '#089FFF', '#7EFF99'],
          }


def calculate_overall_mean_from_different_measurements(means, sample_sizes):
    '''
    Helper function to measure the overall mean of subsequent measurements of the same underlying
    distribution (e.g. several runtime measurements for the same batch size)
    '''
    return sum(means * sample_sizes) / sum(sample_sizes)


def calculate_overall_mean_and_std_from_different_measurements(means, stds, sample_sizes):
    '''
    Helper function calculating the overall mean and standard deviation of subsequent measurements
    of the same underlying distribution (e.g. several runtime measurements for the same batch size)
    '''
    import numpy as np

    mean = calculate_overall_mean_from_different_measurements(means, sample_sizes)
    std = np.sqrt(sum(stds**2 * (sample_sizes) + sample_sizes * (mean - means)**2) / (sum(sample_sizes)))
    return mean, std


def calculate_means_and_stds_per_batch_size(different_batchsizes, sample_size, path):
    '''
    Calculate the mean and standard deviations of the different measurements for each batch size separately
    '''
    import numpy as np
    import pandas as pd

    means = np.empty(len(different_batchsizes))
    stds = np.empty(len(different_batchsizes))
    pd_dataset = pd.read_csv(path, delimiter=",", names=["batch_size", "mean", "std"])
    for i, batchsize in enumerate(different_batchsizes):
        means_per_batchsize = pd_dataset.loc[pd_dataset["batch_size"] == batchsize, "mean"]
        stds_per_batchsize = pd_dataset.loc[pd_dataset["batch_size"] == batchsize, "std"]
        sample_sizes = np.ones_like(means_per_batchsize) * sample_size
        means[i], stds[i] = calculate_overall_mean_and_std_from_different_measurements(means_per_batchsize,
                                                                                       stds_per_batchsize,
                                                                                       sample_sizes)
    return means, stds


def open_csv_file(path, columns):
    import pandas as pd
    pd_dataset = pd.read_csv(path, delimiter=",", names=columns)

    # or with chunking?
    '''
    tp = pd.read_csv(path, delimiter=",", names=columns, iterator=True, chunksize=2000)
    df = pd.concat(tp, ignore_index=True)
    '''
    return pd_dataset


def calculate_means_and_errors_per_batch_size(different_batchsizes, path):
    import pandas as pd
    import numpy as np

    # open the csv file
    pd_dataset = open_csv_file(path, ["batch_size", "runtimes"])

    # create the arrays to be plotted with mean values and up and down errors
    medians = np.empty(len(different_batchsizes))
    err_down = np.empty(len(different_batchsizes))
    err_up = np.empty(len(different_batchsizes))

    for i, batchsize in enumerate(different_batchsizes):
        runtimes_per_batchsize = pd_dataset.loc[pd_dataset["batch_size"] == batchsize, "runtimes"]
        # mean = np.mean(runtimes_per_batchsize)
        median = np.percentile(runtimes_per_batchsize, 50)
        medians[i] = median
        err_down[i] = abs(np.percentile(runtimes_per_batchsize, 16) - median)
        err_up[i] = abs(np.percentile(runtimes_per_batchsize, 84) - median)
    return medians, err_down, err_up


def plot_batchsize_old(different_batchsizes, sample_size, input_path, output_path):
    '''
    Calculate and plot the mean and standard deviation of the runtime per batchsize

    Args:
    different_batchsizes: list(int). The list of different batch sizes to be plotted
    sample_size: int. The underlying statistics of the mean and stds values to be obtained from input_path
    input_path: str. The path to the csv file containing the results of the measurement.
    output_path: str. The path to which the plot has to be saved.
    '''
    import matplotlib.pyplot as plt
    import mplhep as hep

    means, stds = calculate_means_and_stds_per_batch_size(different_batchsizes, sample_size, input_path)
    hep.set_style(hep.style.CMS)

    fig, ax = plt.subplots(1, 1)
    plt.errorbar(different_batchsizes, means / different_batchsizes, yerr=stds / different_batchsizes, capsize=12,
                 marker=".", linestyle="")
    plt.xscale("log")
    plt.xlabel("batch size")
    plt.ylabel("runtime/batch size [ms]")
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(different_batchsizes)))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    plt.xticks(different_batchsizes, different_batchsizes)
    hep.cms.text(text="Simulation, Network test", loc=0)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_batchsize(different_batchsizes, input_path, output_path):
    '''
    Calculate and plot the mean and standard deviation of the runtime per batchsize

    Args:
    different_batchsizes: list(int). The list of different batch sizes to be plotted
    sample_size: int. The underlying statistics of the mean and stds values to be obtained from input_path
    input_path: str. The path to the csv file containing the results of the measurement.
    output_path: str. The path to which the plot has to be saved.
    '''
    import matplotlib.pyplot as plt
    import mplhep as hep

    medians, err_down, err_up = calculate_means_and_errors_per_batch_size(different_batchsizes, input_path)
    hep.set_style(hep.style.CMS)

    fig, ax = plt.subplots(1, 1)
    color = next(ax._get_lines.prop_cycler)['color']
    fill_plot(different_batchsizes, medians / different_batchsizes, err_down / different_batchsizes,
              err_up / different_batchsizes, True, "", color)
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("batch size")
    plt.ylabel("runtime/batch size [ms]")
    plt.ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(different_batchsizes)))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    plt.xticks(different_batchsizes, different_batchsizes)
    hep.cms.text(text="Network test", loc=0)  # hep.cms.text(text="Simulation, Network test", loc=0)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close()


def fill_plot(x, y, yerr_d, yerr_u, filling, model_name, color):
    import matplotlib.pyplot as plt
    import numpy as np
    if filling:
        p1 = plt.plot(x, y, '-', color=color)
        plt.fill_between(x, y - yerr_d, y + yerr_u, alpha=0.5, facecolor=color)
        p2 = plt.fill(np.NaN, np.NaN, alpha=0.5, color=color)
        legend = (p1[0], p2[0])
    else:
        p = plt.errorbar(x, y,
                        yerr=(yerr_d, yerr_u),
                        capsize=12,
                        marker=".", linestyle="")
        legend = p[0]
    return legend


def plot_batchsize_several_measurements(different_batchsizes, input_paths, output_path, measurements,
                                    bs_normalized=True, filling=True):
    import matplotlib.pyplot as plt
    import mplhep as hep

    plotting_values = {}
    for i, input_path in enumerate(input_paths):
        medians, err_down, err_up = calculate_means_and_errors_per_batch_size(different_batchsizes, input_path)
        if bs_normalized:
            medians = medians / different_batchsizes
            err_down = err_down / different_batchsizes
            err_up = err_up / different_batchsizes
        plotting_values[measurements[i]] = {"medians": medians, "err_down": err_down, "err_up": err_up}

    hep.set_style(hep.style.CMS)
    fig, ax = plt.subplots(1, 1)
    to_legend = []
    for i, input_path in enumerate(input_paths):
        color = next(ax._get_lines.prop_cycler)['color']
        legend = fill_plot(different_batchsizes, plotting_values[measurements[i]]["medians"],
                  plotting_values[measurements[i]]["err_down"],
                  plotting_values[measurements[i]]["err_up"], filling, measurements[i],
                  color)  # colors["mpl_standard"][i])
        to_legend += [legend]
    plt.legend(to_legend, measurements)
    plt.xscale("log")
    plt.xlabel("batch size")
    plt.ylabel("runtime/batch size [ms]")
    plt.ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(different_batchsizes)))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    plt.xticks(different_batchsizes, different_batchsizes)
    hep.cms.text(text="Network test", loc=0)  # hep.cms.text(text="Simulation, Network test", loc=0)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close()

# fill_between(x, y-error, y+error,
#     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
