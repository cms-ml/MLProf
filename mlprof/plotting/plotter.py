

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


def plot_batchsize(different_batchsizes, sample_size, input_path, output_path):
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
    plt.ylabel("runtime/batch size [s]")
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(different_batchsizes)))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    plt.xticks(different_batchsizes, different_batchsizes)
    hep.cms.text(text="Simulation, Network test", loc=0)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsizes", type=int, nargs='*',
                        default=[1, 2, 4], help="the list of different batchsizes used during the test")
    parser.add_argument("-nRuns", "--numberRuns", type=int, default=200,
                        help="the statistics of each value in results.csv")
    parser.add_argument("-i", "--inputPath", type=str,
                        help="the path to the results.csv containing the results of the measurement")
    parser.add_argument("-o", "--outputPath", type=str,
                        help="the path to save the file")
    args = parser.parse_args()

    different_batchsizes = np.array(args.batchsizes)
    ipath = args.inputPath
    opath = args.outputPath
    sample_size = args.numberRuns

    plot_batchsize(different_batchsizes, sample_size, ipath, opath)
