import copy
import gc
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import argparse

"""
This is the main script to plot experiments. See the ReadMe for examples on how to plot experiments.
"""


def _plot_loss_f1_precision_recall(
    results,
    total_epochs,
    show_average_line,
    ylim,
    result_folder="results",
    filename=None,
):
    """Initiates the plotting the loss, f1, precision and recall.

    Args:
        results (dict): The results of the experiment
        total_epochs (int): total number of epcochs
        show_average_line (Bool): Whether or not to show an average between all plotted lines.
        ylim (int): The y-axis limit of the plot.
        result_folder (str, optional): The folder where the results shall be saved. Defaults to "results".
        filename (str, optional): The name of the file that shall be saved. Defaults to None.
    """

    # Create a 2x4 grid of subplots
    fig, axs = plt.subplots(2, 4, figsize=(20, 12))

    # Plot each metric
    _plot_metric(
        results,
        "train",
        "Training",
        "loss",
        "Loss",
        total_epochs,
        axs[0, 0],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "train",
        "Training",
        "f1",
        "F1 Score",
        total_epochs,
        axs[0, 1],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "train",
        "Training",
        "precision",
        "Precision",
        total_epochs,
        axs[0, 2],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "train",
        "Training",
        "recall",
        "Recall",
        total_epochs,
        axs[0, 3],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "val",
        "Validation",
        "loss",
        "Loss",
        total_epochs,
        axs[1, 0],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "val",
        "Validation",
        "f1",
        "F1 Score",
        total_epochs,
        axs[1, 1],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "val",
        "Validation",
        "precision",
        "Precision",
        total_epochs,
        axs[1, 2],
        ylim,
        show_average_line,
        False,
    )
    _plot_metric(
        results,
        "val",
        "Validation",
        "recall",
        "Recall",
        total_epochs,
        axs[1, 3],
        ylim,
        show_average_line,
        False,
    )

    # Ensure a tight layout
    plt.tight_layout()

    # Save the plot.
    if filename:
        # Ensure the result directory exists
        os.makedirs(result_folder, exist_ok=True)
        filepath = os.path.join(result_folder, filename)
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")  # Optional: Print confirmation
    else:
        plt.show()

    plt.close(fig)


def _plot_loss(
    results,
    total_epochs,
    show_average_line,
    ylim,
    result_folder="results",
    filename=None,
):
    """Initiates the of the only the loss. this is required for experiments where metrics like f1, precision and recall are not available.

    Args:
        results (dict): The results of the experiment
        total_epochs (int): total number of epcochs
        show_average_line (Bool): Whether or not to show an average between all plotted lines
        ylim (int): The y-axis limit of the plot.
        result_folder (str, optional): The folder where the results shall be saved. Defaults to "results".
        filename (str, optional): The name of the file that shall be saved. Defaults to None.
    """

    # Create a 2x1 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))

    # Plot all the losses
    _plot_metric(
        results,
        "train",
        "Training Loss",
        "loss",
        "Loss",
        total_epochs,
        axs[0],
        ylim,
        show_average_line,
        True,
    )
    _plot_metric(
        results,
        "val",
        "Validation Loss",
        "loss",
        "Loss",
        total_epochs,
        axs[1],
        ylim,
        show_average_line,
        True,
    )

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    plt.tight_layout()

    # Save the plots
    if filename:
        # Ensure the result directory exists
        os.makedirs(result_folder, exist_ok=True)
        filepath = os.path.join(result_folder, filename)
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")
    else:
        plt.show()

    plt.close(fig)


def _plot_metric(
    results,
    metric_type,
    metric_name,
    metric,
    metric_label,
    total_epochs,
    ax,
    loss_lim,
    show_average_line,
    log=False,
):
    """
    Plot a specific metric (F1, loss, precision, recall) for all results over epochs on a given axis.
    If desired, plots the average metric across all results with a black line.

    Args:
        results (list): List of results, each with a metrics attribute.
        metric_type (str): One of 'train' or 'val'.
        metric_name (str): A string to use in the title and labels of the plot.
        metric (str): The specific metric to plot ('f1', 'loss', 'precision', 'recall').
        metric_label (str): The label for the y-axis.
        total_epochs (int): The total number of epochs.
        ax (matplotlib.axes.Axes): The axis to plot on.
        loss_lim (int): Since the loss is not restricted to the range of 0 to 1, the loss_lim defines the upper bound.
        show_average_line (Bool): Whether or not to block a black line to represent the average.
        log (Bool, optional): Whether or not to plot the metric in logarithmic scale.
    """
    # Initialize a list to store the average metric values per epoch
    avg_metric_values = []

    # Iterate through each epoch
    for epoch in range(0, total_epochs):
        epoch_metrics = []

        # Collect metric values from all results for the current epoch
        for result in results:
            # Filter metrics for the current epoch
            result_epoch_metrics = [
                (
                    m[metric_type][metric].item()
                    if isinstance(m[metric_type][metric], torch.Tensor)
                    else m[metric_type][metric]
                )
                for m in result["metrics"]
                if m["epoch"] == epoch
            ]
            if result_epoch_metrics:
                epoch_metrics.append(result_epoch_metrics[0])

        # Calculate the average for this epoch
        if epoch_metrics:
            avg_metric_values.append(sum(epoch_metrics) / len(epoch_metrics))
        else:
            avg_metric_values.append(None)

    # Plot individual result metrics
    for result_idx, result in enumerate(results):
        epochs = [m["epoch"] for m in result["metrics"]]
        metric_values = [
            (
                m[metric_type][metric].item()
                if isinstance(m[metric_type][metric], torch.Tensor)
                else m[metric_type][metric]
            )
            for m in result["metrics"]
        ]
        ax.plot(
            epochs,
            metric_values,
            marker=".",
            label=f"{result['c_id']}",
        )

    # Plot the average metric across results with a black line
    if show_average_line == True:
        ax.plot(
            range(0, total_epochs),
            avg_metric_values,
            linestyle="-",
            color="black",
            linewidth=2,
            label="Average",
        )

    ax.set_title(f"{metric_name} {metric_label} per Epoch", fontsize=18)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel(f"{metric_name} {metric_label}", fontsize=16)

    # If no logarithmic scale is used, plot all metrics, with a range of 0 to 1. For the loss this range can differ.
    if log == False:
        ax.set_ylim(0, 1 if metric in ["f1", "precision", "recall"] else loss_lim)
    # If logarithmic scale is used, always refer to the loss limiter.
    else:
        ax.set_ylim(1, loss_lim)
    ax.set_xticks(range(0, total_epochs + 1, 5))
    ax.legend(fontsize=14)
    ax.grid(True)


def _plot_result_table(dataset, results, result_folder, task):
    """Plots the results in a table.

    Args:
        dataset (str): The name of the choosen dataset.
        results (list): List of results, each with a metrics attribute.
        result_folder (str): The place where the table shall be saved to.
        task (str): The name of the task.

    Returns:
        str: If dataset is unknown, the user will be notified
    """

    filename = "Table" + "_" + task

    # Prepare data for Task 1
    def round_metric(value, decimals=3):
        """Ensure all metrics are moved to CPU and converted to NumPy arrays.

        Args:
            value (float): The value to move to cpu and transform into np array.
            decimals (int, optional): Number of decimals. Defaults to 3.

        Returns:
            float: Rounded value
        """
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if isinstance(value, np.ndarray):
            value = value.item()
        return round(value, decimals)

    def bracket_max_value(values):
        """Creates [] around the highest value.

        Args:
            values (float): All values per column

        Returns:
            str: The highest value with brackets around it.
        """
        max_value = max(values)  # Find the maximum value in the list
        return [f"[{value}]" if value == max_value else str(value) for value in values]

    def bracket_min_value(values):
        """Creates [] around the lowest value.

        Args:
            values (float): All values per column

        Returns:
            str: The lowest value with brackets around it.
        """
        min_value = min(values)  # Find the maximum value in the list
        return [f"[{value}]" if value == min_value else str(value) for value in values]

    # If celeba dataset with the second task is processed, only the loss should be shown in the table.
    if dataset == "celeba" and task == "t2":
        loss_values = [
            round_metric(result["test_metrics"]["loss"]) for result in results
        ]

        data = {
            "result": [f"{result['c_id']}" for result in results],
            "Loss": bracket_min_value(loss_values),
        }

    # Otherwise, all values shall be shown in the table.
    elif dataset == "cifar10" or dataset == "celeba":
        loss_values = [
            round_metric(result["test_metrics"]["loss"]) for result in results
        ]
        precision_values = [
            round_metric(result["test_metrics"]["precision"]) for result in results
        ]
        recall_values = [
            round_metric(result["test_metrics"]["recall"]) for result in results
        ]
        f1_values = [round_metric(result["test_metrics"]["f1"]) for result in results]

        data = {
            "Configuration": [f"{result['c_id']}" for result in results],
            "Loss": bracket_min_value(loss_values),
            "Precision": bracket_max_value(precision_values),
            "Recall": bracket_max_value(recall_values),
            "F1": bracket_max_value(f1_values),
        }
    else:
        raise "Dataset Unknown"

    # Convert to DataFrames
    df = pd.DataFrame(data)

    # Plot the table
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.3] + [0.15] * (len(data) - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5) 

    # Ensure the result directory exists and save thet table
    os.makedirs(result_folder, exist_ok=True)
    filepath = os.path.join(result_folder, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close()


def _get_average_metric_of_experiment(exp_name, results_of_experiment, dataset, task):
    """Calculates the average metrics of an experiment for each epoch.

    Args:
        exp_name (str): The name of the experiment.
        results_of_experiment (list): List of results, each with a metrics attribute.
        dataset (str): The name of the dataset.
        task (str): The name of the task.

    Returns:
        dict: Average results for each metric and epoch
    """

    # Generate a copy of the first experiment to keep the structure and give it a new name.
    avg_result = copy.deepcopy(results_of_experiment[0])
    avg_result["c_id"] = exp_name
    epochs = len(results_of_experiment[0]["metrics"])

    # Iterate though all epochs get the average of all metrics for train and plot for every epoch.
    for epoch in range(epochs):
        avg_result["metrics"][epoch]["train"] = _get_average_metrics_for_epoch(
            results_of_experiment, "train", epoch, dataset, task
        )
        avg_result["metrics"][epoch]["val"] = _get_average_metrics_for_epoch(
            results_of_experiment, "val", epoch, dataset, task
        )

    # Since the test metrics do not have any epoch information, not iteration is neccessary. The average of all experiments for a metric is taken.
    if (
        dataset == "celeba" and task == "t2"
    ):  # Since the second task of celeba only has loss metrics.
        avg_result["test_metrics"]["loss"] = sum(
            [exp["test_metrics"]["loss"] for exp in results_of_experiment]
        ) / len(results_of_experiment)
    else:
        avg_result["test_metrics"]["loss"] = sum(
            [exp["test_metrics"]["loss"] for exp in results_of_experiment]
        ) / len(results_of_experiment)
        avg_result["test_metrics"]["precision"] = sum(
            [exp["test_metrics"]["precision"] for exp in results_of_experiment]
        ) / len(results_of_experiment)
        avg_result["test_metrics"]["recall"] = sum(
            [exp["test_metrics"]["recall"] for exp in results_of_experiment]
        ) / len(results_of_experiment)
        avg_result["test_metrics"]["f1"] = sum(
            [exp["test_metrics"]["f1"] for exp in results_of_experiment]
        ) / len(results_of_experiment)

    return avg_result


def _get_average_metrics_for_epoch(results, train_or_val, epoch, dataset, task):
    """Return the average metric of every epoch

    Args:
        results (list): List of results, each with a metrics attribute.
        train_or_val (str): Whether or not the metric type belongs to train or validation
        epoch (int): The current epoch to process
        dataset (str): The name of the datase.
        task (str): The name of the task.

    Returns:
        dict: Returns the averages as a dictionary
    """
    # Initialize sums for each metric
    sum_loss = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    num_results = len(results)

    if dataset == "celeba" and task == "t2":
        # Iterate over results and sum the metrics for the given round
        for result in results:
            round_metrics = result["metrics"][epoch][
                train_or_val
            ]  # Access train_or_val metrics for the given round
            sum_loss += round_metrics["loss"]

        # Calculate the averages
        avg_loss = sum_loss / num_results

        # Return the averages as a dictionary
        return {"loss": avg_loss}

    else:
        # Iterate over results and sum the metrics for the given round
        for result in results:
            round_metrics = result["metrics"][epoch][
                train_or_val
            ]  # Access train_or_val metrics for the given round
            sum_loss += round_metrics["loss"]
            sum_precision += round_metrics["precision"]
            sum_recall += round_metrics["recall"]
            sum_f1 += round_metrics["f1"]

        # Calculate the averages
        avg_loss = sum_loss / num_results
        avg_precision = sum_precision / num_results
        avg_recall = sum_recall / num_results
        avg_f1 = sum_f1 / num_results

        # Return the averages as a dictionary
        return {
            "loss": avg_loss,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }


def _plot(plot_folder, results, dataset, num_epochs, task, show_average_line):
    """Entry plot function to determine which plot functionalities should be called.

    Args:
        plot_folder (str): The folder name where the plots shalled be saved to.
        results (list): List of results, each with a metrics attribute.
        dataset (str): The name of the dataset.
        num_epochs (int): The total number of epochs.
        task (str): The name of the task.
        show_average_line (Bool): Whether or not to block a black line to represent the average.
    """

    # Check which plotting function to call. This is important since the second task of celeba only has loss metrics.
    if task == "t1":
        if dataset == "cifar10":
            _plot_loss_f1_precision_recall(
                results,
                num_epochs,
                show_average_line,
                2,
                plot_folder,
                "Plot" + "_" + task,
            )
        elif dataset == "celeba":
            _plot_loss_f1_precision_recall(
                results,
                num_epochs,
                show_average_line,
                1,
                plot_folder,
                "Plot" + "_" + task,
            )
        else:
            raise "Unkown Dataset"
    elif task == "t2":
        if dataset == "cifar10":
            _plot_loss_f1_precision_recall(
                results,
                num_epochs,
                show_average_line,
                2,
                plot_folder,
                "Plot" + "_" + task,
            )
        elif dataset == "celeba":
            _plot_loss(
                results,
                num_epochs,
                show_average_line,
                10000,
                plot_folder,
                "Plot" + "_" + task,
            )
        else:
            raise "Unkown Dataset"
    else:
        raise "Unkown Task"


def process_experiment_and_task(plot_folder, results, dataset, task):
    """Plot and create tables for a specific experiment and task.

    Args:

        plot_folder (str): The folder name where the plots shalled be saved to.
        results (list): List of results, each with a metrics attribute.
        dataset (str): The name of the dataset.
        task (str): The name of the task.
    """

    num_epochs = len(results[0]["metrics"])
    # Create the plots
    _plot(plot_folder, results, dataset, num_epochs, task, True)
    # Create the tables
    _plot_result_table(dataset, results, plot_folder, task)


def process_across_all_experiments(
    avg_results_per_experiment, plot_folder, dataset, task
):
    """Plot and create tables across all experiments. To do this use the average of each experiment type.

    Args:
        avg_results_per_experiment (list): The average results of each experiment.
        plot_folder (str): The folder name where the plots shalled be saved to.
        dataset (str): The name of the dataset.
        task (str): The name of the task.
    """

    # Ensure that the experiments to compare are trained on the same numbers of epochs so that comparison makes sense.
    num_epochs = len(avg_results_per_experiment[0]["metrics"])
    for avg_result in avg_results_per_experiment:
        if num_epochs != len(avg_result["metrics"]):
            raise "These models have been trained for a different amount of Epochs. Please adjust!"
        num_epochs = len(avg_result["metrics"])

    # Create the plots
    _plot(plot_folder, avg_results_per_experiment, dataset, num_epochs, task, False)
    # Create the tables
    _plot_result_table(dataset, avg_results_per_experiment, plot_folder, task)


def load_saved_models_of_experiment(paths, device):
    """Load the saved models of an experiment.

    Args:
        paths (str): The path to the experiment and model.

    Returns:
        list: List of the loaded models of an experiment.
    """
    # Load the model weights
    results = []
    for path in paths:
        print(f"Loaded {path}")
        with open(path, "rb") as f:
            c = torch.load(f, map_location=torch.device(device))
        results.append(c)
    return results


def main():
    """Main function to control the plotting and table creation. This is triggered via the command line.
    Information about the specific command can be found in the ReadMe.
    """

    # For plotting the cpu is used.
    device = torch.device("cpu")
    print(f"Using Device: {device}")

    print("PyTorch version:", torch.__version__)
    print("CUDA version used by PyTorch:", torch.version.cuda)

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Entry Point to Plot Results")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="The dataset the experiment have been run with",
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=str,
        help="The name of the folder where the experiments are located. Has to be located within the results folder.",
    )
    parser.add_argument(
        "--cross_configuration_plots_and_tables",
        action="store_true",
        help="Whether or not the average between experiments shall be ploted too.",
    )
    parser.add_argument(
        "--vag",
        action="store_true",
        help="Whether the the experiments focused on different aggregation techniques shall be visualized.",
    )
    parser.add_argument(
        "--vepr",
        action="store_true",
        help="Whether the the experiments focused on varying epochs shall be visualized.",
    )
    parser.add_argument(
        "--vbl",
        action="store_true",
        help="Whether the the experiments focused on varying backbone layers shall be visualized.",
    )
    
    parser.add_argument("--vag_inputs", type=str, nargs='+', help="Specific Model Folder")
    parser.add_argument("--vepr_inputs", type=str, nargs='+', help="Specific Model Folder")
    parser.add_argument("--vbl_inputs", type=str, nargs='+', help="Specific Model Folder")

    args = parser.parse_args()
    t1 = "t1"
    t2 = "t2"

    results_t1 = {}
    results_t2 = {}
    avg_results_per_experiment_t1 = []
    avg_results_per_experiment_t2 = []

    if (
        args.vag_inputs == None
        and args.vepr_inputs == None
        and args.vbl_inputs == None
    ):
        raise "Please Provide at least one experiment to plot"

    def process(path, name):
        experiment = path

        # Load the model metrocs for each experiment.
        path = os.path.join("results", args.folder, path)
        file_paths_t1 = [
            os.path.join(path, t1, f)
            for f in sorted(os.listdir(os.path.join(path, t1)))
            if os.path.isfile(os.path.join(path, t1, f))
        ]
        file_paths_t2 = [
            os.path.join(path, t2, f)
            for f in sorted(os.listdir(os.path.join(path, t2)))
            if os.path.isfile(os.path.join(path, t2, f))
        ]
        results_t1[name] = load_saved_models_of_experiment(file_paths_t1, device)
        results_t2[name] = load_saved_models_of_experiment(file_paths_t2, device)

        # Plot and Create the Table an experiment and task 1.
        process_experiment_and_task(
            os.path.join("results", args.folder, experiment),
            results_t1[name],
            args.dataset,
            "t1",
        )

        # Plot and Create the Table an experiment and task 2.
        process_experiment_and_task(
            os.path.join("results", args.folder, experiment),
            results_t2[name],
            args.dataset,
            "t2",
        )

        # Calculate the average metrics of all clients of an experiment.
        # This is needed to plot the results across different experiments.
        for exp_name, exp in results_t1.items():
            avg_results_per_experiment_t1.append(
                _get_average_metric_of_experiment(exp_name, exp, args.dataset, "t1")
            )
        for exp_name, exp in results_t2.items():
            avg_results_per_experiment_t2.append(
                _get_average_metric_of_experiment(exp_name, exp, args.dataset, "t2")
            )

        # Clean up the memory usage.
        del results_t1[name]
        del results_t2[name]
        gc.collect()
    
    # Experiments with same aggregation technique but different epochs per communication round.
    if args.vepr == True:
        for path in args.vepr_inputs:
            process(path, path)
    
    elif args.vbl == True:
        for path in args.vbl_inputs:
            process(path, path)

    # Experiments with different aggregation technqiques but same epochs per communication round.
    elif args.vag == True:
        for path in args.vag_inputs:
            process(path, path)
    
    else:
        raise "Please set either --vag (for aggregation techniquese experiment), --vbl (for backbone layer experiment) or --vepr (for aggregatioon round experiment)"

    # Plot task 1 across all experiments by using the previously calculated averages.
    process_across_all_experiments(
        avg_results_per_experiment_t1,
        os.path.join("results", args.folder, "cross_configuration_plots_and_tables"),
        args.dataset,
        "t1",
    )

    # Plot task 2 across all experiments by using the previously calculated averages.
    process_across_all_experiments(
        avg_results_per_experiment_t2,
        os.path.join("results", args.folder, "cross_configuration_plots_and_tables"),
        args.dataset,
        "t2",
    )


    # Clean up the memory usage.
    del avg_results_per_experiment_t1
    del avg_results_per_experiment_t2
    gc.collect()

if __name__ == "__main__":
    main()