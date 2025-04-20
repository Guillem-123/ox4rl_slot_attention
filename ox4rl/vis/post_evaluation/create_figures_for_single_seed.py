import pandas as pd
import numpy as np
import os
import os.path as osp
from collections import defaultdict
# read csv files of multiple seeds
from ox4rl.eval.post_evaluation.plotting import plot_recall_per_object_type, plot_ap, plot_prec_recall_data, bar_plot

from ox4rl.eval.post_evaluation.eval_multiple_seeds import *
FOLDER = ".."
# FOLDER = "../output/logs"

def get_dataframe(filename, csv_separator, index_col):
    path = osp.join(FOLDER, filename)
    return pd.read_csv(path, sep=csv_separator, index_col=index_col)




def get_dataframe_per_game(filename, csv_separator, index_col, folder):
    games = [
        "pong",
        "boxing",
        "skiing",
        "seaquest",
        # "kangaroo", ERROR
        # "bowling", ERROR
        "freeway",
        "asterix",
    ]
    dataframes_per_game = defaultdict(list)
    for game in games:
        path = osp.join(folder, game, filename)
        dataframes_per_game[game].append(pd.read_csv(path, sep=csv_separator, index_col=index_col))
    return dataframes_per_game

if __name__ == "__main__":


    category = "relevant"
    # eval model
    # dataframe = get_dataframe("test_metrics.csv", csv_separator=";", index_col=None)
    # dataframes_per_game = {"pong": [dataframe]}
    dataframes_per_game = get_dataframe_per_game("test_metrics.csv", csv_separator=";", index_col=None, folder="/home/spoc/ox4rl/output/logs")
    dataframes_per_game = add_contrived_columns(dataframes_per_game)
    merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    bar_plot(final_dataframe_per_game, None, metric="relevant_f_score", title="Localization", ylabel="F-Score (%)")
    plot_prec_recall_data(final_dataframe_per_game)
    plot_ap(final_dataframe_per_game)
    plot_recall_per_object_type(final_dataframe_per_game)
    bar_plot(final_dataframe_per_game, None, metric="relevant_adjusted_mutual_info_score", title="Mutual Information", ylabel="Mutual Information (%)")

    # eval classifier
    # dataframe = get_dataframe("eval_classifier.csv", csv_separator=",", index_col=0) #TODO: change to get_dataframe_per_game
    # dataframes_per_game = {"pong": [dataframe]}
    dataframes_per_game = get_dataframe_per_game("eval_classifier.csv", csv_separator=",", index_col=0, folder="/home/spoc/ox4rl/output/checkpoints")
    dataframes_per_game = preprocess_eval_classifier_dataframes(dataframes_per_game)
    merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    bar_plot(final_dataframe_per_game, None, metric="relevant_accuracy", title="Classifier Accuracy", ylabel="Accuracy (%)")

    # eval model and classifier
    # dataframes_per_game = get_dataframes_per_game("eval_model_and_classifier.csv", csv_separator=",", index_col=0)
    # dataframe = get_dataframe("eval_model_and_classifier.csv", csv_separator=",", index_col=0) #TODO: change to get_dataframe_per_game
    # dataframes_per_game = {"pong": [dataframe]}
    dataframes_per_game = get_dataframe_per_game("eval_model_and_classifier.csv", csv_separator=",", index_col=0, folder="/home/spoc/ox4rl/output/checkpoints")
    merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    bar_plot(final_dataframe_per_game, metric="relevant_f1_score", title="Detection", ylabel="F-Score (%)")
    bar_plot(final_dataframe_per_game, metric="relevant_precision", title="Detection", ylabel="Precision (%)")
    bar_plot(final_dataframe_per_game, metric="relevant_recall", title="Detection", ylabel="Recall (%)")
    print("done")