
from config import *

import pandas as pd
import matplotlib.pyplot as plt


def display_plot(dataframe, names, title, columns_to_display, unit):

    plt.figure(figsize=(8, 5))

    if len(dataframe) == 1:
        for col in columns_to_display:
            plt.plot(dataframe[0][col], label=col)
        plt.title(f"stats for : {title} on model {names[0]}")
    else:
        for col in columns_to_display:
            for i in range (len(dataframe)):
                plt.plot(dataframe[i][col], label=col+" for "+names[i])

        plt.title(f"stats for : {title} on all models")
    plt.xlabel("Index")
    plt.ylabel(unit)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if show_every_models_statistic:
    all_dataframes = [pd.read_csv(this_folder_path+"/statistics.csv") for this_folder_path in every_models_statistics_path]

    display_plot(all_dataframes, all_models,"median", ["median_real", "median_fake"], "output")
    display_plot(all_dataframes, all_models,"mean", ["mean_real", "mean_fake"], "output")
    display_plot(all_dataframes, all_models, "losses", ["gen_loss", "disc_loss"], "loss")
    display_plot(all_dataframes, all_models, "time", ["time"], "seconds")

else:
    current_model_dataframe = [pd.read_csv(statistics_file_path)]
    display_plot(current_model_dataframe, [model_name],"median/mean", ["median_real", "median_fake", "mean_real", "mean_fake"], "output")
    display_plot(current_model_dataframe,[model_name], "losses", ["gen_loss", "disc_loss"], "loss")
    display_plot(current_model_dataframe,[model_name], "time", ["time"], "seconds")
