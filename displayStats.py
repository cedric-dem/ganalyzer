from config import *

import pandas as pd
import matplotlib.pyplot as plt

def display_plot(dataframe, title, columns_to_display, unit):
    plt.figure(figsize=(8, 5))
    for col in columns_to_display:
        plt.plot(dataframe[col], label=col)
    plt.title(f"stats for : {title}")
    plt.xlabel("Index")
    plt.ylabel(unit)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

df = pd.read_csv(statistics_file_path)

display_plot(df, "median/mean",["median_real","median_fake","mean_real","mean_fake"], "output")
display_plot(df, "losses",["gen_loss","disc_loss"], "loss")
display_plot(df, "time",["time"], "seconds")

