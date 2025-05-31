from config import *

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(statistics_file_path)

groups = {
    "median/mean": ["median_real","median_fake","mean_real","mean_fake"],
    "losses": ["gen_loss","disc_loss"],
    "time":["time"]
}

for titre, column in groups.items():
    plt.figure(figsize=(8, 5))
    for col in column:
        plt.plot(df[col], label=col)
    plt.title(f"stats for : {titre}")
    plt.xlabel("Index")
    plt.ylabel("quantity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

