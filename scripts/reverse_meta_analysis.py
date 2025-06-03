import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

OUTPUT_FOLDER = Path(os.getcwd()) / "output"
DATA_FOLDER = Path(os.getcwd()) / "data"


def run_reverse_meta_analysis(output_folder=OUTPUT_FOLDER, data_folder=DATA_FOLDER):
    if not output_folder.exists():
        output_folder.mkdir()
    output_summaries = []
    for file in data_folder.glob("*.nii"):
        output_summary = f"{output_folder / file.stem}_summary"
        command = (
            "b2rio "
            f"--brain_path {file} "
            f"--output_file {output_folder / file.stem} "
            f"--output_summary f{output_summary}"
        )
        print("Running command:\n", command)
        subprocess.call(command, shell=True)
        print("=" * 20)
        output_summaries.append(output_summary + ".csv")


def get_colordict(palette, number, start):
    pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
    color_d = dict(enumerate(pal, start=start))
    return color_d


def plot_reverse_meta_analysis(summary_fname):
    df = (
        pd.read_csv(summary_fname)
        .sort_values(by="mean", ascending=False)
        .head(6)
        .sort_values(by="mean", ascending=True)
    )
    df["term"] = df["term"].str.capitalize()

    n = df["mean"].shape[0]
    n_col = 1

    color_dict = get_colordict("viridis", n, 0)
    x_lim = max(df["mean"])

    # Circle patch
    N = n
    M = 1

    x = list(df["mean"])
    y = list(range(0, len(x)))
    labels = list(df["term"])
    labels.insert(0, "")
    labels.append("")
    c = np.arange(1, 15, 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    circles = [plt.Circle((0.5, i), radius=r / 8) for i, r in zip(y, x)]
    col = PatchCollection(circles, array=c, cmap="cool")
    ax.add_collection(col)

    ax.set(xticks=np.arange(M + 1), yticks=np.arange(-1, N + 1))
    ax.set_xticks([])
    ax.set_yticklabels(labels, fontsize=12)
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["bottom"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.grid(which="minor")
    plt.savefig(str(summary_fname).replace(".csv", "_circles.pdf"))

    index_list = [[i[0], i[-1] + 1] for i in np.array_split(range(n), n_col)]

    fig, axs = plt.subplots(1, n_col, figsize=(16, 8), facecolor="white", squeeze=False)
    for col, idx in zip(range(0, 5), index_list):
        df_tmp = df[idx[0] : idx[-1]]
        idx_df_tmp = list(df_tmp.index)
        color_l = [color_dict.get(i) for i in idx_df_tmp]
        x = list(df_tmp["mean"])
        y = list(range(0, len(x)))

        sns.barplot(
            x=x,
            y=y,
            data=df_tmp,
            alpha=0.9,
            orient="h",
            ax=axs[0][col],
            palette=color_l,
            hue=y,
            legend=False,
        )
        axs[0][col].set_xlim(0, x_lim)  # set X axis range max: x_lim or n/100+1
        axs[0][col].set_yticks(axs[0][col].get_yticks())
        axs[0][col].set_yticklabels(df_tmp["term"], fontsize=12)
        axs[0][col].spines["bottom"].set_color("white")
        axs[0][col].spines["right"].set_color("white")
        axs[0][col].spines["top"].set_color("white")
        axs[0][col].spines["left"].set_color("white")

    plt.tight_layout()
    plt.savefig(str(summary_fname).replace(".csv", "_bars.pdf"))


if __name__ == "__main__":
    output_summaries = run_reverse_meta_analysis()
    for output_summary in output_summaries:
        plot_reverse_meta_analysis(output_summary)
