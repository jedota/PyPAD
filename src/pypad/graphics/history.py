import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from packaging import version
from tensorflow import __version__ as tfver

from pypad.graphics.aesthetics import set_colour_theme

logger = logging.getLogger("pypad.graphics.history")


def plot_history(
    history,
    title="Training Loss & Accuracy",
    xlabel="Epoch",
    ylabel="History",
    figsize=(16, 9),
    context="notebook",
    font_scale=1.5,
    linewidth=4,
    colour_theme="paper",
):
    sns.set_context(context, font_scale=font_scale, rc={"lines.linewidth": linewidth})
    style, palette = set_colour_theme(theme=colour_theme)
    sns.set_style(style)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize="x-large")
    ax.set_xlabel(xlabel, fontsize="large")
    ax.set_ylabel(ylabel, fontsize="large")

    # epochs = len(history.history["loss"])
    if version.parse(tfver) >= version.parse("2.0.0"):
        data = pd.DataFrame(
            [
                (loss, val_loss, acc, val_acc)
                for loss, val_loss, acc, val_acc in zip(
                    history.history["loss"],
                    history.history["val_loss"],
                    history.history["accuracy"],
                    history.history["val_accuracy"],
                )
            ],
            columns=["Train Loss", "Val. Loss", "Train Acc.", "Val. Acc."],
        )
    else:
        data = pd.DataFrame(
            [
                (loss, val_loss, acc, val_acc)
                for loss, val_loss, acc, val_acc in zip(
                    history.history["loss"],
                    history.history["val_loss"],
                    history.history["acc"],
                    history.history["val_acc"],
                )
            ],
            columns=["Train Loss", "Val. Loss", "Train Acc.", "Val. Acc."],
        )
    sns.lineplot(data=data, palette=palette, ax=ax)
    fig.patch.set_alpha(0)

    return fig
