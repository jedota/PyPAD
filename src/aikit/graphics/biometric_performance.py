import logging

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from aikit.evaluation.scores import pad_scores, split_scores
from aikit.graphics.aesthetics import set_colour_theme

sns.set(style="darkgrid")
logger = logging.getLogger("aikit.graphics.biometric_performance")


def performance_evaluation(
    data,
    threshold=0.5,
    log_scale=False,
    colour_theme="paper",
    bbox=True,
    figsize=(12, 9),
    context="notebook",
    font_scale=1.5,
    linewidth=3,
):
    sns.set_context(context, font_scale=font_scale)
    style, _ = set_colour_theme(theme=colour_theme)
    palette = sns.color_palette(["#fb6962", "#79de79", "#fcfc99", "#a8e4ef", "#000000"])
    sns.set_style(style)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlim=np.array([0, 1]), xticks=np.arange(0, 1.1, 0.1))
    if log_scale:
        ax.set_yscale("log")
    plt.xlabel("Comparison Score")
    plt.ylabel("Probability Density")

    sns.kdeplot(data=data, common_norm=False, linewidth=0, ax=ax)

    apx, apy = ax.get_lines()[0].get_data()
    bfx, bfy = ax.get_lines()[1].get_data()
    ax.fill_between(apx, apy, where=(apx < threshold), alpha=0.7, facecolor=palette[0])
    ax.fill_between(bfx, bfy, where=(bfx >= threshold), alpha=0.7, facecolor=palette[1])
    ax.fill_between(bfx, bfy, where=(bfx < threshold), alpha=0.7, facecolor=palette[2])
    ax.fill_between(apx, apy, where=(apx >= threshold), alpha=0.7, facecolor=palette[3])

    ax.axvline(threshold, color=palette[4], linestyle="--", linewidth=linewidth)
    if bbox:
        xbbox = threshold + 0.025 if threshold <= 0.5 else threshold - 0.170
        ax.text(
            xbbox,
            0.5,
            r"$\tau = %.4f$" % threshold,
            va="center",
            transform=ax.transAxes,
            bbox={
                "facecolor": style["axes.facecolor"],
                "edgecolor": palette[4],
                "alpha": 0.8,
                "pad": 4,
            },
        )

    ap_handle = mpatches.Patch(color=palette[0], label="Attack presentation")
    bf_handle = mpatches.Patch(color=palette[1], label="Bona fide")
    fr_handle = mpatches.Patch(color=palette[2], label="False rejection")
    fa_handle = mpatches.Patch(color=palette[3], label="False acceptance")
    th_handle = mlines.Line2D(
        [],
        [],
        color=palette[4],
        linestyle="--",
        label=r"Threshold at $\tau = %.2f$" % threshold,
    )
    plt.legend(handles=[ap_handle, bf_handle, fr_handle, fa_handle, th_handle], loc=0)

    fig.patch.set_alpha(0)
    return fig


def performance_evaluation_morphing(
    data,
    title="Morphing Evaluation",
    threshold=0.5,
    log_scale=False,
    bbox=True,
    colour_theme="paper",
    figsize=(12, 9),
    context="notebook",
    font_scale=1.5,
    linewidth=3,
):
    sns.set_context(context, font_scale=font_scale)
    style, _ = set_colour_theme(theme=colour_theme)
    palette = sns.color_palette(
        [
            "#79de79",
            "#fb6962",
            "#fcfc99",
            "#000000",
        ]
    )
    sns.set_style(style)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlim=np.array([0, 1]), xticks=np.arange(0, 1.1, 0.1))
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize="x-large")
    plt.xlabel("Comparison Score", fontsize="large")
    plt.ylabel("Probability Density", fontsize="large")

    sns.kdeplot(
        data=data,
        palette=palette[:-1],
        common_norm=False,
        fill=True,
        alpha=0.5,
        linewidth=0,
        ax=ax,
    )

    ax.axvline(
        threshold,
        color=palette[3],
        linestyle="--",
        linewidth=linewidth,
    )
    if bbox:
        xbbox = threshold + 0.025 if threshold <= 0.5 else threshold - 0.170
        ax.text(
            xbbox,
            0.5,
            r"$\tau = %.4f$" % threshold,
            va="center",
            transform=ax.transAxes,
            bbox={
                "facecolor": style["axes.facecolor"],
                "edgecolor": palette[3],
                "alpha": 0.8,
                "pad": 4,
            },
        )

    bfp_handle = mpatches.Patch(color=palette[0], label="Mated")
    atk_handle = mpatches.Patch(color=palette[1], label="Attack")
    imp_handle = mpatches.Patch(color=palette[2], label="Non-mated")
    thr_handle = mlines.Line2D(
        [],
        [],
        color=palette[3],
        linestyle="--",
        label=r"Threshold at $\tau = %.2f$" % threshold,
    )
    ax.legend(handles=[bfp_handle, atk_handle, imp_handle, thr_handle], loc=0)

    fig.patch.set_alpha(0)
    return fig
