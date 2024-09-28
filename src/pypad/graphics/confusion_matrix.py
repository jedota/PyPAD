import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from pypad.graphics.aesthetics import set_colour_theme

logger = logging.getLogger("pypad.graphics.confusion_matrix")


def plot_confusion_matrix(
    y_true,
    y_score,
    bonafide_label=1,
    threshold=0.5,
    class_names=None,
    xlabel="Predicted",
    ylabel="Ground Truth",
    figsize=(10, 10),
    context="notebook",
    fontsize=36,
    font_scale=2,
    draw_zeros=True,
    colour_theme="paper",
    cmap=None,
):
    """Plots a multiclass classification confusion matrix.

    Parameters
    ----------
    y_true : [type]
        [description]
    y_score : [type]
        [description]
    bonafide_label : int, optional
        [description], by default 1
    threshold : float, optional
        [description], by default 0.5
    class_names : [type], optional
        [description], by default None
    xlabel : str, optional
        Label of the abscissa, by default "Predicted"
    ylabel : str, optional
        Label of the ordinate, by default "Ground Truth"
    figsize : tuple, optional
        Figure size, by default (10, 10)
    context : str, optional
        Seaborn plot context. must be None, or one of "paper", "notebook", "talk" or
        "poster". By default "notebook"
    fontsize : int, optional
        Font size, by default 36
    font_scale : int, optional
        Font scale, by default 2
    draw_zeros : bool, optional
        [description], by default True
    colour_theme : str, optional
        Plot colour style, must be a valid entry from
        pypad.graphics.aesthetics.set_colour_theme(). None or invalid values will load
        the default "light" colour mode. By default "paper"
    cmap : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    y_pred = []
    labels = np.unique(y_true)
    mask = ~(labels != bonafide_label)
    for score in y_score:
        if score[..., bonafide_label] > threshold:
            y_pred.append(bonafide_label)
        else:
            masked = np.ma.masked_array(score, mask=mask)
            y_pred.append(np.argmax(masked))

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    if class_names is None:
        class_names = labels.astype(str)
    return _confusion_matrix(
        cm,
        class_names,
        xlabel,
        ylabel,
        figsize,
        context,
        fontsize,
        font_scale,
        draw_zeros,
        colour_theme,
        cmap,
    )


def plot_system_confusion_matrix(
    attack_scores,
    bonafide_scores,
    threshold=0.5,
    class_names=np.array(["Bona fide", "Attack"]),
    xlabel="Predicted",
    ylabel="Ground Truth",
    figsize=(10, 10),
    context="notebook",
    fontsize=36,
    font_scale=2,
    draw_zeros=True,
    colour_theme="paper",
    cmap=None,
):
    """Plots a binary classification confusion matrix to assess the performance of a
    biometric system

    Parameters
    ----------
    attack_scores : [type]
        [description]
    bonafide_scores : [type]
        [description]
    threshold : float, optional
        [description], by default 0.5
    class_names : [type], optional
        [description], by default np.array(["Bona fide", "Attack"])
    xlabel : str, optional
        Label of the abscissa, by default "Predicted"
    ylabel : str, optional
        Label of the ordinate, by default "Ground Truth"
    figsize : tuple, optional
        Figure size, by default (10, 10)
    context : str, optional
        Seaborn plot context. must be None, or one of "paper", "notebook", "talk" or
        "poster". By default "notebook"
    fontsize : int, optional
        Font size, by default 36
    font_scale : int, optional
        Font scale, by default 2
    draw_zeros : bool, optional
        [description], by default True
    colour_theme : str, optional
        Plot colour style, must be a valid entry from
        pypad.graphics.aesthetics.set_colour_theme(). None or invalid values will load
        the default "light" colour mode. By default "paper"
    cmap : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    # tp = [s > threshold for s in bonafide_scores].count(True)
    # fp = [s <= threshold for s in attack_scores].count(True)
    # fn = [s <= threshold for s in bonafide_scores].count(True)
    # tn = [s > threshold for s in attack_scores].count(True)
    tp = [s >= threshold for s in bonafide_scores].count(True)
    fp = [s >= threshold for s in attack_scores].count(True)
    fn = [s < threshold for s in bonafide_scores].count(True)
    tn = [s < threshold for s in attack_scores].count(True)

    cm = np.array([[tp, fn], [fp, tn]])
    return _confusion_matrix(
        cm,
        class_names,
        xlabel,
        ylabel,
        figsize,
        context,
        fontsize,
        font_scale,
        draw_zeros,
        colour_theme,
        cmap,
    )


def _confusion_matrix(
    cm,
    class_names,
    xlabel,
    ylabel,
    figsize,
    context,
    fontsize,
    font_scale,
    draw_zeros,
    colour_theme,
    cmap,
):
    sns.set_context(context, font_scale=font_scale)
    style, _ = set_colour_theme(theme=colour_theme)
    sns.set_style(style)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    h, w = cm.shape[:2]
    for i in range(w):
        for j in range(h):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = f"{c}/{s[0]}\n{p:.1f}%"
            elif c == 0 and not draw_zeros:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{c}\n{p:.1f}%"

    cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm.columns.name = xlabel
    cm.index.name = ylabel

    _, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(
        cm,
        cmap=cmap,
        annot=annot,
        fmt="",
        annot_kws={"size": fontsize},
        ax=ax,
    )
    fig = heatmap.get_figure()
    fig.patch.set_alpha(0)

    return fig
