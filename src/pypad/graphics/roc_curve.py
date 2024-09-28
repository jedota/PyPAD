import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from pypad.graphics.aesthetics import set_colour_theme

logger = logging.getLogger("pypad.graphics.roc_curve")


class ROCCurve:
    """A class for plotting Receiver Operating Characteristic curves.

    In this class, systems can be added using the add_system() function, and the curve
    can be plotted using the plot() function.
    """

    def __init__(
        self,
        title="ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        figsize=(10, 10),
        context="notebook",
        font_scale=1.5,
        linewidth=4,
    ):
        """Class instantiation. Visual details such as title, axis labels, size,
        plotting context, font size, and line width can be changed when creating the
        class, or accessed later in the standard Python fashion. Note that the context
        parameter must be None, or one of 'paper', 'notebook', 'talk' or 'poster', which
        are supported by the Seaborn library.

        Parameters
        ----------
        title : str, optional
            Plot title, by default "ROC Curve"
        xlabel : str, optional
            Label of the abscissa, by default "False Positive Rate"
        ylabel : str, optional
            Label of the ordinate, by default "True Positive Rate"
        figsize : tuple, optional
            Figure size, by default (10, 10)
        context : str, optional
            Seaborn plot context. must be None, or one of "paper", "notebook", "talk" or
            "poster". By default "notebook"
        font_scale : float, optional
            Font scale, by default 1.5
        linewidth : int, optional
            Line width for each plotted system, by default 4
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.context = context
        self.font_scale = font_scale
        self.linewidth = linewidth

        # Changing these is not recommended.
        self.xlim = np.array([0, 1])
        self.ylim = np.array([0, 1])
        self.xticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.yticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.systems = {}

    def set_system(self, y_true, y_score, bonafide_class=1, label=None):
        """Adds a system to the dictionary, automatically assigning a label to it, if
        not specified.

        Parameters
        ----------
        y_true : numpy.ndarray
            True (ground truth) labels
        y_score : numpy.ndarray
            Target scores (confidence values) of the bona fide class
        bonafide_class : int, optional
            Target class, by default 1
        label : str, optional
            System name, by default None
        """
        if not label:
            label = f"System {len(self.systems) + 1}"

        fpr, tpr, thresholds = roc_curve(
            y_true, y_score, pos_label=bonafide_class, drop_intermediate=True
        )

        self.systems[label] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    def plot(self, colour_theme="paper"):
        """Plots the ROC curve for all the systems added through the function
        set_system().

        Parameters
        ----------
        colour_theme : str, optional
            Plot colour mode, must be a valid entry from
            pypad.graphics.aesthetics.set_colour_mode(). None or invalid values will
            load the default "paper" colour mode

        Returns
        -------
        matplotlib.figure.Figure
            The ROC curve figure
        """
        sns.set_context(
            self.context,
            font_scale=self.font_scale,
            rc={"lines.linewidth": self.linewidth},
        )
        style, palette = set_colour_theme(theme=colour_theme)
        sns.set_style(style)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(self.title, fontsize="x-large")
        ax.set_xlabel(self.xlabel, fontsize="large")
        ax.set_ylabel(self.ylabel, fontsize="large")
        ax.set(xlim=self.xlim, ylim=self.ylim)
        ax.set_xticks(self.xticks)
        ax.set_yticks(self.yticks)

        data = pd.DataFrame(
            [
                (system, fpr, tpr)
                for system in self.systems
                for fpr, tpr in zip(
                    self.systems[system]["fpr"], self.systems[system]["tpr"]
                )
            ],
            columns=["system", "fpr", "tpr"],
        )
        sns.lineplot(
            data=data,
            x="fpr",
            y="tpr",
            hue="system",
            style="system",
            palette=palette,
            ci=None,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0).get_frame().set_edgecolor(
            style["legend.edgecolor"]
        )
        fig.patch.set_alpha(0)

        return fig

    def close(self):
        plt.close()
