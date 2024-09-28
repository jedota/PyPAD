import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from pypad.metrics.det_curve import det_curve, eer
from pypad.graphics.aesthetics import set_colour_theme

sns.set(style="darkgrid")
logger = logging.getLogger("pypad.graphics.det_plot")


class DETPlot:
    """A class for plotting Detection Error Tradeoff curves, used to visually assess the
    performance of operating points for various security measures and convenience
    measures of different detection algorithms. The DET curve differs from the ROC by
    axis warping using the probit function.

    The axes are scaled and labelled so that a normal Gaussian distribution will plot as
    a straight line.

    For more information visit:
    - https://sites.google.com/site/bosaristoolkit/home/bosaristoolkit_userguide.pdf
    - https://christoph-busch.de/files/Busch-TelAviv-PAD-180116.pdf

    In this class, systems can be added using the set_system() function, and the curve
    can be plotted using the plot() function.
    """

    def __init__(
        self,
        title="DET Curve",
        xlabel="APCER (in %)",
        ylabel="BPCER (in %)",
        figsize=(10, 10),
        context="notebook",
        font_scale=1.5,
        linewidth=4,
        plot_100=False,
    ):
        """Class instantiation. Visual details such as title, axis labels, size,
        plotting context, font size, and line width can be changed when creating the
        class, or accessed later in the standard Python fashion. Note that the context
        parameter must be None, or one of "paper", "notebook", "talk" or "poster", which
        are supported by the Seaborn library.

        Parameters
        ----------
        title : str, optional
            Plot title, by default "DET Curve"
        xlabel : str, optional
            Label of the abscissa, by default "APCER (in %)"
        ylabel : str, optional
            Label of the ordinate, by default "BPCER (in %)"
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

        # Changing these is not recommended
        if not plot_100:
            self.xlim = np.array([1e-4, 5e-1])
            self.ylim = np.array([1e-4, 5e-1])
            ticks = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
            tick_labels = np.array(
                ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"]
            )
        else:
            self.xlim = np.array([1e-4, 0.99])
            self.ylim = np.array([1e-4, 0.99])
            ticks = np.array(
                [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 0.99]
            )
            tick_labels = np.array(
                ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
            )
        self.xticks = ticks
        self.yticks = ticks
        self.xticklabels = tick_labels
        self.yticklabels = tick_labels
        self.systems = {}

    def set_system(
        self, attack_scores, bonafide_scores, drop_intermediate=False, label=None
    ):
        """Adds a system to the dictionary, automatically assigning a label to it, if
        not specified. False Positive and False Negative scores are computed using the
        DET function.

        Parameters
        ----------
        attack_scores : numpy.ndarray
            Array object of score probabilities for all PAI species class
        bonafide_scores : numpy.ndarray
            Array object of bona fide score probabilities
        drop_intermediate : bool, optional
            Whether reduntant points in the curve should be dropped, by default False
        label : str, optional
            System name, by default None
        """
        if not label:
            label = f"System {len(self.systems) + 1}"

        fpr, fnr, thresholds = det_curve(
            attack_scores, bonafide_scores, drop_intermediate=drop_intermediate
        )
        eer_, eer_thres = eer(fpr, fnr, thresholds, percentage=True)

        self.systems[label] = {
            "fpr": fpr,
            "fnr": fnr,
            "thresholds": thresholds,
            "eer": eer_,
            "eer_thres": eer_thres,
        }

    def set_system_pais(
        self,
        attack_true,
        attack_scores,
        bonafide_scores,
        pais_names=None,
        drop_intermediate=False,
        label=None,
        pais_label=None,
    ):
        """Adds a PAD system with multiple PAIS to the dictionary.

        This functions calls set_system() internally, which automatically assigns a
        label to the system, if not specified. Aditionally, the corresponding PAIS is
        set to the label for each curve, so you do not need to do this yourself.

        Parameters
        ----------
        attack_true : numpy.ndarray
            Array object of ground truth labels for multiple PAI species class
        attack_scores : numpy.ndarray
            Array object of score probabilities for multiple PAI species class
        bonafide_scores : numpy.ndarray
            Array object of bona fide score probabilities
        drop_intermediate : bool, optional
            Whether reduntant points in the curve should be dropped, by default False
        label : str, optional
            System name, by default None
        """
        pais = np.unique(attack_true)
        if pais_names is None:
            pais_names = pais.astype(str)
        if label is None:
            label = f"System {len(self.systems) + 1}"
        if pais_label is None:
            pais_label = "; PAIS "
        for species in pais:
            pais_scores = attack_scores[attack_true == species]
            self.set_system(
                pais_scores,
                bonafide_scores,
                drop_intermediate=drop_intermediate,
                label=f"{label}{pais_label}{pais_names[species]}",
            )

    def plot(self, colour_theme="paper"):
        """Plots the DET curve for all the systems added through the function
        set_system() or set_system_pais(). Scaling is done using the probit (ppf)
        function.

        Parameters
        ----------
        colour_theme : str, optional
            Plot colour mode, must be a valid entry from
            pypad.graphics.aesthetics.set_colour_mode(). None or invalid values will
            load the default "paper" colour mode

        Returns
        -------
        matplotlib.figure.Figure
            The DET curve figure
        """
        sns.set_context(
            self.context,
            font_scale=self.font_scale,
            rc={"lines.linewidth": self.linewidth},
        )
        style, palette = set_colour_theme(theme=colour_theme)
        sns.set_style(style)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xscale("function", functions=(norm.ppf, norm.cdf))
        ax.set_yscale("function", functions=(norm.ppf, norm.cdf))

        ax.set_title(self.title, fontsize="x-large")
        ax.set_xlabel(self.xlabel, fontsize="large")
        ax.set_ylabel(self.ylabel, fontsize="large")
        ax.set(xlim=self.xlim, ylim=self.ylim)
        ax.set_xticks(self.xticks)
        ax.set_yticks(self.yticks)
        ax.set_xticklabels(self.xticklabels)
        ax.set_yticklabels(self.yticklabels)

        # Plot EER
        eer_ = np.linspace(-1, 1, 10000)
        ax.plot(
            eer_,
            eer_,
            color=style["eer.color"],
            linewidth=-(-self.linewidth // 4),
            label="Equal Error Rate",
        )

        # Plot BPCER10 and 20
        ax.axvline(
            x=0.05, color="black", linewidth=-(-self.linewidth // 2), linestyle="--"
        )
        ax.axvline(
            x=0.1, color="black", linewidth=-(-self.linewidth // 2), linestyle="--"
        )

        data = pd.DataFrame(
            [
                (system, fpr, fnr)
                for system in self.systems
                for fpr, fnr in zip(
                    self.systems[system]["fpr"], self.systems[system]["fnr"]
                )
            ],
            columns=["system", "fpr", "fnr"],
        )
        sns.lineplot(
            data=data,
            x="fpr",
            y="fnr",
            hue="system",
            style="system",
            palette=palette,
            errorbar=None,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))
        for i, label in enumerate(labels[:-1]):
            labels[i] = f"{label} ({self.systems[label]['eer']:.2f}%)"
        ax.legend(handles=handles, labels=labels, loc=0).get_frame().set_edgecolor(
            style["legend.edgecolor"]
        )
        fig.patch.set_alpha(0)

        return fig

    def close(self):
        plt.close()
