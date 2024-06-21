import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aikit.metrics.det_curve import det_curve
from aikit.metrics.iso_30107_3 import bpcer_ap
from aikit.graphics.aesthetics import set_colour_theme

sns.set(style="darkgrid")
logger = logging.getLogger("aikit.graphics.erc_plot")


class ERCPlot:
    """[summary]"""

    def __init__(
        self,
        attack_potential=10,
        title="Error vs. Reject Curve",
        xlabel="Ratio of Unconsidered Images (in %)",
        ylabel=None,
        figsize=(10, 10),
        context="notebook",
        font_scale=1.5,
        linewidth=4,
    ):
        """[summary]

        Parameters
        ----------
        attack_potential : int, optional
            [description], by default 10
        title : str, optional
            [description], by default "Error vs. Reject Curve"
        xlabel : str, optional
            [description], by default "Ratio of Unconsidered Images (in %)"
        ylabel : [type], optional
            [description], by default None
        figsize : tuple, optional
            [description], by default (10, 10)
        context : str, optional
            [description], by default "notebook"
        font_scale : float, optional
            [description], by default 1.5
        linewidth : int, optional
            [description], by default 4
        """
        self.attack_potential = attack_potential
        self.title = title
        self.xlabel = xlabel
        self.ylabel = f"BPCER{attack_potential} (in %)"
        self.figsize = figsize
        self.context = context
        self.font_scale = font_scale
        self.linewidth = linewidth

        # Changing these is not recommended
        self.xlim = np.array([0, 95])
        self.xlim = np.array([0, 55])
        # self.ylim = np.array([1e-4, 5e-1])

        self.xticks = np.linspace(0, 90, num=19, dtype=np.uint8)
        self.xticks = np.linspace(0, 50, num=11, dtype=np.uint8)
        # self.yticks = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
        # self.yticklabels = np.array(
        #     ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"]
        # )
        self.systems = {}

    def set_system(
        self,
        attack_scores,
        bonafide_scores,
        attack_magnitudes,
        bonafide_magnitudes,
        drop_intermediate=False,
        label=None,
    ):
        """[summary]

        Parameters
        ----------
        attack_scores : [type]
            [description]
        bonafide_scores : [type]
            [description]
        drop_intermediate : bool, optional
            [description], by default False
        label : [type], optional
            [description], by default None
        """
        if not label:
            label = f"System {len(self.systems) + 1}"

        sort_arr = np.argsort(attack_magnitudes)[::-1]
        attack_scores = np.take_along_axis(attack_scores, sort_arr, axis=0)
        attack_magnitudes = np.take_along_axis(attack_magnitudes, sort_arr, axis=0)

        sort_arr = np.argsort(bonafide_magnitudes)[::-1]
        bonafide_scores = np.take_along_axis(bonafide_scores, sort_arr, axis=0)
        bonafide_magnitudes = np.take_along_axis(bonafide_magnitudes, sort_arr, axis=0)

        unconsidered_ratio = np.linspace(1, 0, num=21)
        bpcer = np.zeros(len(unconsidered_ratio), dtype=np.float64)

        for i, ratio in enumerate(unconsidered_ratio):
            unconsidered_attack_images = int(np.around(len(attack_scores) * ratio))
            unconsidered_bonafide_images = int(np.around(len(bonafide_scores) * ratio))
            if unconsidered_attack_images == 0 or unconsidered_bonafide_images == 0:
                bpcer[i] = bpcer[i - 1]
                continue
            fpr, fnr, thresholds = det_curve(
                attack_scores[:unconsidered_attack_images],
                bonafide_scores[:unconsidered_bonafide_images],
                drop_intermediate=drop_intermediate,
            )
            bpcer[i] = bpcer_ap(
                fpr, fnr, thresholds, self.attack_potential, percentage=True
            )[0]

        self.systems[label] = {
            "ratio": (0.95 - unconsidered_ratio) * 100,
            "bpcer_ap": bpcer,
        }

    def plot(self, colour_theme="paper"):
        """[summary]

        Parameters
        ----------
        colour_theme : str, optional
            [description], by default "paper"

        Returns
        -------
        [type]
            [description]
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
        ax.set(xlim=self.xlim)  # , ylim=self.ylim)
        ax.set_xticks(self.xticks)
        # ax.set_yticks(self.yticks)
        # ax.set_yticklabels(self.yticklabels)

        data = pd.DataFrame(
            [
                (system, ratio, bpcer)
                for system in self.systems
                for ratio, bpcer in zip(
                    self.systems[system]["ratio"], self.systems[system]["bpcer_ap"]
                )
            ],
            columns=["system", "ratio", "bpcer_ap"],
        )
        sns.lineplot(
            data=data,
            x="ratio",
            y="bpcer_ap",
            hue="system",
            style="system",
            palette=palette,
            ci=None,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))
        ax.legend(handles=handles, labels=labels, loc=0).get_frame().set_edgecolor(
            style["legend.edgecolor"]
        )
        fig.patch.set_alpha(0)

        return fig

    def close(self):
        """[summary]"""
        plt.close()
