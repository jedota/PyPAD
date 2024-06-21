import logging

logger = logging.getLogger("aikit.graphics.aesthetics")


def set_colour_theme(theme="light"):
    """Colour themes definitions for plotting functions.

    Parameters
    ----------
    theme : str, optional
        Colour mode style to retrieve, by default "light"

    Returns
    -------
    tuple
        A dict with the colour theme, and a str or seaborn.palettes._ColorPalette for
        the colour pallete
    """
    themes = {
        "light": {
            "axes.facecolor": "#EAEAF2",
            "axes.edgecolor": "white",
            "axes.labelcolor": ".15",
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "grid.color": "white",
            "grid.linestyle": "-",
            "legend.facecolor": ".8",
            "legend.edgecolor": ".8",
            "text.color": ".15",
            "xtick.color": ".15",
            "ytick.color": ".15",
            # 'font.family': "serif",
            # 'font.sans-serif': "Helvetica",
            "savefig.facecolor": "#EAEAF2",
            "savefig.edgecolor": "#white",
            "savefig.transparent": True,
            "eer.color": ".66",
        },
        "dark": {
            "axes.facecolor": "#171718",
            "axes.edgecolor": ".3",
            "axes.labelcolor": ".85",
            "figure.facecolor": "black",
            "figure.edgecolor": ".3",
            "grid.color": ".3",
            "grid.linestyle": "-",
            "legend.facecolor": ".7",
            "legend.edgecolor": ".7",
            "text.color": ".85",
            "xtick.color": ".85",
            "ytick.color": ".85",
            "savefig.facecolor": "#171718",
            "savefig.edgecolor": ".3",
            "savefig.transparent": True,
            "eer.color": ".33",
        },
        "paper": {
            "axes.facecolor": "white",
            "axes.edgecolor": ".15",
            "axes.labelcolor": "black",
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "grid.color": ".8",
            "grid.linestyle": "--",
            "legend.facecolor": ".8",
            "legend.edgecolor": ".8",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "#white",
            "savefig.transparent": True,
            "eer.color": ".66",
        },
    }
    palettes = {
        "light": "viridis",
        "dark": "magma",
        "paper": "Dark2",  # "Set1" "Set2" "Dark2" "gist_ncar" "tab10" "bright"
    }

    if theme not in themes:
        logger.warning(
            f"Colour mode '{theme}' not supported. Using the default light colour mode"
        )
        theme = "light"

    return themes[theme], palettes[theme]
