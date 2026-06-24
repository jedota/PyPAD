
from .aesthetics import set_colour_theme
from .biometric_performance import (
    performance_evaluation,
    performance_evaluation_morphing,
)
from .confusion_matrix import plot_confusion_matrix, plot_system_confusion_matrix
from .det_plot import DETPlot
from .erc_plot import ERCPlot
from .history import plot_history
from .roc_curve import ROCCurve

__all__ = [
    # Aesthetics & Themes
    "set_colour_theme",
    
    # Performance Evaluation
    "performance_evaluation",
    "performance_evaluation_morphing",
    
    # Curve & Metric Plots
    "DETPlot",
    "ERCPlot",
    "ROCCurve",
    
    # Matrix & History Plots
    "plot_confusion_matrix",
    "plot_system_confusion_matrix",
    "plot_history",
]