import logging

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

logger = logging.getLogger("pypad.metrics.det_curve")


def det_curve(
    attack_scores, bonafide_scores, step_sampling=0.000005, drop_intermediate=False
):
    """_summary_

    Parameters
    ----------
    attack_scores : _type_
        _description_
    bonafide_scores : _type_
        _description_
    step_sampling : float, optional
        _description_, by default 0.000005
    drop_intermediate : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    y_true = np.concatenate(
        (np.zeros(attack_scores.shape), np.ones(bonafide_scores.shape))
    )
    y_score = np.concatenate((attack_scores, bonafide_scores))

    fpr, tpr, thresholds = roc_curve(
        y_true, y_score, drop_intermediate=drop_intermediate
    )
    thresholds[0] = 1.0

    # print(
    #     f"THRESHOLD MIN: {np.min(thresholds)}\nTHRESHOLD MAX: {np.max(thresholds)}\n"
    #     f"FPR MIN: {np.min(fpr)}\nFPR MAX: {np.max(fpr)}\n"
    #     f"TPR MIN: {np.min(tpr)}\nTPR MAX: {np.max(tpr)}\n"
    # )
    thresh_sampling = np.arange(thresholds.min(), thresholds.max(), step_sampling)
    # PATCH FIX
    # thresh_sampling = np.arange(0, 1, step_sampling)
    fpr_interp = interp1d(thresholds, fpr, bounds_error=False, fill_value=(0, 1))
    fpr = fpr_interp(thresh_sampling)
    fnr_interp = interp1d(thresholds, 1 - tpr, bounds_error=False, fill_value=(0, 1))
    fnr = fnr_interp(thresh_sampling)

    return fpr, fnr, thresh_sampling


def det_curve_deprecated(attack_scores, bonafide_scores, drop_intermediate=True):
    """See sidekit.bosaris.detplot

    Computes the False Positive Rate and False Negative Rate across a series of
    operating points.

    Parameters
    ----------
    attack_scores : numpy.ndarray
        Array object of score probabilities for a single PAI species
    bonafide_scores : numpy.ndarray
        Array object of bona fide score probabilities
    drop_intermediate : bool, optional
        Whether reduntant points in the curve should be dropped, by default True

    Returns
    -------
    tuple
        Sorted False Positive Rate points, False Negative Rate points, and thresholds as
        a tuple of three numpy.ndarray objects
    """
    if attack_scores.ndim > 1 and bonafide_scores.ndim > 1:
        raise ValueError(
            "Arrays of attack and bona fide scores must be one dimensional"
        )
    elif attack_scores.ndim > 1:
        raise ValueError("Array of attack scores must be one dimensional")
    elif bonafide_scores.ndim > 1:
        raise ValueError("Array of bona fide scores must be one dimensional")

    npais = attack_scores.shape[0]
    nbf = bonafide_scores.shape[0]
    if npais < 1 and nbf < 1:
        raise ValueError("Arrays of attack and bona fide scores are empty")
    elif npais < 1:
        raise ValueError("Array of attack scores is empty")
    elif nbf < 1:
        raise ValueError("Array of bona fide scores is empty")

    total_presentations = nbf + npais

    fpr = np.zeros((total_presentations + 1))
    fnr = np.zeros((total_presentations + 1))

    scores = np.zeros((total_presentations, 2))
    scores[:npais, 0] = attack_scores
    scores[:npais, 1] = 0
    scores[npais:, 0] = bonafide_scores
    scores[npais:, 1] = 1

    scores = _sort_det(scores)

    sumbf = np.cumsum(scores[..., 1], axis=0, dtype=np.float64)
    sumpais = npais - (np.arange(1, total_presentations + 1) - sumbf)

    fpr[0] = 1
    fnr[0] = 0
    fpr[1:] = sumpais / npais
    fnr[1:] = sumbf / nbf
    thresholds = np.r_[scores[..., 0][0], scores[..., 0]]

    if drop_intermediate:
        idx = _filter_det(fpr, fnr)
        return fpr[idx], fnr[idx], thresholds[idx]

    return fpr, fnr, thresholds


def det_curve_pais(attack_true, attack_scores, bonafide_scores, drop_intermediate=True):
    """Computes the APCER and BPCER across a series of operating points for each
    different PAIS. Interally uses the det_curve() function after filtering attack
    scores by species.

    Parameters
    ----------
    attack_true : numpy.ndarray
        Array object of ground truth labels for multiple PAI species class
    attack_scores : numpy.ndarray
        Array object of score probabilities for multiple PAI species
    bonafide_scores : numpy.ndarray
        Array object of bona fide score probabilities
    drop_intermediate : bool, optional
        Whether reduntant points in the curve should be dropped, by default True

    Returns
    -------
    dict
        A dictionary whose key-value pair are the PAI species labels, and a tuple of
        three numpy.ndarray objects containing the sorted APCER points, BPCER points,
        and thresholds for each particular PAI species
    """
    pais = np.unique(attack_true)
    det_pais = {}
    for species in pais:
        pais_scores = attack_scores[attack_true == species]
        det_pais[species] = det_curve(
            pais_scores, bonafide_scores, drop_intermediate=drop_intermediate
        )
    return det_pais


def eer(apcer_points, bpcer_points, thresholds, percentage=False):
    idx = np.nanargmin(np.abs(apcer_points - bpcer_points))
    return (
        apcer_points[idx] if not percentage else apcer_points[idx] * 100,
        thresholds[idx],
    )


def eer_deprecated(
    apcer_points, bpcer_points, thresholds, interpolation=True, percentage=False
):
    """Computes the Equal Error Rate (EER) of the TOE. Use the det_curve() function to
    calculate the needed parameters.

    Parameters
    ----------
    apcer_points : numpy.ndarray
        Array object of APCER/False Positive rates
    bpcer_points : numpy.ndarray
        Array object of BPCER/False Negative rates
    thresholds : numpy.ndarray
        Array object containing the corresponding thresholds for each point
    interpolation: bool, optional

    percentage : bool, optional
        Returns EER as a percentage, by default False

    Returns
    -------
    tuple
        A tuple of two floats containing the ERR and the corresponding threshold
    """
    if np.array_equal(apcer_points, np.array([1.0, 0.0])):
        return 0.0, thresholds[-1]
    if len(np.unique(apcer_points)) <= 2:
        interpolation = False
    if interpolation:
        eer_ = 0.5 * (
            apcer_points[np.nanargmin(np.abs(apcer_points - bpcer_points))]
            + bpcer_points[np.nanargmin(np.abs(apcer_points - bpcer_points))]
        )
    else:
        eer_ = max(
            apcer_points[np.nanargmin(np.abs(apcer_points - bpcer_points))],
            bpcer_points[np.nanargmin(np.abs(apcer_points - bpcer_points))],
        )
    threshold = thresholds[np.nanargmin(np.abs(apcer_points - bpcer_points))]
    return eer_ if not percentage else eer_ * 100, threshold


def eer_pais(det_pais, percentage=False):
    """Computes the Equal Error Rate (EER) for each different PAIS. Use the
    det_curve_pais() function to calculate the needed parameters.

    Parameters
    ----------
    det_pais : dict
        A dictionary containing the sorted APCER, BPCER, and threshold for each point in
        the DET curve for each particular PAI species. See det_curve_pais()
    percentage : bool, optional
        Returns EER as a percentage, by default False

    Returns
    -------
    dict
        A dictionary whose key-value pairs are the PAI species labels, and the
        corresponding Equal Error Rate for that particular PAI species
    """
    try:
        return {
            species: eer(*det_pais[species], percentage=percentage)
            for species in det_pais
        }
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return None


def _sort_det(x, col=""):
    if x.ndim != 2:
        raise ValueError("Scores array must be a 2D matrix")
    if col == "":
        list(range(1, x.shape[1]))

    ndx = np.arange(x.shape[0])

    # Sort second column ascending
    ind = np.argsort(x[:, 1], kind="mergesort")
    ndx = ndx[ind]

    # Reverse to descending order
    ndx = ndx[::-1]

    # Sort first column ascending
    ind = np.argsort(x[ndx, 0], kind="mergesort")

    ndx = ndx[ind]
    sorted_scores = x[ndx, :]
    return sorted_scores


def _filter_det(fpr, fnr):
    """Computes indices corresponding to non-redutant points in the sequences of the
    False Positive Rate and False Negative Rate. This can be used to drop redundant
    points from the DET curve, which results in a faster plotting of an indentical
    curve.

    See: sidekit.bosaris.detplot.__filter_roc__

    Parameters
    ----------
    fpr : numpy.ndarray
        Array object of False Positive rates
    fnr : numpy.ndarray
        Array object of False Negative rates

    Returns
    -------
    numpy.ndarray
        Array of non-redutant points in the DET curve
    """
    idx = [0]
    last_fpr = 0
    last_fnr = 0

    for i in range(1, fpr.shape[0]):
        if (fpr[i] == last_fpr) | (fnr[i] == last_fnr):
            pass
        else:
            last_fpr = fpr[i - 1]
            last_fnr = fnr[i - 1]
            idx.append(i)

    idx.append(fpr.shape[0] - 1)
    return np.array(idx)
