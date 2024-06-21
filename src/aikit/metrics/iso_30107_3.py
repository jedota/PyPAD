"""ISO/IEC 30107-3 metrics implementation"""

import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger("aikit.metrics.iso_30107_3")


def apcer_pais(attack_true, attack_scores, threshold=0.5, percentage=False):
    """Computes the Attack Presentation Classification Error Rate (APCER) for multiple
    PAI species at a given operating point. The scores should correspond to the
    probability scores for the bona fide class when the PAD subsystem is evaluated
    against presentation attacks.

    The maximun error rate of all given PAI species corresponds to the worst case
    scenario, which is the one that should be reported.

    Parameters
    ----------
    attack_true : numpy.ndarray
        Array object of ground truth labels for multiple PAI species class
    attack_scores : numpy.ndarray
        Array object of score probabilities for multiple PAI species class
    threshold : float, optional
        Operating point to evaluate, by default 0.5
    percentage : bool, optional
        Returns APCER as a percentage, by default False

    Returns
    -------
    dict
        A dictionary whose key-value pairs are the PAI species labels, and the
        corresponding error rate for that particular PAI species at the given threshold

    Raises
    ------
    ValueError
        An error is raised if the probability scores array is not unidimensional
    """
    if attack_scores.ndim > 1:
        raise ValueError(
            "The probability scores array must be unidimensional, containing only the"
            " classification scores for the bona fide class"
        )
    pais = np.unique(attack_true)
    apcer_ = {}
    for species in pais:
        npais = attack_scores[attack_true == species]
        apcer_[species] = (
            apcer(npais, threshold=threshold)
            if not percentage
            else apcer(npais, threshold=threshold) * 100
        )
    return apcer_


def apcer_max(attack_true, attack_scores, threshold=0.5):
    """Computes the APCER for the PAI species corresponding to the worse case scenario
    at a given operating point.

    Parameters
    ----------
    attack_true : numpy.ndarray
        Array object of ground truth labels for multiple PAI species class
    attack_scores : numpy.ndarray
        Array object of score probabilities for multiple PAI species class
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    float
        APCER for the PAI species corresponding to the worse case scenario at the given
        threshold
    """
    return max(apcer_pais(attack_true, attack_scores, threshold=threshold).values())


def apcer(attack_scores, threshold=0.5):
    """Computes the APCER for a single PAI species at a given operating point.

    Parameters
    ----------
    attack_scores : numpy.ndarray
        Array object of score probabilities for a single PAI species
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    float
        APCER at the given threshold

    Raises
    ------
    ValueError
        An error is raised if the probability scores array is not unidimensional
    """
    if attack_scores.ndim > 1:
        raise ValueError(
            "The probability scores array must be unidimensional, containing only the"
            " classification scores for the PAIS class"
        )
    return np.mean(attack_scores >= threshold)


def apcer_ap(
    apcer_points, bpcer_points, thresholds, attack_potential, percentage=False
):
    ap = np.argmin(np.abs(1 / attack_potential - bpcer_points))
    return (
        apcer_points[ap] if not percentage else apcer_points[ap] * 100,
        thresholds[ap],
    )


def apcer_ap_deprecated(
    apcer_points, bpcer_points, thresholds, attack_potential, percentage=False
):
    """Computes the APCER at the given Attack Potential by using a fixed BPCER value.
    Please note that the AP is a percentage value. Examples:
        - BPCER at AP=20% may be reported as APCER5
        - BPCER at AP=10% may be reported as APCER10
        - BPCER at AP=5% may be reported as APCER20
        - BPCER at AP=1% may be reported as APCER100
        - BPCER at AP=0.1% may be reported as APCER1000
        - BPCER at AP=0.01% may be reported as APCER10000

    To compute the required APCER, BPCER and threshold points, please see
    aikit.metrics.det_curve.det_curve().

    Parameters
    ----------
    apcer_points : numpy.ndarray
        Array object of APCER points
    bpcer_points : numpy.ndarray
        Array object of BPCER points
    thresholds : numpy.ndarray
        Array object containing the corresponding threshold points
    attack_potential : int
        Attack Potential
    percentage : bool, optional
        Returns BPCER as a percentage, by default False

    Returns
    -------
    tuple
        BPCER and the corresponding threshold at the given AP

    Raises
    ------
    ValueError
        An error is raised for invalid AP values
    """
    if attack_potential < 1:
        raise ValueError("Attack Potential variable must greater than 0")

    try:
        thres_interpolation = interp1d(bpcer_points, thresholds)
    except ValueError:
        print(
            "Data points have too little variability to correctly interpolate a valid"
            " threshold point."
        )
        return None, None
    try:
        apcer_interpolation = interp1d(bpcer_points, apcer_points)
    except ValueError:
        print(
            "Data points have too little variability to correctly interpolate a valid"
            " APCER point."
        )
        return None, None

    fixed_bpcer = 1 / attack_potential
    apcer_ = (
        apcer_interpolation(fixed_bpcer)
        if not percentage
        else apcer_interpolation(fixed_bpcer) * 100
    )

    return np.float64(apcer_), np.float64(thres_interpolation(fixed_bpcer))


def bpcer(bonafide_scores, threshold=0.5):
    """Computes the Bona Fide Presentation Classification Error Rate (BPCER) at a given
    operating point.

    Please note that the BPCER is the same as the FRR when Attack Potential is not taken
    into account.

    Parameters
    ----------
    bonafide_scores : numpy.ndarray
        Array object of bona fide score probabilities
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    np.float
        BPCER at the given threshold

    Raises
    ------
    ValueError
        An error is raised if the probability scores array is not unidimensional
    """
    if bonafide_scores.ndim > 1:
        raise ValueError(
            "The probability scores array must be unidimensional, containing only the"
            " classification scores for the bona fide class"
        )
    return np.mean(bonafide_scores < threshold)


def bpcer_ap(
    apcer_points, bpcer_points, thresholds, attack_potential, percentage=False
):
    ap = np.argmin(np.abs(1 / attack_potential - apcer_points))
    return (
        bpcer_points[ap] if not percentage else bpcer_points[ap] * 100,
        thresholds[ap],
    )


def bpcer_ap_deprecated(
    apcer_points, bpcer_points, thresholds, attack_potential, percentage=False
):
    """Computes the BPCER at the given Attack Potential by using a fixed APCER value.
    Please note that the AP is a percentage value. Examples:
        - APCER at AP=20% may be reported as BPCER5
        - APCER at AP=10% may be reported as BPCER10
        - APCER at AP=5% may be reported as BPCER20
        - APCER at AP=1% may be reported as BPCER100
        - APCER at AP=0.1% may be reported as BPCER1000
        - APCER at AP=0.01% may be reported as BPCER10000

    To compute the required APCER, BPCER and threshold points, please see
    aikit.metrics.det_curve.det_curve().

    Parameters
    ----------
    apcer_points : numpy.ndarray
        Array object of APCER points
    bpcer_points : numpy.ndarray
        Array object of BPCER points
    thresholds : numpy.ndarray
        Array object containing the corresponding threshold points
    attack_potential : int
        Attack Potential
    percentage : bool, optional
        Returns BPCER as a percentage, by default False

    Returns
    -------
    tuple
        BPCER and the corresponding threshold at the given AP

    Raises
    ------
    ValueError
        An error is raised for invalid AP values
    """
    if attack_potential < 1:
        raise ValueError("Attack Potential variable must greater than 0")

    try:
        thres_interpolation = interp1d(
            apcer_points, thresholds, fill_value="extrapolate"
        )
    except ValueError:
        print(
            "Data points have too little variability to correctly interpolate a valid"
            " threshold point."
        )
        return None, None
    try:
        bpcer_interpolation = interp1d(
            apcer_points, bpcer_points, fill_value="extrapolate"
        )
    except ValueError:
        print(
            "Data points have too little variability to correctly interpolate a valid"
            " BPCER point."
        )
        return None, None

    fixed_apcer = 1 / attack_potential
    bpcer_ = (
        bpcer_interpolation(fixed_apcer)
        if not percentage
        else bpcer_interpolation(fixed_apcer) * 100
    )

    return np.float64(bpcer_), np.float64(thres_interpolation(fixed_apcer))


def iapar(attack_scores, threshold=0.5):
    """Alias for apcer(). Computes the Impostor Attack Presentation Accept Rate (IAPAR)
    at a given operating point.

    Parameters
    ----------
    attack_scores : numpy.ndarray
        Array object of score probabilities for a single PAI species
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    float
        IAPAR at the given threshold
    """
    return apcer(attack_scores, threshold=threshold)


def frr(bonafide_scores, threshold=0.5):
    """Alias for bpcer(). Computes the False Rejection Rate (FRR) at a given operating
    point.

    Parameters
    ----------
    bonafide_scores : numpy.ndarray
        Array object of bona fide score probabilities
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    np.float
        False Rejection Rate at the given threshold
    """
    return bpcer(bonafide_scores, threshold=threshold)


def riapar(
    attack_scores, bonafide_scores, attack_threshold=0.5, bonafide_threshold=0.5
):
    """Computes the Relative Impostor Attack Presentation Accept Rate (RIAPAR) at a
    given operating point.

    Parameters
    ----------
    attack_scores : numpy.ndarray
        Array object of score probabilities for a single PAI species
    bonafide_scores : numpy.ndarray
        Array object of bona fide score probabilities
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    np.float
        RIAPAR at the given threshold
    """
    iapar_ = iapar(attack_scores, threshold=attack_threshold)
    frr_ = frr(bonafide_scores, threshold=bonafide_threshold)
    return 1 + (iapar_ - (1 - frr_))


def acer(attack_true, attack_scores, bonafide_scores, threshold=0.5):
    """Computes the Average Classification Error Rate (ACER), also known as Half-Total
    Error Rate (HTER), at a given operating point.

    THIS METRIC IS NOT CONFORMANT WITH THE ISO/IEC 30107-3 STANDARD!

    Parameters
    ----------
    attack_true : numpy.ndarray
        Array object of ground truth labels for multiple PAI species class
    attack_scores : numpy.ndarray
        Array object of score probabilities for multiple PAI species class
    bonafide_scores : numpy.ndarray
        Array object of bona fide score probabilities
    threshold : float, optional
        Operating point to evaluate, by default 0.5

    Returns
    -------
    np.float
        ACER at the given threshold

    Raises
    ------
    ValueError
        An error is raised if the probability scores array is not unidimensional
    """
    if attack_scores.ndim > 1 or bonafide_scores.ndim > 1:
        raise ValueError(
            "The probability scores array must be unidimensional, containing only the"
            " classification scores for the bona fide class"
        )
    apcer_ = apcer_max(attack_true, attack_scores, threshold=threshold)
    bpcer_ = bpcer(bonafide_scores, threshold=threshold)
    return 0.5 * (apcer_ + bpcer_)
