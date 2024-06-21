import logging

import numpy as np

from aikit.metrics.iso_30107_3 import apcer_pais

logger = logging.getLogger("aikit.metrics.scores")


def split_scores(y_true, y_score, bonafide_label=1):
    """Splits a numpy array of scores into two arrays, one containing the
    impostor/attack presentation scores, and the other the bona fide class scores.
    Additional numpy arrays, containing the ground truth labels for the impostor/attack
    presentation scores and the bona fide scores, are also returned. Please note that
    impostor/attack presentation scores will contain the values for all impostor/PAI
    species; these scores can be separated by utilizing split_attack_scores() using the
    results from this function. To instead isolate the worst case scenario, utilize the
    max_error_pais_scores() function.

    Parameters
    ----------
    y_true : numpy.ndarray
        Array object of ground truth labels
    y_score : numpy.ndarray
        Array object of score probabilities for all clases
    bonafide_label : int, optional
        Label of the bona fide class, by default 1

    Returns
    -------
    tuple
        Probability scores and ground truth labels arrays for the impostor/PAI species,
        and the bona fide class. This is retured as a tuple of four numpy.ndarray
        objects
    """
    attack_scores = y_score[y_true != bonafide_label]
    bonafide_scores = y_score[y_true == bonafide_label]
    attack_true = y_true[y_true != bonafide_label]
    bonafide_true = y_true[y_true == bonafide_label]
    return attack_scores, bonafide_scores, attack_true, bonafide_true


def split_attack_scores(attack_true, attack_scores):
    """Returns the scores for all the different PAI species separately. The scores are
    contained in a dictionary where each key represents the label of the PAI species.

    Parameters
    ----------
    attack_true : numpy.ndarray
        Array object of ground truth labels for all PAI species class
    attack_scores : numpy.ndarray
        Array object of score probabilities for all PAI species class

    Returns
    -------
    dict
        A dictionary whose key-pair values are the PAI species labels, and the
        numpy.ndarray containing the scores for that particular PAI species
    """
    pais = np.unique(attack_true)
    return {species: attack_scores[attack_true == species] for species in pais}


def max_error_pais_scores(attack_true, attack_scores, threshold=0.5):
    """Returns the scores and class label of the PAI species corresponding to the worst
    case scenario.

    Parameters
    ----------
    attack_true : numpy.ndarray
        Array object of ground truth labels for all PAI species class
    attack_scores : numpy.ndarray
        Array object of score probabilities for all PAI species class

    Returns
    -------
    tuple
        Score probabilities array and (single) ground truth label for the PAI species
        corresponding to the worst case scenario. This is retured as a tuple of one
        numpy.ndarray object and one integer value
    """
    apcer = apcer_pais(attack_true, attack_scores, threshold=threshold)
    max_pais = max(apcer, key=apcer.get)
    # max_pais = np.argmax([*apcer.values()])
    return attack_scores[attack_true == max_pais], int(max_pais)


def pad_scores(*class_scores, manual_padding=None):
    """Examples:
        - pad_scores(bonafide_scores, attack_scores)
        - pad_scores(bonafide_scores, attack_scores, impostor_scores)

    Returns
    -------
    [type]
        [description]
    """
    if not manual_padding:
        max_ = max([len(scores) for scores in class_scores])
    else:
        max_ = manual_padding
    return [
        np.pad(
            scores, (0, max_ - scores.shape[0]), mode="constant", constant_values=None
        )
        for scores in class_scores
    ]
