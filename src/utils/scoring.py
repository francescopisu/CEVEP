"""
This file defines methods for computing evaluation metrics of interest.
"""
import numpy as np
from typing import Dict, Callable, Tuple
from sklearn.metrics import confusion_matrix, brier_score_loss
import scikits.bootstrap as boot

from src._typing import ConfMatrixQuant, Seed, ArrayLike


def compute_conf_matrix(target_labels: ArrayLike,
                        pred_labels: ArrayLike) -> ConfMatrixQuant:
    """
    Compute confusion matrix-related quantities.

    Parameters
    ----------
    target_labels: ArrayLike of shape (n_obs,)
        Array of ground-truth labels for positive class.
    pred_labels: ArrayLike of shape (n_obs,)
        Array of predicted labels for positive class.

    Returns
    -------
    ConfMatrixQuant
        A tuple of four floats corresponding to true negatives, false positives,
        false negatives and true positives.
    """
    return confusion_matrix(target_labels, pred_labels).ravel()


def compute_conf_matrix_metric(target_labels: ArrayLike,
                               pred_labels: ArrayLike,
                               metric_name: str = "fpr") -> float:
    """
    Compute the requested confusion matrix-related metric.

    Parameters
    ----------
    conf_matrix_quantities: ConfMatrixQuant
        A tuple of four floats corresponding to true negatives, false positives,
        false negatives and true positives.
    metric_name: str (default = "tpr")
        The confusion matrix-related metric to be computed.
        Possible options are:
            - "tpr": true positive rate;
            - "fpr": false positive rate;
            - "fnr": false negative rate;
            - "tnr": true negative rate;
            - "ppv": positive predictive value;
            - "npv": negative predictive value;

    Returns
    -------
    float
        the value of the requested metric.
    """
    tn, fp, fn, tp = compute_conf_matrix(target_labels=target_labels,
                                         pred_labels=pred_labels)
    metric_value = None

    if metric_name == 'fpr':
        metric_value = fp / (fp + tn)
    elif metric_name == 'tpr':
        metric_value = tp / (fn + tp)
    elif metric_name == 'tnr':
        metric_value = tn / (fp + tn)
    elif metric_name == 'fnr':
        metric_value = fn / (fn + tp)
    elif metric_name == 'ppv':
        metric_value = tp / (tp + fp)
    elif metric_name == 'npv':
        metric_value = tn / (tn + fn)

    return metric_value


def compute_metrics(preds: ArrayLike,
                    target: ArrayLike,
                    metrics: Dict[str, Callable],
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute the specified metrics using either predicted
    probabilities or labels derived applying the specified
    classification threshold.

    Parameters
    ----------
    preds : ArrayLike of shape (n_obs, n_classes)
      Array of predicted probabilities of each class.
    target : ArrayLike of shape (n_obs,)
      Array of ground-truth labels.
    metrics : Dict[str, Callable]
      a dictionary specifying the metrics that should
      be computed. Keys are metric names and values
      are corresponding callables that will be used
      to compute such metrics.
    threshold : float (default = 0.5)
      the classification threshold used to derived labels
      from predicted probabilities.

    Returns
    -------
    Dict[str, float]
      a dictionary with metric names and corresponding values.
    """
    # apply threshold to predicted probabilities of +ive class
    if preds.ndim == 2:
        pos_class_proba = preds[:, 1]
    else:
        pos_class_proba = preds

    labels = np.where(pos_class_proba >= threshold, 1, 0)

    scores = {}
    for metric_name, metric_fn in metrics.items():
        if metric_fn:
            if metric_name in ['ROCAUC', 'PRAUC', 'Brier']:
                # use probabilities of +ive class in this case
                scores[metric_name] = metric_fn(target, pos_class_proba)
            else:
                scores[metric_name] = metric_fn(target, labels)

    return scores


def bootstrap_median_ci(target: ArrayLike,
                        preds: ArrayLike,
                        metric: Callable,
                        n_boot: int = 5000,
                        seed: Seed = 1234) -> Tuple[float, Tuple[float, float]]:
    """
    Computes median and 95% CI of a metric via BCa method
    using the non-parametric bootstrap.

    Parameters
    ----------
    target: ArrayLike of shape (n_obs,)
        Array of ground-truths.
    preds: ArrayLike of shape (n_obs,)
        Array of either predicted probabilities or labels.
    metric: Callable
        The metric to be bootstrapped.
    n_boot: int (default = 5000)
        The number of bootstrap rounds to compute.
    seed: Seed
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        Median statistics and 95% confidence interval using the BCa approach
    """
    conf_int, dist = boot.ci(data=(target, preds),
                             statfunction=metric,
                             multi="paired",
                             n_samples=n_boot,
                             method="bca",
                             seed=seed,
                             return_dist=True)

    median = np.median(dist).item()
    return median, (conf_int[0], conf_int[1])
