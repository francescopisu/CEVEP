"""
This file contains various generic utility methods that do not fall within
data or input-output methods.
"""

import math
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from collections import defaultdict
import pandas as pd


def timestamp() -> str:
    return datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')


def show_cross_val_results(cross_val_results: Dict[str, Tuple[float, float, float]]):
    """
    Iterates over the dictionary of cross-validation results and generates
    a good-looking dataframe of median - [CI lower, CI upper] for each
    metric and set.

    Parameters
    ----------
    cross_val_results: Dict[str, Tuple[float, float, float]]
        Dictionary of results from cross-validation procedure.

    Returns
    -------
    pd.DataFrame
        A dataframe showing cross-validation results with median - [CI lower, CI upper]
        for each metric and set.
    """
    results = defaultdict(list)
    metric_names = []

    for key, (med, lower_ci, upper_ci) in cross_val_results.items():
        set_name, metric_name = key.split("_")
        metric_names.append(metric_name)
        results[set_name].append(f"{med} - [{lower_ci}-{upper_ci}]")

    print(metric_names)
    output = pd.DataFrame.from_dict(results)
    output.index = metric_names[:math.floor(len(metric_names) / 2)]
    return output


def add_extension(estimator_filename: str) -> str:
    """
    Adds the .pkl extension to the estimator's filename (if needed).

    Parameters
    ----------
    estimator_filename: str
        Filename of the estimator. It may or may not end with the
        extension .pkl

    Returns
    -------
    str
        The estimator's filename ending with .pkl
    """
    parts = estimator_filename.split(".")
    model_name = parts[0]
    extension = parts[1] if len(parts) > 1 else "pkl"
    estimator_filename = f"{model_name}.{extension}"

    return estimator_filename


def predict_proba(data, model_=None):
    a = np.zeros((data.shape[0],))
    probas = model_.predict(data)
    return np.c_[a, probas]
