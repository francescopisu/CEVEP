"""
This file contains various generic utility methods that do not fall within
data or input-output methods.
"""

import math
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from collections import defaultdict
from itertools import product
import pandas as pd
from yacs.config import CfgNode


def timestamp() -> str:
    return datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')

def show_cross_val_results(cross_val_results: Dict[str, Tuple[float, float, float]],
                           conf: CfgNode):
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
    # results = defaultdict(list)
    print(cross_val_results)
    tuples = list(product(conf.EVAL.ALGO_SHORT_NAMES, conf.EVAL.SET_NAMES))

    header = pd.MultiIndex.from_tuples(tuples, names=["Algorithm", "Set"])
    index = conf.EVAL.METRIC_NAMES
    results = pd.DataFrame(columns=header, index=index)

    for key, (med, lower_ci, upper_ci) in cross_val_results.items():
        algo_name, set_name, metric_name = key.split("_")

        results.loc[metric_name, (algo_name, set_name)] = f"{med} [{lower_ci}-{upper_ci}]"

    return results


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


def get_threshold(conf, which):
    if which == "calcification":
        t = conf.EVAL.THRESHOLD_ONLY_CALCIFICATION
    elif which == "clinical":
        t = conf.EVAL.THRESHOLD_ONLY_CLINICAL
    elif which == "all":
        t = conf.EVAL.THRESHOLD

    return t