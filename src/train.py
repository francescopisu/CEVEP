"""
This module deals with the training and hyperparameter tuning of LightGBM.
"""
import sys
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from yacs.config import CfgNode
import warnings
from typing import Dict, Any, Tuple
from interpret.glassbox import ExplainableBoostingClassifier

from src.utils.io import load_obj
from src._typing import ArrayLike, CVScheme

warnings.filterwarnings("ignore")


def train_model(X: ArrayLike, y: ArrayLike, conf: CfgNode) -> ExplainableBoostingClassifier:
    """
    Train an Explainable Boosting Classifier on specified data.

    Parameters
    ----------
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    conf: CfgNode
        A yacs configuration node to access configuration values.

    Returns
    -------
    A fitted Explainable Boosting Classifier
    """
    ebm = ExplainableBoostingClassifier(random_state=conf.MISC.SEED,
                                              interactions=15,
                                              learning_rate=0.02,
                                              min_samples_leaf=5,
                                              outer_bags=35,
                                              inner_bags=35,
                                              max_bins=128,
                                              max_leaves=3,
                                              n_jobs=10)
    
    ebm.fit(X, y)

    return ebm
    # lr_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=conf.MISC.SEED)
    # model = LogisticRegressionCV(
    #     Cs = 20, cv = lr_cv, penalty = "l2", solver = "liblinear",
    #     n_jobs = 10
    # )
    # model.fit(X, y)
    # return model