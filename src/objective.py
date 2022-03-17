"""
This module defines an optimization objective for the Optuna optimization procedure.
"""
from optuna.integration.lightgbm import LightGBMPruningCallback
import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Callable
from yacs.config import CfgNode

from src._typing import ArrayLike, CVScheme


class ObjectiveNCV:
    """
    Custom objective for LightGBM optimization through the Optuna framework.

    Arguments
    ---------
    X: ArrayLike of shape (n_obs, n_features)
        Design matrix containing feature vectors.
    y: ArrayLike of shape (n_obs,)
        Ground-truth values.
    lgbm_params: Dict[str, Any]
        Base LightGBM parameters
    eval_metric: Callable
        A custom evaluation metric used to assess performance
        on the validation set.
    conf: CfgNode
        A yacs configuration node.
    """

    def __init__(self, X: ArrayLike, y: ArrayLike,
                 lgbm_params: Dict[str, Any],
                 inner_cv: CVScheme,
                 eval_metric: Callable,
                 conf: CfgNode):
        self.dtrain = lgb.Dataset(X, label=y)
        self.lgbm_params = lgbm_params
        self.cv = inner_cv
        self.eval_metric = eval_metric
        self.conf = conf

    def _get_space(self, trial):
        """
        Defines the hyperparameter space to sample parameters from
        in a given trial.
        """
        space = {
            "reg_alpha": trial.suggest_float("lambda_l1", 0.1, 10.0),
            "reg_lambda": trial.suggest_float("lambda_l2", 0.1, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 15),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "n_estimators": trial.suggest_int("n_estimators", 30, 120),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
        }

        return space

    def __call__(self, trial):
        space = self._get_space(trial)
        all_params = {**self.lgbm_params, **space}

        # Add a callback for pruning.
        pruning_callback = LightGBMPruningCallback(trial, self.conf.TUNING.METRIC)

        res = lgb.cv(params=all_params,
                     train_set=self.dtrain,
                     folds=self.cv,
                     callbacks=[pruning_callback,
                                lgb.early_stopping(self.conf.TUNING.EARLY_STOPPING_ROUNDS)],
                     feval=self.eval_metric)

        # len(res['auc-mean']) = length is equal to the number of boosted trees created
        # during the training procedure. e.g.: 91 means that 91 trees were built, and each n-th value
        # in res['auc-mean'] is the mean cross-validation AUC obtained by the system with n trees so far.
        # as long as the mean cv auc keeps is increasing the system keeps building trees.
        # if the system doesn't register increase in mean CV  auc after EARLY_STOPPING_ROUNDS
        # the training procedure is halted. In that case len(res['auc-mean']) will be equal to the number of
        # trees created to that point and the last value corresponds to the mean CV AUC for the last
        # boosted tree, i.e. the highest metric value reached.
        return np.mean(res[f'{self.conf.TUNING.METRIC}-mean'])
