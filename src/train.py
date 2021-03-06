"""
This module deals with the training and hyperparameter tuning of LightGBM.
"""
import optuna
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from yacs.config import CfgNode
import warnings
from typing import Dict, Any, Tuple

from src.utils.io import load_obj
from src.utils.scoring import _brier_loss
from src.objective import ObjectiveNCV
from src._typing import ArrayLike, CVScheme

warnings.filterwarnings("ignore")


def _get_base_lgbm_params(conf: CfgNode) -> Dict[str, Any]:
    """
    Returns base LightGBM parameters.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node.

    Returns
    -------
    Dict[str, Any]
        A dictionary of LightGBM fixed parameters.
    """
    base_params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': "None",
        'num_threads': 12,
        'random_state': conf.MISC.SEED,
        'seed': conf.MISC.SEED,
        'force_row_wise': True,
        'deterministic': True,
        'verbose': -1,
        'feature_pre_filter': False
    }

    return base_params


def _get_lgbm_params(conf: CfgNode) -> Dict[str, Any]:
    """
    Merges base LightGBM parameters with additional
    parameters found via hyperparameter tuning and
    stored in the pickle object specified at configuration
    level.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node.

    Returns
    -------
    Dict[str, Any]
        A dictionary of LightGBM parameters.
    """
    base_params = _get_base_lgbm_params(conf)

    # load best model's parameters
    best_params = load_obj(conf.OUTPUT.BEST_PARAMS)

    return {**base_params, **best_params}


def fit_lgbm(conf: CfgNode,
             params: Dict[str, Any],
             X: ArrayLike,
             y: ArrayLike) -> lgb.Booster:
    """
    Trains a LightGBM boosting machine on the specified data.
    A randomized split is performed to get validation data
    for early stopping purposes.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node.
    params: Dict[str, Any]
        A dictionary of LightGBM parameters.
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.

    Returns
    -------
    lgb.Booster
        The trained LightGBM booster.

    """
    Xt, Xv, yt, yv = train_test_split(X, y,
                                      test_size=conf.DATA.TEST_SIZE,
                                      stratify=y,
                                      random_state=conf.MISC.SEED)
    dtrain = lgb.Dataset(Xt, label=yt)
    dval = lgb.Dataset(Xv, label=yv)

    model = lgb.train(params,
                      dtrain,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(conf.TUNING.EARLY_STOPPING_ROUNDS)],
                      feval=_brier_loss
                      )

    return model


def tune_lgbm(conf: CfgNode,
              base_params: Dict[str, Any],
              X: ArrayLike,
              y: ArrayLike,
              cv: CVScheme) -> Tuple[lgb.Booster, Dict]:
    """
    Perform hyperparameter tuning of LightGBM boosting machine.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node.
    base_params: Dict[str, Any]
        A dictionary of LightGBM base parameters.
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    cv: CVScheme
        A cross-validation scheme used to internally
        evaluate parameter settings.

    Returns
    -------
    Tuple[lgb.Booster, Dict]
        A tuple of a tuned LightGBM booster with associated parameters.
    """
    # base_params = _get_base_lgbm_params(conf)

    hpo_params = {
        "n_trials": conf.TUNING.N_TRIALS,
    }

    obj = ObjectiveNCV(X=X, y=y,
                       lgbm_params=base_params,
                       eval_metric=_brier_loss,
                       conf=conf,
                       inner_cv=cv)
    sampler = optuna.samplers.TPESampler(seed=conf.MISC.SEED)
    study = optuna.create_study(sampler=sampler, direction=conf.TUNING.DIRECTION)
    study.optimize(obj, **hpo_params)

    # after the optimization save best parameters, fit LGB model
    best_params = study.best_params

    # re-fit the model with params on the whole training set
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=conf.TUNING.EVAL_SIZE,
                                      stratify=y,
                                      random_state=conf.MISC.SEED)
    dtrain = lgb.Dataset(Xt, label=yt)
    dval = lgb.Dataset(Xv, label=yv)

    best_model = lgb.train(
        params={**best_params, **base_params},
        train_set=dtrain,
        valid_sets=[dval],
        feval=_brier_loss,
        callbacks=[lgb.early_stopping(conf.TUNING.EARLY_STOPPING_ROUNDS)],
    )

    return best_model, best_params
