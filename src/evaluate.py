"""
This file deals with evaluating the generalization performance of a fitted model.
"""
import sys
import optuna
import lightgbm as lgb
from typing import Dict, Callable, List, Tuple, Any
import numpy as np
from functools import partial
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    fbeta_score,
    precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from yacs.config import CfgNode

from src.utils.scoring import (
    compute_conf_matrix_metric,
    compute_metrics,
    bootstrap_median_ci,
    _brier_loss
)
from src._typing import CVScheme, Estimator, ArrayLike
from src.objective import ObjectiveNCV


def compute_cross_val_conf_intervals(cross_val_results: Dict[str, List[float]],
                                     alpha: float = 0.05) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes confidence intervals for the cross-validation results
    with the desired significance level using the percentile method.

    Parameters
    ----------
    cross_val_results: Dict[str, List[float]]
        A dictionary of results of the cross validation procedure.
    alpha: float (default = 0.05)
        The significance level alpha for computing confidence intervals.
        The corresponding confidence level will be (1-alpha)%
        E.g.:
        alpha = 0.05
        confidence level = (1-0.05)% = 95%

    Returns
    -------
    Dict[str, Tuple[float, float, float]]
        A dictionary where keys are of the form SET_METRIC and values are
        tuple of median score, lower and upper bound of confidence interval.
    """
    results_ = dict()
    for k, scores in cross_val_results.items():
        med = np.median(scores).round(3)
        ci_lower = np.percentile(scores, alpha * 100 / 2).round(2)
        ci_upper = np.percentile(scores, 100 - (alpha * 100 / 2)).round(2)

        results_[k] = (med, ci_lower, ci_upper)

    return results_


def cross_val_performance(conf: CfgNode,
                          # base_model: Estimator,
                          params: Dict[str, Any],
                          X: ArrayLike, y: ArrayLike,
                          cv: CVScheme,
                          eval_metrics: Dict[str, Callable],
                          set_names: List[str] = None,
                          instantiate_each_time: bool = False,
                          **kwargs) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute mean and 95% CI for each evaluation metric
    using the percentile method on scores obtained via
    cross-validation (specified by the cv generator `cv`).

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node to access configuration values.
    params: Dict[str, Any]
        A dictionary containing LightGBM's parameters to cross-validate.
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    cv: CVScheme
        A cross-validation scheme to get splits.
    eval_metrics: Dict[str, Callable]
        A dictionary of evaluation metrics.
        Example:
        {
            "auc": roc_auc_score,
            "brier_loss": brier_score_loss
        }
    set_names: List[str]
        A list of names corresponding to sets that predictions
        were computed on.
        Example:
        ["train", "val"]
    instantiate_each_time: bool (default = False)
        Whether the `base_model` needs to be instantiated
        after each split.
    **kwargs:
        Any keyword argument specified will be passed to the constructor
        of base_model.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary where keys are of the form SET_METRIC and values
        are list of floats corresponding to metric values computed at
        each CV split.
    """
    # save results of each metric in each fold
    if set_names is None:
        set_names = ["train", "val"]

    early_stopping_callback = lgb.early_stopping(conf.TUNING.EARLY_STOPPING_ROUNDS)

    results = defaultdict(list)
    for fold, (train_idxs, val_idxs) in enumerate(cv.split(X, y)):
        print(f"{fold + 1}", end=", ")
        # get training and validation parts
        Xt, Xv = X.iloc[train_idxs], X.iloc[val_idxs]
        yt, yv = y.iloc[train_idxs], y.iloc[val_idxs]

        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)
        Xv = scaler.transform(Xv)

        # model.fit(Xt, yt)
        dtrain = lgb.Dataset(Xt, label=yt)
        dval = lgb.Dataset(Xv, label=yv)
        booster = lgb.train(params=params,
                            train_set=dtrain,
                            valid_sets=[dval],
                            callbacks=[early_stopping_callback],
                            feval=_brier_loss)

        # make predictions on both train and validation portions
        train_preds = booster.predict(Xt)
        val_preds = booster.predict(Xv)

        # compute evaluation metrics of interest
        train_scores = compute_metrics(train_preds, yt, eval_metrics)
        val_scores = compute_metrics(val_preds, yv, eval_metrics)

        # save scores
        for s, scores_dict in zip(set_names, [train_scores, val_scores]):
            for metric_name, score in scores_dict.items():
                key = "{}_{}".format(s, metric_name)
                results[key].append(score)

    # compute confidence intervals based on the percentile method.
    results_with_cis = compute_cross_val_conf_intervals(results, alpha=conf.EVAL.ALPHA)
    return results_with_cis


def test_performance(conf: CfgNode,
                     model: Estimator,
                     X_test: ArrayLike, y_test: ArrayLike,
                     eval_metrics: Dict[str, Callable]) -> Dict[str, str]:
    """
    Compute test set performance of the specified model.
    In greater detail, the model is trained on the whole training set
    and used to predict probabilities (if is suppoted by the model)
    on the test data. Then, median and 95% CI are computed for each
    evaluation metric and returned in a dictionary.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node to access configuration values.
    model : Estimator
      A fitted estimator to be tested on test data.
    X_test: ArrayLike of shape (n_obs, n_features)
        Design matrix containing feature values of test data.
    y_test: ArrayLike of shape (n_obs,)
        A vector of test data ground-truth labels.
    eval_metrics: Dict[str, Callable]
        A dictionary of evaluation metrics.
        Example:
        {
            "auc": roc_auc_score,
            "brier_loss": brier_score_loss,
            ...
        }

    Returns
    -------
    Dict[str, str]
        A dictionary of metrics values computed on the test set.
    """
    # Predict on test set
    if isinstance(model, lgb.Booster):
        pred_probas_test = model.predict(X_test)
        pos_class_probas = pred_probas_test
    else:
        pred_probas_test = model.predict_proba(X_test)
        pos_class_probas = pred_probas_test[:, 1]

    # predicted labels with class. threshold of 0.5
    pred_test_labels = np.where(pos_class_probas > conf.EVAL.THRESHOLD, 1, 0)

    out = {}
    for metric_name, metric_fn in eval_metrics.items():
        print("Bootstrapping {}..".format(metric_name))
        if metric_name == 'auc':
            # use probabilities
            preds = pos_class_probas
        else:
            # use labels
            preds = pred_test_labels

        med, conf_int = bootstrap_median_ci(target=y_test,
                                            preds=preds,
                                            metric=metric_fn,
                                            n_boot=conf.EVAL.BOOT_ROUNDS,
                                            seed=conf.MISC.SEED)
        print(f"Metric name: {metric_name}, Median value:{med} ")
        out[metric_name] = "{:.3f} [{:.3f}-{:.3f}]".format(med, conf_int[0], conf_int[1])

    return out


def nested_cv(conf: CfgNode,
              base_params: Dict[str, Any],
              X: ArrayLike,
              y: ArrayLike,
              outer_cv: CVScheme,
              inner_cv: CVScheme) -> Tuple[lgb.Booster, Dict, Dict]:
    """
    Perform nested cross-validation of a binary LightGBM classifier
    with inner model selection through bayesian optimization with the Optuna framework.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node to access configuration values.
    base_params: Dict[str, Any]
        Base LightGBM parameters.
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    outer_cv: CVScheme
        A cross-validation scheme for outer model evaluation.
    inner_cv: CVScheme
        A cross-validation scheme for inner model evaluation.

    Returns
    -------
    Tuple[lgb.Booster, Dict, Dict]
        A tuple consisting of the best LightGBM booster and associated
        parameters after hyperparameter tuning and the results of
        outer the cross-validation evaluation.
    """
    """
    Metric values achieved by LGBM trained with the best parameters setting
    (among the N parameter samples) for each split
    E.g.:
    {
        "split_0": 0.7,
        "split_1": 0.74,
        "split_2": 0.81,
        "split_3": 0.88,
        "split_4": 0.71
    }
    """
    results = dict()

    """
    best params among the N samples for each outer split
    E.g.:
    {
        "split_0": {"l1_lambda": 0.9, ... },
        "split_1": {"l1_lambda": 0.911, ... },
        "split_2": {"l1_lambda": 0.92, ... },
        "split_3": {"l1_lambda": 0.76, ... },
        "split_4": {"l1_lambda": 0.811, ... }
    }
    """
    params = dict()

    hpo_params = {
        "n_trials": conf.TUNING.N_TRIALS,
    }

    early_stopping_callback = lgb.early_stopping(conf.TUNING.EARLY_STOPPING_ROUNDS)

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        Xtrain, ytrain = X.loc[train_idx], y.loc[train_idx]
        Xtest, ytest = X.loc[test_idx], y.loc[test_idx]

        obj = ObjectiveNCV(X=Xtrain, y=ytrain,
                           lgbm_params=base_params,
                           inner_cv=inner_cv,
                           eval_metric=_brier_loss,
                           conf=conf)

        sampler = optuna.samplers.TPESampler(seed=conf.MISC.SEED)
        study = optuna.create_study(sampler=sampler, direction=conf.TUNING.DIRECTION)
        study.optimize(obj, **hpo_params)

        # best set of parameters with related best mean auc among the parameter settings
        # sampled in 500 trials
        params[f"split_{i}"] = study.best_params

        # refit estimator on whole training set with best params
        dtrain = lgb.Dataset(Xtrain, label=ytrain)
        dtest = lgb.Dataset(Xtest, label=ytest)

        booster = lgb.train(params={**study.best_params, **base_params},
                            train_set=dtrain,
                            valid_sets=[dtest],
                            callbacks=[early_stopping_callback],
                            feval=_brier_loss)

        preds = booster.predict(Xtest)
        test_auc = roc_auc_score(ytest, preds)
        results[f"split_{i}"] = test_auc

    # find split associated with best evaluation test auc
    best_split = max(results, key=results.get)

    # retrieve best params associated with this split
    best_params = params[best_split]

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=conf.TUNING.EVAL_SIZE,
                                      stratify=y,
                                      random_state=conf.MISC.SEED)
    dtrain = lgb.Dataset(Xt, label=yt)
    dval = lgb.Dataset(Xv, label=yv)

    booster = lgb.train(params={**best_params, **base_params},
                        train_set=dtrain,
                        valid_sets=[dval],
                        callbacks=[early_stopping_callback],
                        feval=_brier_loss)

    return booster, best_params, results


def get_evaluation_metrics() -> Dict[str, Callable]:
    eval_metrics = {
        "sensitivity": partial(compute_conf_matrix_metric, metric_name="tpr"),
        "specificity": partial(compute_conf_matrix_metric, metric_name="tnr"),
        "fpr": partial(compute_conf_matrix_metric, metric_name="fpr"),
        "fnr": partial(compute_conf_matrix_metric, metric_name="fnr"),
        "precision": precision_score,
        "ppv": partial(compute_conf_matrix_metric, metric_name="ppv"),
        "npv": partial(compute_conf_matrix_metric, metric_name="npv"),
        "f1": partial(fbeta_score, beta=1),
        "f2": partial(fbeta_score, beta=2),
        "brier": brier_score_loss
    }

    return eval_metrics
