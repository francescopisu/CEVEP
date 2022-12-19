"""
This file deals with evaluating the generalization performance of a fitted model.
"""
import sys
from tqdm import tqdm
import pandas as pd
import optuna
from optuna.integration import OptunaSearchCV
# import lightgbm as lgb
from typing import Dict, Callable, List, Tuple, Any
import numpy as np
from functools import partial
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve
from interpret.glassbox import ExplainableBoostingClassifier
from collections import defaultdict
from yacs.config import CfgNode

from src.utils.scoring import (
    compute_conf_matrix_metric,
    compute_metrics,
    bootstrap_median_ci,
    _brier_loss
)
from src._typing import CVScheme, Estimator, ArrayLike


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


def cross_validate(X: ArrayLike,
                   y: ArrayLike,
                   cv: CVScheme,
                   conf: CfgNode) -> List[Tuple[Estimator, Dict, Dict]]:
    """
    Estimate generalization performance through cross-validation.

    Parameters
    ----------
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    cv: CVScheme
        A cross-validation scheme for model evaluation.
    conf: CfgNode
        A yacs configuration node to access configuration values.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple consisting of average values of evaluation metrics and
        predicted probabilities for each subject (out-of-fold).
    """
    pd.options.mode.chained_assignment = None
    results = defaultdict(list)
    preds = dict()

    eval_metrics = get_evaluation_metrics()


    preds = dict()
    gts = []
    gts_train = []
    probas = []
    probas_train = []
    errors = []
    correct = []


    # lr_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=conf.MISC.SEED)
    # model = LogisticRegressionCV(
    #     Cs = 20, cv = lr_cv, penalty = "l2", solver = "liblinear",
    #     n_jobs = 10
    # )

    for i, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y)), total=cv.get_n_splits(X, y)):
        Xtrain, ytrain = X.loc[train_idx, :], y[train_idx]
        Xtest, ytest = X.loc[test_idx, :], y[test_idx]

        # instantiate EBM
        model = ExplainableBoostingClassifier(random_state=conf.MISC.SEED,
                                              interactions=15,
                                              learning_rate=0.01,
                                              min_samples_leaf=5,
                                              outer_bags=35,
                                              inner_bags=35,
                                              max_bins=128,
                                              max_interaction_bins=32,
                                              max_leaves=3,
                                              max_rounds=100,
                                              n_jobs=10)

        # scale numeric variables
        # numeric_cols = Xtrain.select_dtypes(include=np.float64).columns.tolist()
        # scaler = StandardScaler()
        # Xtrain.loc[:, numeric_cols] = scaler.fit_transform(Xtrain.loc[:, numeric_cols])
        # Xtest.loc[:, numeric_cols] = scaler.transform(Xtest.loc[:, numeric_cols])

        # run hyperparameter tuning
        model.fit(Xtrain, ytrain)

        # predict on both sets
        train_preds = model.predict_proba(Xtrain)
        test_preds = model.predict_proba(Xtest)

        # pool ground-truths and predicted probabilities
        gts.append(ytest)
        gts_train.append(ytrain)
        probas.append(test_preds)
        probas_train.append(train_preds)

        # compute evaluation metrics
        fpr, tpr, thresholds = roc_curve(ytrain, train_preds[:, 1])
        idx = np.argmax(tpr - fpr)
        thresh = thresholds[idx]

        # use threshold to classify validation instances
        labels_val = np.where(test_preds[:, 1] >= thresh, 1, 0)

        # find out errors
        error_idxs = Xtest[(ytest != labels_val)].index
        errors.append(error_idxs)
        correct_idxs = Xtest[(ytest == labels_val)].index
        correct.append(correct_idxs)

        train_scores = compute_metrics(train_preds, ytrain, eval_metrics, threshold=thresh)
        test_scores = compute_metrics(test_preds, ytest, eval_metrics, threshold=thresh)

        # save scores
        for s, scores_dict in zip(conf.EVAL.SET_NAMES, [train_scores, test_scores]):
            for metric_name, score in scores_dict.items():
                key = "{}_{}_{}".format("EBM", s, metric_name)
                results[key].append(score)

    # concatenate ground-truths and predicted probabilities
    preds["EBM"] = {
        "gt_conc": np.concatenate(gts),
        "gt_conc_train": np.concatenate(gts_train),
        "probas_conc": np.concatenate(probas),
        "probas_conc_train": np.concatenate(probas_train),
        "gt": gts,
        "gt_train": gts_train,
        "probas": probas,
        "probas_train": probas_train,
        "errors": errors,
        "correct": correct,        
    }

    # compute confidence intervals based on the percentile method.
    results_with_cis = compute_cross_val_conf_intervals(results, alpha=conf.EVAL.ALPHA)

    return results_with_cis, preds


def test_performance(conf: CfgNode,
                     model: Estimator,
                     X_test: ArrayLike, y_test: ArrayLike,
                     eval_metrics: Dict[str, Callable],
                     threshold: float) -> Dict[str, str]:
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
    threshold: float
        the threshold to use to dichotomize predicted probabilities

    Returns
    -------
    Dict[str, str]
        A dictionary of metrics values computed on the test set.
    """
    pred_probas_test = model.predict_proba(X_test)
    pos_class_probas = pred_probas_test[:, 1]

    # predicted labels with class. threshold of 0.5
    pred_test_labels = np.where(pos_class_probas >= threshold, 1, 0)

    out = {}
    for metric_name, metric_fn in eval_metrics.items():
        print("Bootstrapping {}..".format(metric_name))
        if metric_name in ['ROCAUC', 'PRAUC', 'Brier']:
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



def get_evaluation_metrics() -> Dict[str, Callable]:
    eval_metrics = {
        "Sensitivity": partial(compute_conf_matrix_metric, metric_name="tpr"),
        "Specificity": partial(compute_conf_matrix_metric, metric_name="tnr"),
        "FPR": partial(compute_conf_matrix_metric, metric_name="fpr"),
        "FNR": partial(compute_conf_matrix_metric, metric_name="fnr"),
        "Precision": precision_score,
        "PPV": partial(compute_conf_matrix_metric, metric_name="ppv"),
        "NPV": partial(compute_conf_matrix_metric, metric_name="npv"),
        "F1": partial(fbeta_score, beta=1),
        "F2": partial(fbeta_score, beta=2),
        "ROCAUC": roc_auc_score,
        "PRAUC": average_precision_score,
        "Brier": brier_score_loss
    }

    return eval_metrics
