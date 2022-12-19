"""
This file implements a CLI to start the training, tuning and testing of the model.
"""
from typing import Dict, Optional, List
import os
import sys
import time
import numpy as np
import pandas as pd
import click
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from src.utils.io import load_obj, save_obj
from src.utils.misc import show_cross_val_results, add_extension
from src.utils.data import load_preprocess
from config.defaults import get_defaults
from src.evaluate import (
    test_performance,
    cross_validate,
    get_evaluation_metrics,
)
from src.train import train_model
from src.utils.misc import get_threshold


def bootstrap(new_options: Optional[List] = None,
              mode: str = "train") -> Dict:
    """
    This function is responsible for the bootstrap phase prior to
    training or testing the model.
    It is responsible for:
        - loading the default configuration values
        - updating defaults by merging CLI argumentsc
        - loading either train or test dataset
        - instantiating or loading a model

    Parameters
    ----------
    new_options: new options coming from CLI arguments. They will be merged
        with defaults.
    mode: str (default = "train")
        Modality of execution. Options are train, test, cv and tune.
    Returns
    -------
    Dict
        A dictionary containing preprocessed data, a model and configuration data.
    """
    defaults = get_defaults()
    if new_options:
        defaults.merge_from_list(new_options)
    defaults.freeze()

    # load datasets
    which = 'train'  # default
    if mode in ['train', 'cv']:
        which = 'train'
    elif mode == 'internal_test':
        which = 'test'
    elif mode == 'external_test':  # mode == 'external_test'
        which = 'external'

    X, y = load_preprocess(defaults, which=which)

    # a. Get fixed parameters and search spaces for nested cross-validation
    # b. Get fixed parameters and search space for a specific algorithm to be tuned
    # c. load an already fitted model (after optimization/tuning)

    model = None
    params = None
    algorithms = None

    if mode in ['internal_test', 'external_test']:
        # load model from defaults.OUTPUT.FITTED_MODEL_PATH
        model = load_obj(defaults.OUTPUT.FITTED_MODEL_PATH)

    return {
        "data": (X, y),
        "defaults": defaults,
        "model": model,
    }


@click.group()
def cli():
    pass


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def train(fitted_model_filename):
    """
    Train the model on the training set and save the fitted model to
    output/fitted_models/<fitted_model_filename>.pkl directory.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to dump the fitted model.
    """
    click.echo("Mode: training.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)

    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    boot_data = bootstrap(new_options, mode="train")
    defaults = boot_data['defaults']

    X_train, y_train = boot_data['data']
    X_train["symptoms"] = y_train
    print(os.getcwd())
    X_train.to_csv("../input/train_preprocessed.csv", index=False)
    fitted_model = train_model(X_train, y_train, defaults)

    # dump fitted model
    os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    save_obj(fitted_model, defaults.OUTPUT.FITTED_MODEL_PATH)


@click.command()
@click.option("--which", default='internal')
@click.option("--fitted_model_filename", default='model.pkl')
@click.option("--which_threshold", default='all')
def test(which, fitted_model_filename, which_threshold):
    """
    Test a fitted model on the test set.
    By default, we look for a fitted model in the output/fitted_model
    directory.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to load the already fitted model.
        b) to derive the model name (part preceding the .extension) which is then used to
        save the test results in output/results/test_results_<model name>.csv
    """
    click.echo("Mode: test.")
    defaults = get_defaults()

    # bootstrap input
    fitted_model_filename = add_extension(fitted_model_filename)
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    mode = "{}_test".format(which)
    boot_data = bootstrap(new_options, mode=mode)

    model = boot_data['model']
    X_test, y_test = boot_data['data']
    defaults = boot_data['defaults']

    eval_metrics = get_evaluation_metrics()
    threshold = get_threshold(defaults, which_threshold)
    test_results = test_performance(conf=defaults,
                                    model=model,
                                    X_test=X_test, y_test=y_test,
                                    eval_metrics=eval_metrics,
                                    threshold=threshold)
    results = pd.DataFrame(test_results.values(), index=test_results.keys(), columns=["test"])

    results_filename = "test_results_{}.csv".format(fitted_model_filename.split(".")[0])
    results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, results_filename)
    results.to_csv(results_path)


@click.command()
@click.option('--exp_name', default="model.pkl")
def cross_validation(exp_name):
    """
    Evaluate an estimator by means of cross-validation on the training data.

    Here exp_name is used as follows:
    a) to derive the model name (part preceding the .extension) which is then used to
        save the test results in output/results/test_results_<model name>.csv
    Note that is not used to update the fitted_model_path since we do not store
    any fitted model when we cross-validate.
    """
    click.echo("Mode: cross-validation.")

    # bootstrap input
    boot_data = bootstrap(new_options=None, mode="cv")

    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']

    cv = RepeatedStratifiedKFold(n_splits=defaults.EVAL.OUTER_N_SPLITS,
                                 n_repeats=defaults.EVAL.N_REPEATS,
                                 random_state=defaults.MISC.SEED)
    eval_metrics = get_evaluation_metrics()

    cv_results, cv_preds = cross_validate(X=X_train, y=y_train,
                          cv=cv,
                          conf=defaults)
    
    cv_results_formatted = show_cross_val_results(cv_results, conf=defaults)

    cv_results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, "cv_results_{}.csv".format(exp_name))
    cv_results_formatted.to_csv(cv_results_path)

    # save predictions
    cv_preds_path = os.path.join(defaults.OUTPUT.PREDS_PATH, "cv_pooled_preds_{}.pkl".format(exp_name))
    save_obj(cv_preds, cv_preds_path)

cli.add_command(train)
cli.add_command(test)
cli.add_command(cross_validation)

if __name__ == '__main__':
    cli()
