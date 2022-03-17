"""
This file implements a CLI to start the training, tuning and testing of the model.
"""
from typing import Dict, Optional, List
import os
import numpy as np
import pandas as pd
import click
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from src.utils.io import load_data, load_obj, save_obj
from src.utils.misc import show_cross_val_results, add_extension
from src.utils.data import load_split_preprocess
from config.defaults import get_defaults
from src.evaluate import (
    test_performance,
    cross_val_performance,
    get_evaluation_metrics,
    nested_cv
)
from src.train import (
    _get_lgbm_params,
    _get_base_lgbm_params,
    fit_lgbm, tune_lgbm
)


def bootstrap(new_options: Optional[List] = None,
              mode: str = "train") -> Dict:
    """
    This function is responsible for the bootstrap phase prior to
    training or testing the model.
    It is responsible for:
        - loading the default configuration values
        - updating defaults by merging CLI arguments
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
    X, y = load_data(defaults, test=mode == 'test')

    # get LightGBM paramaters or load an already fitted model
    model = None
    params = None
    if mode == 'train' or mode == 'cv':
        params = _get_lgbm_params(conf=defaults)
    elif mode == 'ncv' or mode == 'tune':
        params = _get_base_lgbm_params(conf=defaults)
    elif mode == 'test':
        # load model from defaults.OUTPUT.FITTED_MODEL_PATH
        model = load_obj(defaults.OUTPUT.FITTED_MODEL_PATH)

    return {
        "data": (X, y),
        "defaults": defaults,
        "model": model,
        "params": params
    }


@click.group()
def cli():
    pass


@click.command()
def data_bootstrap():
    """
    Command to load the data, split the data into
    training and test set and apply basic preprocessing.
    """
    defaults = get_defaults()
    load_split_preprocess(defaults)


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
    lgbm_params = boot_data['params']
    defaults = boot_data['defaults']

    X_train, y_train = boot_data['data']
    fitted_model = fit_lgbm(defaults, lgbm_params, X_train, y_train)

    # dump fitted model
    os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    save_obj(fitted_model, defaults.OUTPUT.FITTED_MODEL_PATH)


@click.command()
@click.option("--fitted_model_filename", default='model.pkl')
def test(fitted_model_filename):
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
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]
    boot_data = bootstrap(new_options, mode="test")

    model = boot_data['model']
    X_test, y_test = boot_data['data']
    defaults = boot_data['defaults']

    eval_metrics = get_evaluation_metrics()
    test_results = test_performance(conf=defaults,
                                    model=model,
                                    X_test=X_test, y_test=y_test,
                                    eval_metrics=eval_metrics)
    results = pd.DataFrame(test_results.values(), index=test_results.keys(), columns=["test"])

    results_filename = "test_results_{}.csv".format(fitted_model_filename.split(".")[0])
    results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, results_filename)
    results.to_csv(results_path)


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def cross_validate(fitted_model_filename):
    """
    Evaluate an estimator by means of cross-validation on the training data.

    Here fitted_model_filename is used as follows:
    a) to derive the model name (part preceding the .extension) which is then used to
        save the test results in output/results/test_results_<model name>.csv
    Note that is not used to update the fitted_model_path since we do not store
    any fitted model when we cross-validate.
    """
    click.echo("Mode: cross-validation.")

    # bootstrap input
    boot_data = bootstrap(new_options=None, mode="cv")

    params = boot_data['params']
    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']

    cv = RepeatedStratifiedKFold(n_splits=defaults.EVAL.N_SPLITS,
                                 n_repeats=defaults.EVAL.N_REPEATS,
                                 random_state=defaults.MISC.SEED)
    eval_metrics = get_evaluation_metrics()

    perf = cross_val_performance(conf=defaults,
                                 params=params,
                                 X=X_train, y=y_train,
                                 cv=cv,
                                 eval_metrics=eval_metrics,
                                 set_names=["train", "val"],
                                 instantiate_each_time=False)
    cv_results = show_cross_val_results(perf)
    results_filename = "cross_val_results_{}.csv".format(fitted_model_filename.split(".")[0])
    results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH_PATH, results_filename)
    cv_results.to_csv(results_path)


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def nested_cross_validation(fitted_model_filename):
    """
    Model selection and evaluation by means of nested cross-validation.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to dump the fitted model.
    """
    click.echo("Mode: Nested cross-validation.")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    # don't reserve dev set at this point since we need to do it in each cv fold
    boot_data = bootstrap(new_options=new_options, mode="ncv")

    base_params = boot_data['params']
    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']

    cv = StratifiedKFold(n_splits=defaults.EVAL.N_SPLITS,
                         shuffle=defaults.EVAL.SHUFFLE,
                         random_state=defaults.MISC.SEED)
    final_model, best_params, outer_results = nested_cv(defaults,
                                                        base_params=base_params,
                                                        X=X_train, y=y_train,
                                                        outer_cv=cv,
                                                        inner_cv=cv)

    # dump results
    fitted_model_best_params_path = os.path.join(defaults.OUTPUT.PARAMS_PATH,
                                                 "best_params_{}".format(fitted_model_filename))
    # ncv_results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH,
    #                                 "ncv_results_{}".format(fitted_model_filename))

    os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    save_obj(final_model, defaults.OUTPUT.FITTED_MODEL_PATH)
    save_obj(best_params, fitted_model_best_params_path)
    # save_obj(outer_results, ncv_results_path)


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def tune(fitted_model_filename):
    """
    Perform hyperparameter tuning of LightGBM classifier.
    """
    click.echo("Mode: tuning.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]
    boot_data = bootstrap(new_options, mode="tune")

    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']
    base_params = boot_data['params']

    cv = StratifiedKFold(n_splits=defaults.EVAL.N_SPLITS,
                         shuffle=defaults.EVAL.SHUFFLE,
                         random_state=defaults.MISC.SEED)
    tuned_model, best_params = tune_lgbm(conf=defaults,
                                         base_params=base_params,
                                         X=X_train, y=y_train,
                                         cv=cv)

    # dump fitted model
    fitted_model_best_params_path = os.path.join(defaults.OUTPUT.PARAMS_PATH,
                                                 "best_params_{}".format(fitted_model_filename))

    os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    save_obj(tuned_model, defaults.OUTPUT.FITTED_MODEL_PATH)
    save_obj(best_params, fitted_model_best_params_path)


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def get_predictions(fitted_model_filename):
    """
    Use a fitted model to predict probabilities and save it
    in the results folder.
    """
    click.echo("Mode: predicting probabilities.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]
    boot_data = bootstrap(new_options, mode="test")

    model = boot_data['model']
    X_test, y_test = boot_data['data']
    test_proba = model.predict(X_test)
    test_proba = np.c_[y_test, test_proba]

    boot_data = bootstrap(new_options, mode="train")
    X_train, y_train = boot_data['data']
    train_proba = model.predict(X_train)
    train_proba = np.c_[y_train, train_proba]

    train_results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, "train_target_proba.csv")
    test_results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, "test_target_proba.csv")
    pd.DataFrame(train_proba, columns=['target', 'proba']).to_csv(train_results_path, index=False)
    pd.DataFrame(test_proba, columns=['target', 'proba']).to_csv(test_results_path, index=False)


cli.add_command(data_bootstrap)
cli.add_command(train)
cli.add_command(test)
cli.add_command(cross_validate)
cli.add_command(nested_cross_validation)
cli.add_command(tune)
cli.add_command(get_predictions)

if __name__ == '__main__':
    cli()
