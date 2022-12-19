"""
This file contains methods related to input-output operations, such as
loading and saving models.
"""

import pickle
from typing import Tuple, Union, Any
import pandas as pd  # type: ignore
from yacs.config import CfgNode  # type: ignore

from src._typing import PathLike, Estimator, ArrayLike


def load_data(conf: CfgNode, which: str = "train") -> Tuple[ArrayLike, ArrayLike]:
    """
    Loads data located in folder conf.DATA.<TRAIN|TEST>_DATA_PATH stored in csv format.

    Parameters
    ----------
    conf: CfgNode
        A Configuration node storing configuration information.
    which: str (default = "train")
        Which dataset to load.

    Returns
    -------
    Union[Tuple[ArrayLike, ArrayLike]]
        A tuple of features and ground-truths corresponding to
        the specified dataset.
    """
    if which == 'internal_test':
        data_path = conf.DATA.TEST_DATA_PATH
    elif which == 'external_test':
        data_path = conf.DATA.EXTERNAL_DATA_PATH
    else:
        data_path = conf.DATA.TRAIN_DATA_PATH

    data = pd.read_csv(data_path)
    X, y = data.drop(conf.DATA.TARGET, axis=1), data[conf.DATA.TARGET]

    return X, y


def save_obj(obj: Any, path: PathLike):
    """
    Dump the model to the specified path.
    Parameters
    ----------
    obj: Estimator
        The model to be saved
    path: PathLike
        The path where to save the model
    """
    with open(path, "wb") as f_w:
        pickle.dump(obj, f_w)


def load_obj(path: PathLike) -> Any:
    """
    Load the model located at the specified path.

    Parameters
    ----------
    path: PathLike
        The path where to find the model.

    Returns
    -------
    Estimator
        The loaded model
    """
    with open(path, "rb") as f_r:
        obj = pickle.load(f_r)

    return obj
