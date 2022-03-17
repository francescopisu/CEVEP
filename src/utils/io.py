"""
This file contains methods related to input-output operations, such as
loading and saving models.
"""

import pickle
from typing import Tuple, Union, Any
import pandas as pd  # type: ignore
from yacs.config import CfgNode  # type: ignore

from src._typing import PathLike, Estimator, ArrayLike


def load_data(conf: CfgNode, test: bool = False) -> Tuple[ArrayLike, ArrayLike]:
    """
    Loads data located in folder conf.DATA.<TRAIN|TEST>_DATA_PATH stored in csv format.

    Parameters
    ----------
    conf: CfgNode
        A Configuration node storing configuration information.
    test: bool (default = False)
        Whether to load the test data or not.

    Returns
    -------
    Union[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
        A tuple of features and ground-truths corresponding to the test set (if train = False)
        or a tuple of train features/ground-truths and validation features/ground-truths
        if train = True and reserve_dev_set = True.
    """
    data_path = conf.DATA.TRAIN_DATA_PATH
    if test:
        data_path = conf.DATA.TEST_DATA_PATH

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
