"""
Helper functions to deal with data.
"""
import numpy as np
import math
from typing import Tuple, Callable
from yacs.config import CfgNode  # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

from src._typing import ArrayLike, Seed


def load_preprocess(conf: CfgNode, which: str = "train") -> Tuple[ArrayLike, ArrayLike]:
    """
    Load the dataframe and apply basic preprocessing.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node.
    which: str (Default = "train")
        Which dataset to load.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        A tuple consisting of a design matrix and a ground-truth vector corresponding
        to the target.
    """
    if which == 'external':
        data_path = conf.DATA.EXTERNAL_DATA_PATH
    else:
        data_path = conf.DATA.TRAIN_DATA_PATH

    df = pd.read_csv(data_path)
    df = df.drop(conf.DATA.TO_DROP, axis=1, errors="ignore")
    X, y = df.drop(conf.DATA.TARGET, axis=1), df[conf.DATA.TARGET]

    if "smoker_status" in X.columns:
        X["smoker_status"] = X["smoker_status"].apply(lambda v: "no" if v == "no" else "yes")
    if "calcification_type_left" in X.columns:
        X["calcification_type_left"] = X["calcification_type_left"].apply(lambda v: v.replace(" ", ""))
    if "calcification_type_right" in X.columns:
        X["calcification_type_right"] = X["calcification_type_right"].apply(lambda v: v.replace(" ", ""))

    le = LabelEncoder()
    oe = OrdinalEncoder()

    to_encode = conf.DATA.TO_LABEL_ENCODE
    if conf.DATA.APPLY_OHE:
        X = pd.get_dummies(X, columns=conf.DATA.TO_OHE, prefix=conf.DATA.PREFIXES)
    else:
        to_encode += conf.DATA.TO_OHE

    y = le.fit_transform(y)
    X[to_encode] = oe.fit_transform(X[to_encode]).astype(np.uint8)

    # convert uint8 dtypes to categorical
    cat_cols = X.select_dtypes(include=np.uint8).columns.tolist()
    X[cat_cols] = X[cat_cols].astype("category")

    # subset data if necessary
    if conf.DATA.SUBSET_DATA:
        if conf.DATA.WHICH_SUBSET == "clinical":
            X = X[conf.DATA.CLINICAL_FEATURE_SUBSET]
        elif conf.DATA.WHICH_SUBSET == "calcification":
            X = X.drop(conf.DATA.CLINICAL_FEATURE_SUBSET, axis=1)

    return X, y
