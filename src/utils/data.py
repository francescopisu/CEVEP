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


def load_split_preprocess(conf: CfgNode):
    """
    This function loads the data, splits the data into
    training and test set and apply basic preprocessing.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node to access configuration data.
    """
    train_filename = 'train'
    test_filename = 'test'

    df = pd.read_csv(conf.DATA.DATASET_PATH)

    df["smoker_status"] = df["smoker_status"].apply(lambda v: "no" if v == "no" else "yes")
    df["calcification_type_left"] = df["calcification_type_left"].apply(lambda v: v.replace(" ", ""))
    df["calcification_type_right"] = df["calcification_type_right"].apply(lambda v: v.replace(" ", ""))

    # drop unuseful features
    df = df.drop(conf.DATA.TO_DROP, axis=1)

    X, y = df.drop(conf.DATA.TARGET, axis=1), df[conf.DATA.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=conf.DATA.TEST_SIZE,
                                                        random_state=conf.MISC.SEED,
                                                        stratify=y)

    le = LabelEncoder()
    oe = OrdinalEncoder()

    # start pre-processing
    to_encode = conf.DATA.TO_LABEL_ENCODE
    if conf.DATA.APPLY_OHE:
        X_train = pd.get_dummies(X_train, columns=conf.DATA.TO_OHE, prefix=conf.DATA.PREFIXES)
        X_test = pd.get_dummies(X_test, columns=conf.DATA.TO_OHE, prefix=conf.DATA.PREFIXES)
    else:
        to_encode += conf.DATA.TO_OHE

        train_filename += '_no_ohe'
        test_filename += '_no_ohe'

    if conf.DATA.SCALE:
        scaler = StandardScaler()
        X_train[conf.DATA.TO_SCALE] = scaler.fit_transform(X_train[conf.DATA.TO_SCALE])
        X_test[conf.DATA.TO_SCALE] = scaler.transform(X_test[conf.DATA.TO_SCALE])
    else:
        train_filename += '_no_scale'
        test_filename += '_no_scale'

    y_train = le.fit_transform(y_train)
    X_train[to_encode] = oe.fit_transform(X_train[to_encode]).astype(np.uint8)

    # test set
    y_test = le.transform(y_test)
    X_test[to_encode] = oe.transform(X_test[to_encode]).astype(np.uint8)

    # add target vectors to feature matrices
    X_train[conf.DATA.TARGET] = y_train
    X_test[conf.DATA.TARGET] = y_test

    train_filename += f'_{conf.MISC.SEED}.csv'
    test_filename += f'_{conf.MISC.SEED}.csv'

    train_data_path = conf.DATA.BASE_INPUT_PATH + f'/{train_filename}'
    test_data_path = conf.DATA.BASE_INPUT_PATH + f'/{test_filename}'

    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)
