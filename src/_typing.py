"""
This file defines custom types which are used in type signatures
throughout the codebase.
"""
from typing import Tuple, Union, TypeVar, List
import pathlib
import sys
import os
import random
import numpy
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.model_selection._split import BaseCrossValidator, _RepeatedSplits

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Union[numpy.ndarray, List[List[float]]]

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

PathLike: TypeAlias = Union[str, pathlib.Path, os.PathLike]
Estimator: TypeAlias = TypeVar("Estimator", bound=BaseEstimator)
ConfMatrixQuant: TypeAlias = Tuple[float, float, float, float]
Seed: TypeAlias = Union[int, random.seed, numpy.random.seed]
CVScheme: TypeAlias = TypeVar("CVScheme", BaseCrossValidator, _RepeatedSplits)
