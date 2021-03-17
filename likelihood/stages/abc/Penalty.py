from abc import ABCMeta
from typing import Tuple, TypeVar

from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray

_Penalty_gradinfo_t = TypeVar("_Penalty_gradinfo_t")


class Penalty(Stage[_Penalty_gradinfo_t], metaclass=ABCMeta):
    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        assert False  # pragma: no cover
