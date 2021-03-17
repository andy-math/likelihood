from abc import ABCMeta
from typing import TypeVar

from likelihood.stages.abc.Stage import Stage

_Penalty_gradinfo_t = TypeVar("_Penalty_gradinfo_t")


class Penalty(Stage[_Penalty_gradinfo_t], metaclass=ABCMeta):
    pass
