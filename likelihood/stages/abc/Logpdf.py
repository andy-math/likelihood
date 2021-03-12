from __future__ import annotations

from abc import ABCMeta
from typing import TypeVar

from likelihood.stages.abc.Stage import Stage

_Logpdf_gradinfo_t = TypeVar("_Logpdf_gradinfo_t")


class Logpdf(Stage[_Logpdf_gradinfo_t], metaclass=ABCMeta):
    pass
