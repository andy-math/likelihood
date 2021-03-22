from abc import ABCMeta
from typing import List, Optional, Sequence, Tuple, TypeVar

from likelihood.stages.abc.Stage import Stage
from numerical.typedefs import ndarray

_Penalty_gradinfo_t = TypeVar("_Penalty_gradinfo_t")


class Penalty(Stage[_Penalty_gradinfo_t], metaclass=ABCMeta):
    coeff_names: List[str]
    index: Optional[List[int]]

    def __init__(
        self, coeff_names: List[str], input: Sequence[int], output: Sequence[int]
    ) -> None:
        super().__init__((), input, output)
        self.coeff_names = coeff_names

    def make_index(self, names: List[str]) -> None:
        index: List[int] = []
        for c in self.coeff_names:
            found = False
            for i, n in enumerate(names):
                if c == n:
                    index.append(i)
                    found = True
                    break
            if not found:
                assert False  # pragma: no cover
        self.index = index

    def get_constraint(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        assert False  # pragma: no cover
