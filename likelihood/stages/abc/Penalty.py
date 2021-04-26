from abc import ABCMeta
from typing import List, Optional, Tuple, TypeVar

from likelihood.stages.abc.Stage import Constraints, Stage

_Penalty_gradinfo_t = TypeVar("_Penalty_gradinfo_t")


class Penalty(Stage[_Penalty_gradinfo_t], metaclass=ABCMeta):
    coeff_names: Tuple[str, ...]
    index: Optional[Tuple[int, ...]]

    def __init__(
        self,
        coeff_names: Tuple[str, ...],
        input: Tuple[int, ...],
        output: Tuple[int, ...],
    ) -> None:
        super().__init__((), input, output)
        self.coeff_names = coeff_names

    def make_index(self, names: Tuple[str, ...]) -> None:
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
        self.index = tuple(index)

    def get_constraint(self) -> Constraints:
        assert False  # pragma: no cover
