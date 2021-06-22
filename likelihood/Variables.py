from __future__ import annotations

from typing import Optional, Tuple

import numpy
import overloads.dyn_typing as dynT
from numerical.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN, isunique


class Variables:
    data_names: Tuple[str, ...]
    sheet: ndarray

    def __init__(self, *datas: Tuple[str, Optional[ndarray]]) -> None:
        length_tracker = dynT.SizeVar()
        for d in datas:
            assert dynT.Tuple(
                (
                    dynT.Str(),
                    dynT.Optional(dynT.NDArray(numpy.float64, (length_tracker,))),
                )
            )._isinstance(d)
            if d[1] is not None:
                assertNoInfNaN(d[1])
        assert length_tracker.value is not None
        data_names = tuple(name for name, _ in datas)
        assert isunique(data_names)

        zeros = numpy.zeros((length_tracker.value,))
        self.data_names = data_names
        self.sheet = numpy.stack(
            [var if var is not None else zeros for _, var in datas], axis=1
        )

    def index(self, *, from_: int, to: int) -> Variables:
        return Variables(
            *((name, self.sheet[from_:to, i]) for i, name in enumerate(self.data_names))
        )
