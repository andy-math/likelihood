from __future__ import annotations

from datetime import datetime
from typing import Generic, Optional, Tuple, TypeVar

import numpy
import overloads.dyn_typing as dynT
from overloads.typedefs import ndarray
from overloads.shortcuts import assertNoInfNaN, isunique

T = TypeVar("T", int, datetime)


class Variables(Generic[T]):
    data_names: Tuple[str, ...]
    date: Tuple[T, ...]
    sheet: ndarray

    def __init__(
        self, date: Tuple[T, ...], *datas: Tuple[str, Optional[ndarray]]
    ) -> None:
        length_tracker = dynT.SizeVar()
        assert dynT.Union(
            dynT.List(dynT.Class(datetime), length_tracker),
            dynT.List(dynT.Int(), length_tracker),
        )._isinstance(list(date))
        for d in datas:
            assert dynT.Tuple(
                (
                    dynT.Str(),
                    dynT.Union(
                        dynT.Optional(dynT.NDArray(numpy.float64, (length_tracker,))),
                        dynT.Optional(dynT.NDArray(numpy.int64, (length_tracker,))),
                    ),
                )
            )._isinstance(d)
            if d[1] is not None and d[1].dtype == numpy.int64:
                d = (d[0], d[1].astype(numpy.float64))
            if d[1] is not None:
                assertNoInfNaN(d[1])
        assert length_tracker.value is not None
        data_names = tuple(name for name, _ in datas)
        assert isunique(data_names)

        zeros = numpy.zeros((length_tracker.value,))
        self.data_names = data_names
        self.date = date
        self.sheet = numpy.stack(
            [var if var is not None else zeros for _, var in datas], axis=1
        )

    def index(self, *, from_: int, to: int) -> Variables[T]:
        return Variables(
            self.date[from_:to],
            *(
                (name, self.sheet[from_:to, i])
                for i, name in enumerate(self.data_names)
            ),
        )

    def subset(self, *data_names: str) -> Variables[T]:
        assert isunique(data_names)
        for name in data_names:
            assert name in self.data_names, f"变量{name}未出现于此变量表中"
        return Variables(
            self.date,
            *(
                (name, self.sheet[:, self.data_names.index(name)])
                for name in data_names
            ),
        )
