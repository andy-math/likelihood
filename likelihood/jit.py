from __future__ import annotations

import pickle
from typing import Any, Callable, Dict, Generic, NoReturn, Tuple, TypeVar

import numba  # type: ignore

T2 = TypeVar("T2", covariant=True)


_Jitted_Function_Cache: Dict[bytes, Tuple[Any, Any]] = {}


class Jitted_Function(Generic[T2]):
    signature: numba.core.typing.templates.Signature
    dependent: Tuple[Jitted_Function[Any], ...]
    pickled_bytecode: bytes

    def __init__(
        self,
        signature: numba.core.typing.templates.Signature,
        dependent: Tuple[Jitted_Function[Any], ...],
        generator: Callable[..., T2],
    ) -> None:
        # picklable test
        self.pickled_bytecode = pickle.dumps(generator)
        self.signature = signature
        self.dependent = dependent

    def _compile(self) -> Tuple[T2, T2]:
        if self.pickled_bytecode in _Jitted_Function_Cache:
            return _Jitted_Function_Cache[self.pickled_bytecode]

        generator: Callable[..., T2] = pickle.loads(self.pickled_bytecode)

        func = numba.njit(self.signature)(
            generator(*[x.func() for x in self.dependent])
        )
        py_func = generator(*[x.py_func() for x in self.dependent])

        _Jitted_Function_Cache[self.pickled_bytecode] = (func, py_func)

        return func, py_func

    def func(self) -> T2:
        func, _ = self._compile()
        return func

    def py_func(self) -> T2:
        _, py_func = self._compile()
        return py_func

    def __call__(_) -> NoReturn:
        assert False  # pragma: no cover
