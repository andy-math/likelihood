from typing import Any, Callable, Optional

from numba import njit  # type: ignore
from numba.core.typing.templates import Signature  # type: ignore


class Jitted_Function:
    origin: Optional[Callable[..., Any]]
    jitted: Optional[Callable[..., Any]]

    def __init__(self, origin: Callable[..., Any], signature: Signature) -> None:
        self.origin = origin
        self.jitted = njit(signature)(origin)

    def __call__(self, *args: Any, debug: bool, **kwds: Any) -> Any:
        if debug:
            assert self.origin is not None
            return self.origin(*args, **kwds)
        else:
            assert self.jitted is not None
            return self.jitted(*args, **kwds)
