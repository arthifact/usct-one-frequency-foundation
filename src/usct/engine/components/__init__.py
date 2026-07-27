"""Built-in stage components; importing this package registers every adapter."""

# Each submodule runs its register(...) calls at import time.
from usct.engine.components import (  # noqa: F401  (registration side effects)
    forcing,
    forward,
    measurement,
    medium,
    mesh,
)
