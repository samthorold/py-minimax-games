"""Type stubs for `minimax_games._native`, the Rust extension module built by
maturin from rust/ (docs/star-realms.md, "Compiled engine").

At run time `_native` is the compiled module beside this directory, which wins
over it on import (a directory without `__init__.py` is only a namespace
package); these stubs are for the type checker. Keep them in step with
rust/src/python.rs.
"""

from minimax_games._native import star_realms as star_realms
