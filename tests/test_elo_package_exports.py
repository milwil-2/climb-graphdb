"""Guard: every elo submodule is re-exported from the package (issue #52).

``from climber_network.elo import <submodule>`` flakes under mypy
order-dependently unless ``<submodule>`` is an explicit, always-resolved
attribute of the package. We achieve that by re-exporting each submodule in
``climber_network/elo/__init__.py`` and listing it in ``__all__`` (mirroring the
``source``/``geo`` packages). This test fails if a new submodule is added
without being re-exported, so the determinism can't silently regress.
"""

from __future__ import annotations

import pkgutil

import climber_network.elo as elo_pkg


def _discovered_submodules() -> set[str]:
    """Names of all non-private submodules physically present in the elo package."""
    return {
        name for _, name, _ in pkgutil.iter_modules(elo_pkg.__path__) if not name.startswith("_")
    }


def test_all_submodules_are_reexported() -> None:
    """Each on-disk elo submodule must be listed in ``climber_network.elo.__all__``."""
    missing = _discovered_submodules() - set(elo_pkg.__all__)
    assert not missing, (
        f"elo submodules missing from __all__: {sorted(missing)}. "
        "Add them to the re-export list in climber_network/elo/__init__.py "
        "to keep `from climber_network.elo import <submodule>` deterministic (#52)."
    )


def test_reexported_submodules_are_attributes() -> None:
    """Every re-exported submodule name must resolve as a package attribute."""
    for name in _discovered_submodules():
        assert hasattr(elo_pkg, name), (
            f"climber_network.elo.{name} is not importable as an attribute"
        )
