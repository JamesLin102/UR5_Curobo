"""Scene registry: ``load("demo_cube")`` -> SceneSpec.

Both processes resolve their scene through here, by the same name, so a
mismatch between them is a wrong name rather than a silent geometry drift.
"""

import importlib
import pkgutil
from typing import List

from .base import Body, SceneSpec, WatchBox

# Re-exported so a scene module, and anything reading one, can import the whole
# contract from `scenes` without reaching into `scenes.base`.
__all__ = ["Body", "SceneSpec", "WatchBox", "DEFAULT", "available", "load"]

DEFAULT = "demo_cube"


def available() -> List[str]:
    """Every scene module in this package, for --help and error messages."""
    return sorted(
        m.name for m in pkgutil.iter_modules(__path__) if m.name != "base"
    )


def load(name: str) -> SceneSpec:
    """Import scenes/<name>.py and return its SCENE.

    Raises with the list of real options rather than a bare ImportError: the
    two processes are started separately and a typo in one of them would
    otherwise show up as an inexplicable disagreement about the world.
    """
    try:
        mod = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"unknown scene {name!r}; available: {', '.join(available())}"
        ) from exc
    scene = getattr(mod, "SCENE", None)
    if not isinstance(scene, SceneSpec):
        raise SystemExit(f"scene {name!r} does not define a SceneSpec named SCENE")
    return scene
