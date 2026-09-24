"""Scene registry: ``load("pick_place")`` -> SceneSpec.

A scene belongs to its example: ``load(name)`` reads ``scripts/<name>/scene.py``
and returns its SCENE. Both processes resolve their scene through here, by the
same name, so a mismatch between them is a wrong name rather than a silent
geometry drift.

``register(name, spec)`` adds a scene built in code (a check's synthetic one),
for this process only.
"""

import importlib
import os
from typing import Dict, List

from .base import Body, SceneSpec, WatchBox

# Re-exported so a scene module, and anything reading one, can import the whole
# contract from `scenes` without reaching into `scenes.base`.
__all__ = ["Body", "SceneSpec", "WatchBox", "DEFAULT", "available", "load", "register"]

DEFAULT = "pick_place"

# scripts/, where the examples live, one package each.
_EXAMPLES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_REGISTERED: Dict[str, SceneSpec] = {}


def available() -> List[str]:
    """Every example with a scene.py, for --help and error messages."""
    return sorted(
        d for d in os.listdir(_EXAMPLES)
        if os.path.isfile(os.path.join(_EXAMPLES, d, "scene.py"))
        and os.path.isfile(os.path.join(_EXAMPLES, d, "__init__.py"))
    )


def register(name: str, spec: SceneSpec) -> None:
    """Make `spec` loadable as `name` in this process."""
    if not isinstance(spec, SceneSpec):
        raise TypeError(f"scene {name!r} is not a SceneSpec")
    _REGISTERED[name] = spec


def load(name: str) -> SceneSpec:
    """Import scripts/<name>/scene.py and return its SCENE.

    Raises with the list of real options rather than a bare ImportError: the
    two processes are started separately and a typo in one of them would
    otherwise show up as an inexplicable disagreement about the world.
    """
    if name in _REGISTERED:
        return _REGISTERED[name]
    if name not in available():
        raise SystemExit(f"unknown scene {name!r}; available: {', '.join(available())}")
    scene = getattr(importlib.import_module(f"{name}.scene"), "SCENE", None)
    if not isinstance(scene, SceneSpec):
        raise SystemExit(f"scene {name!r} does not define a SceneSpec named SCENE")
    return scene
