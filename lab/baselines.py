"""Load frozen third-party policies for local development and evaluation."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


AgentFunction = Callable[[dict[str, Any]], dict[str, Any]]


# These are local evaluation opponents, not submission candidates. Each file
# stays separate from our frozen V45 development base for independent audits.
EXTERNAL_BASELINES: dict[str, tuple[str, str]] = {
    "sdy2842": ("sdy2842", "agent"),
    "v46": ("v46", "agent"),
    "indar_top10": ("indarkarhana_top10", "agent"),
    # The notebook publishes historical wrappers; its manifest selects V58.
    "kaito_v58": ("kaito_v58", "kaggle_agent_v58"),
    "nathan_pipe7": ("nathan_pipe7", "agent"),
    "boatlee_v16": ("boatlee_v16", "agent"),
    "indar_pasture": ("indar_pasture", "kaggriculture_e776_agent"),
}


def _load_module(source: Path, module_name: str) -> ModuleType:
    """Execute one source file in a fresh, isolated module namespace."""

    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load third-party policy: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_v45_base_module() -> ModuleType:
    """Load one isolated V45 module from the SHA-verified frozen source.

    The source stays in ``third_party/v45`` so its upstream notices and exact
    provenance remain visible.  Variants must be implemented separately rather
    than editing that frozen copy in place.
    """

    source = Path(__file__).resolve().parents[1] / "third_party" / "v45" / "main.py"
    return _load_module(source, "local_v45_reference")


def load_v45_base_agent() -> AgentFunction:
    """Load a fresh callable V45 base policy for one local episode."""

    module = load_v45_base_module()
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise RuntimeError("V45 reference does not expose a callable agent")
    return agent


# Temporary compatibility alias for historical notebooks and experiment records.
load_v45_agent = load_v45_base_agent


def load_external_agent(name: str) -> AgentFunction:
    """Load a fresh audited opponent policy for a single local episode.

    The reload is necessary because several public policies retain route
    cursors and telemetry in module globals.
    """

    try:
        directory, entry_point = EXTERNAL_BASELINES[name]
    except KeyError as error:
        choices = ", ".join(sorted(EXTERNAL_BASELINES))
        raise ValueError(f"Unknown external baseline {name!r}; choose from {choices}") from error
    source = Path(__file__).resolve().parents[1] / "third_party" / directory / "main.py"
    if name == "indar_pasture":
        # This public submission is a multi-file package.  Clear its package
        # modules before each episode so module-level cursors cannot leak, and
        # expose its frozen directory only while executing its entry file.
        for module_name in tuple(sys.modules):
            if module_name == "e776_pkg" or module_name.startswith("e776_pkg."):
                sys.modules.pop(module_name, None)
        sys.path.insert(0, str(source.parent))
        try:
            module = _load_module(source, f"local_kaggriculture_{name}")
        finally:
            sys.path.pop(0)
    else:
        module = _load_module(source, f"local_kaggriculture_{name}")
    agent = getattr(module, entry_point, None)
    if not callable(agent):
        raise RuntimeError(f"External baseline {name!r} lacks callable {entry_point!r}")
    return agent
