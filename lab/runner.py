"""Run a deterministic local Kaggriculture match and print its terminal rewards."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

from kaggle_environments import make

from .baselines import EXTERNAL_BASELINES, load_external_agent, load_v45_base_agent
from .agents import (
    carrot_loop_agent,
    melon_loop_agent,
    strawberry_loop_agent,
    tomato_loop_agent,
    wheat_loop_agent,
)
from .portfolio_agent import make_portfolio_agent
from .route14_agent import make_route14_agent
from .region_phase1 import make_region_phase1_agent
from .route14_phase1 import make_route14_phase1_agent

Agent = str | Callable[[dict[str, Any]], dict[str, Any]]
AgentSpec = str | Callable[[dict[str, Any]], dict[str, Any]]

AGENTS: dict[str, Agent] = {
    "random": "random",
    "starter": "starter",
    "wheat": wheat_loop_agent,
    "carrot": carrot_loop_agent,
    "tomato": tomato_loop_agent,
    "strawberry": strawberry_loop_agent,
    "melon": melon_loop_agent,
    "v45_base": "__fresh_v45_base__",
    # Compatibility name for existing records; use v45_base in new experiments.
    "v45": "__fresh_v45_base__",
    "portfolio_executor": "__fresh_portfolio_executor__",
    "route14": "__fresh_route14__",
    "route14_phase1": "__fresh_route14_phase1__",
    "region_phase1": "__fresh_region_phase1__",
    **{name: f"__fresh_{name}__" for name in EXTERNAL_BASELINES},
}


def _resolve_agent(candidate: AgentSpec) -> Agent:
    """Return a fresh stateful V45 base policy for every independent episode."""

    if callable(candidate):
        return candidate
    if candidate not in AGENTS:
        raise ValueError(f"Agents must be chosen from: {', '.join(sorted(AGENTS))}")
    if candidate in {"v45", "v45_base"}:
        return load_v45_base_agent()
    if candidate == "portfolio_executor":
        return make_portfolio_agent()
    if candidate == "route14":
        return make_route14_agent()
    if candidate == "route14_phase1":
        return make_route14_phase1_agent()
    if candidate == "region_phase1":
        return make_region_phase1_agent()
    if candidate in EXTERNAL_BASELINES:
        return load_external_agent(candidate)
    return AGENTS[candidate]


def _agent_label(candidate: AgentSpec) -> str:
    if isinstance(candidate, str):
        return candidate
    return str(getattr(candidate, "label", getattr(candidate, "__name__", "custom")))


def run_match(
    left: AgentSpec,
    right: AgentSpec,
    steps: int,
    seed: int,
    *,
    prepare: Callable[[Any], Callable[[], None] | None] | None = None,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one reproducible episode without rendering or writing a replay.

    ``prepare`` receives the environment after it is built and before the
    episode starts.  It may install a temporary recorder and return a restore
    callable.  The restore callable runs even when the episode raises.
    """

    if steps < 2:
        raise ValueError("steps must be at least 2")

    left_agent = _resolve_agent(left)
    right_agent = _resolve_agent(right)
    episode = {"episodeSteps": steps, "seed": seed}
    if configuration:
        episode.update(configuration)

    environment = make("kaggriculture", configuration=episode, debug=True)
    restore = prepare(environment) if prepare is not None else None
    started = time.perf_counter()
    try:
        environment.run([left_agent, right_agent])
    finally:
        if restore is not None:
            restore()
    elapsed_seconds = time.perf_counter() - started
    terminal = environment.steps[-1]

    return {
        "left": _agent_label(left),
        "right": _agent_label(right),
        "steps": steps,
        "seed": seed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "rewards": [state.reward for state in terminal],
        "statuses": [state.status for state in terminal],
        "telemetry": [
            dict(getattr(left_agent, "telemetry", {})),
            dict(getattr(right_agent, "telemetry", {})),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", choices=sorted(AGENTS), default="v45_base")
    parser.add_argument("--right", choices=sorted(AGENTS), default="starter")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_match(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
