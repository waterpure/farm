"""Small transparent agents used only to validate the local environment."""

from __future__ import annotations


def single_tile_crop_agent(obs: dict, crop: str) -> dict:
    """Grow one crop on the spawn tile and sell it immediately.

    This deliberately weak family is useful as a smoke-test policy class: all
    variants differ only in crop economics, so paired local matches must expose the
    expected differences without confounding route planning or worker scheduling.
    """

    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    farmer_x, farmer_y = farm["farmer"]
    tile = farm["tiles"][farmer_y][farmer_x]
    market: list[list[object]] = []

    seed_costs = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
    maturity_days = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 12}
    if crop not in seed_costs:
        raise ValueError(f"Unsupported crop: {crop}")

    seeds = private["seeds"].get(crop, 0)
    if seeds == 0 and farm["money"] >= seed_costs[crop]:
        market.append(["BUY_SEED", crop, 1])

    product = private["shed"].get(crop, 0)
    if product:
        market.append(["SELL", crop, product])

    farmer_action: list[object] = ["PASS"]
    if tile is None and seeds:
        farmer_action = ["PLANT", crop]
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop:
        crop_age = obs["day"] - tile["planted_day"]
        if crop_age >= maturity_days[crop]:
            farmer_action = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer_action = ["WATER"]

    return {"farmer": farmer_action, "hands": [], "market": market}


def wheat_loop_agent(obs: dict) -> dict:
    return single_tile_crop_agent(obs, "WHEAT")


def carrot_loop_agent(obs: dict) -> dict:
    return single_tile_crop_agent(obs, "CARROT")


def tomato_loop_agent(obs: dict) -> dict:
    return single_tile_crop_agent(obs, "TOMATO")


def strawberry_loop_agent(obs: dict) -> dict:
    return single_tile_crop_agent(obs, "STRAWBERRY")


def melon_loop_agent(obs: dict) -> dict:
    return single_tile_crop_agent(obs, "MELON")
