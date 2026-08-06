"""
Dynamic Python tools for the Roshn Saudi League (RSL) Transfer Market & FFP
Advisor Agent.

Every tool returns a plain ``dict`` (JSON-serializable) with a ``success``
boolean flag so the calling agent orchestrator can feed results straight
back into a Gemini function-response turn without any further translation.
On invalid input, tools return ``{"success": False, "error": "..."}``
instead of raising, since a raised exception cannot be cleanly surfaced to
the LLM inside a function-calling loop.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Shared constants (mirrors the rules described in the RSL transfer dossier)
# ---------------------------------------------------------------------------

MAX_SENIOR_FOREIGN_PLAYERS = 8
MAX_U21_FOREIGN_PLAYERS = 2
U21_BIRTH_YEAR_CUTOFF = 2005  # Born on/after this year => U21-eligible for 2026 season
HOMEGROWN_MIN_RATIO = 0.5

CURRENCY_RATES_TO_SAR: dict[str, float] = {
    "EUR": 4.08,
    "GBP": 4.78,
    "USD": 3.75,
}

TIER_SCORE_WEIGHTS: dict[int, int] = {1: 40, 2: 25, 3: 10}
TERMS_AGREED_BONUS = 30
FEE_AGREED_BONUS = 30
RIVAL_TRANSFER_PENALTY = -20

_SQUADS_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "squads.json",
)


def _load_squads() -> dict[str, dict[str, int]]:
    """Load the squad quota snapshot from data/squads.json."""
    with open(_SQUADS_JSON_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_club(club_name: str, squads: dict[str, dict[str, int]]) -> str | None:
    """Case-insensitive / hyphen-insensitive club name resolution."""
    normalized = club_name.strip().lower().replace(" ", "-")
    for key in squads:
        if key.lower().replace(" ", "-") == normalized:
            return key
    return None


# ---------------------------------------------------------------------------
# Tool 1: calculate_transfer_score
# ---------------------------------------------------------------------------

def calculate_transfer_score(
    source_tier: int,
    personal_terms_agreed: bool,
    fee_agreed: bool,
    is_rival_transfer: bool = False,
) -> dict[str, Any]:
    """Compute a 0-100 "likelihood to happen" score for a transfer rumor.

    Weighting model:
      - Source tier:      Tier 1 = +40, Tier 2 = +25, Tier 3 = +10
      - Personal terms agreed: +30
      - Fee agreed:             +30
      - Rival-transfer penalty (player linked to a direct league rival at the
        same time): -20
      Final score is clamped to the [0, 100] range.

    Args:
        source_tier: Reliability tier of the reporting source (1, 2, or 3).
        personal_terms_agreed: Whether personal terms are reported agreed.
        fee_agreed: Whether a transfer fee is reported agreed between clubs.
        is_rival_transfer: Whether the same player is also strongly linked
            to a direct rival club, which reduces confidence in this specific
            move actually completing.

    Returns:
        A dict with the numeric score, a human-readable confidence label,
        and a breakdown of every component that contributed to the score.
    """
    if source_tier not in TIER_SCORE_WEIGHTS:
        return {
            "success": False,
            "error": f"Invalid source_tier={source_tier!r}. Must be 1, 2, or 3.",
        }

    breakdown: dict[str, int] = {"tier_base": TIER_SCORE_WEIGHTS[source_tier]}
    raw_score = breakdown["tier_base"]

    if personal_terms_agreed:
        breakdown["personal_terms_agreed"] = TERMS_AGREED_BONUS
        raw_score += TERMS_AGREED_BONUS
    if fee_agreed:
        breakdown["fee_agreed"] = FEE_AGREED_BONUS
        raw_score += FEE_AGREED_BONUS
    if is_rival_transfer:
        breakdown["rival_transfer_penalty"] = RIVAL_TRANSFER_PENALTY
        raw_score += RIVAL_TRANSFER_PENALTY

    clamped_score = max(0, min(100, raw_score))

    if clamped_score >= 80:
        confidence = "Very Likely"
    elif clamped_score >= 60:
        confidence = "Likely"
    elif clamped_score >= 40:
        confidence = "Possible"
    else:
        confidence = "Unlikely"

    return {
        "success": True,
        "transfer_score": clamped_score,
        "raw_score_before_clamp": raw_score,
        "confidence_label": confidence,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Tool 2: filter_latest_updates
# ---------------------------------------------------------------------------

def filter_latest_updates(
    updates: list[dict[str, Any]], max_age_hours: int = 48
) -> dict[str, Any]:
    """Filter and sort a list of rumor/update dicts by freshness.

    Each item in ``updates`` should contain either:
      - an ``age_hours`` numeric field (hours since the update was reported), or
      - a ``timestamp`` ISO-8601 string field, from which age is computed
        relative to now (UTC).

    Items missing both fields are skipped and reported separately rather
    than causing the whole call to fail.

    Args:
        updates: List of update/rumor dicts.
        max_age_hours: Maximum age (in hours) for an update to be kept.

    Returns:
        A dict with the filtered, newest-first list of updates, a count,
        and any skipped/malformed entries.
    """
    if not isinstance(updates, list):
        return {"success": False, "error": "updates must be a list of dicts."}

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for item in updates:
        if not isinstance(item, dict):
            skipped.append({"item": item, "reason": "not a dict"})
            continue

        age_hours = item.get("age_hours")
        if age_hours is None and item.get("timestamp"):
            try:
                ts = datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_hours = (now - ts).total_seconds() / 3600.0
            except ValueError:
                skipped.append({"item": item, "reason": "unparseable timestamp"})
                continue

        if age_hours is None:
            skipped.append({"item": item, "reason": "missing age_hours/timestamp"})
            continue

        if age_hours < 0:
            skipped.append({"item": item, "reason": "negative age_hours"})
            continue

        if age_hours <= max_age_hours:
            enriched = dict(item)
            enriched["age_hours"] = round(float(age_hours), 2)
            kept.append(enriched)

    kept.sort(key=lambda x: x["age_hours"])

    return {
        "success": True,
        "max_age_hours": max_age_hours,
        "count": len(kept),
        "updates": kept,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Tool 3: check_squad_registration
# ---------------------------------------------------------------------------

def check_squad_registration(
    club_name: str, is_foreign: bool, birth_year: int
) -> dict[str, Any]:
    """Validate whether a new signing can be registered under RSL quota rules.

    Rules applied (2026 season, see dossier Section 1):
      - Max 8 senior foreign players per club.
      - Max 2 additional U21 foreign players (born >= 2005) per club, on top
        of the 8 senior slots.
      - Saudi national players are not subject to a foreign-quota limit.

    Args:
        club_name: RSL club name (case-insensitive), must exist in
            data/squads.json.
        is_foreign: Whether the incoming player is a foreign national.
        birth_year: Player's birth year, used to determine U21 eligibility.

    Returns:
        A dict describing current quota usage, whether registration is
        currently possible, which slot type would be used, and remaining
        capacity.
    """
    try:
        squads = _load_squads()
    except (OSError, json.JSONDecodeError) as exc:
        return {"success": False, "error": f"Could not load squads.json: {exc}"}

    resolved = _resolve_club(club_name, squads)
    if resolved is None:
        return {
            "success": False,
            "error": f"Unknown club '{club_name}'. Known clubs: {sorted(squads)}",
        }

    squad = squads[resolved]
    senior_foreign = squad.get("senior_foreign_players", 0)
    u21_foreign = squad.get("u21_foreign_players", 0)
    senior_slots_remaining = MAX_SENIOR_FOREIGN_PLAYERS - senior_foreign
    u21_slots_remaining = MAX_U21_FOREIGN_PLAYERS - u21_foreign

    result: dict[str, Any] = {
        "success": True,
        "club": resolved,
        "current_squad": squad,
        "is_foreign": is_foreign,
        "birth_year": birth_year,
        "senior_foreign_slots_remaining": senior_slots_remaining,
        "u21_foreign_slots_remaining": u21_slots_remaining,
        "already_over_senior_quota": senior_foreign > MAX_SENIOR_FOREIGN_PLAYERS,
        "already_over_u21_quota": u21_foreign > MAX_U21_FOREIGN_PLAYERS,
    }

    if not is_foreign:
        result["eligible"] = True
        result["slot_used"] = "domestic (no foreign quota applies)"
        result["reason"] = "Saudi national players are not subject to the foreign player quota."
        return result

    is_u21_eligible = birth_year >= U21_BIRTH_YEAR_CUTOFF
    result["is_u21_eligible_by_birth_year"] = is_u21_eligible

    if is_u21_eligible and u21_slots_remaining > 0:
        result["eligible"] = True
        result["slot_used"] = "U21 foreign development slot"
        result["reason"] = (
            f"Player born {birth_year} qualifies for a U21 foreign slot "
            f"({u21_slots_remaining} of {MAX_U21_FOREIGN_PLAYERS} remaining)."
        )
    elif senior_slots_remaining > 0:
        result["eligible"] = True
        result["slot_used"] = "senior foreign slot"
        if is_u21_eligible:
            result["reason"] = (
                "U21 slots are full; player registered against a senior foreign slot "
                f"instead ({senior_slots_remaining} of {MAX_SENIOR_FOREIGN_PLAYERS} remaining)."
            )
        else:
            result["reason"] = (
                f"Player born {birth_year} is not U21-eligible; registered against a "
                f"senior foreign slot ({senior_slots_remaining} of "
                f"{MAX_SENIOR_FOREIGN_PLAYERS} remaining)."
            )
    else:
        result["eligible"] = False
        result["slot_used"] = None
        result["reason"] = (
            f"No available slots: senior foreign quota "
            f"({senior_foreign}/{MAX_SENIOR_FOREIGN_PLAYERS}) and U21 foreign quota "
            f"({u21_foreign}/{MAX_U21_FOREIGN_PLAYERS}) are both exhausted."
        )

    return result


# ---------------------------------------------------------------------------
# Tool 4: filter_under21_foreign_slot
# ---------------------------------------------------------------------------

def filter_under21_foreign_slot(birth_year: int) -> dict[str, Any]:
    """Determine whether a player is eligible for a U21 foreign development slot.

    Per RSL rules, a foreign player must be born on or after 2005 (i.e. aged
    21 or under for the 2026 season) to qualify for one of the two dedicated
    U21 foreign slots.

    Args:
        birth_year: The player's birth year.

    Returns:
        A dict with eligibility, approximate age for the 2026 season, and
        the cutoff year used for the determination.
    """
    if not isinstance(birth_year, int) or birth_year < 1900 or birth_year > 2026:
        return {
            "success": False,
            "error": f"Invalid birth_year={birth_year!r}.",
        }

    season_reference_year = 2026
    approximate_age = season_reference_year - birth_year
    eligible = birth_year >= U21_BIRTH_YEAR_CUTOFF

    return {
        "success": True,
        "birth_year": birth_year,
        "u21_birth_year_cutoff": U21_BIRTH_YEAR_CUTOFF,
        "approximate_age_2026_season": approximate_age,
        "eligible_for_u21_foreign_slot": eligible,
        "reason": (
            f"Born {birth_year} >= cutoff {U21_BIRTH_YEAR_CUTOFF}: eligible for one of "
            f"the {MAX_U21_FOREIGN_PLAYERS} U21 foreign development slots."
            if eligible
            else f"Born {birth_year} < cutoff {U21_BIRTH_YEAR_CUTOFF}: must be registered "
            "against a senior foreign slot instead."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 5: check_homegrown_player_ratio
# ---------------------------------------------------------------------------

def check_homegrown_player_ratio(club_name: str) -> dict[str, Any]:
    """Check whether a club meets the RSL 50% homegrown (Saudi) player guideline.

    Args:
        club_name: RSL club name (case-insensitive), must exist in
            data/squads.json.

    Returns:
        A dict with the Saudi/foreign player counts, the computed ratio,
        and whether the club meets the minimum 50% guideline.
    """
    try:
        squads = _load_squads()
    except (OSError, json.JSONDecodeError) as exc:
        return {"success": False, "error": f"Could not load squads.json: {exc}"}

    resolved = _resolve_club(club_name, squads)
    if resolved is None:
        return {
            "success": False,
            "error": f"Unknown club '{club_name}'. Known clubs: {sorted(squads)}",
        }

    squad = squads[resolved]
    saudi = squad.get("total_saudi_players", 0)
    foreign_total = squad.get("senior_foreign_players", 0) + squad.get("u21_foreign_players", 0)
    total = saudi + foreign_total

    if total == 0:
        return {"success": False, "error": f"Club '{resolved}' has an empty registered squad."}

    ratio = saudi / total
    meets_requirement = ratio >= HOMEGROWN_MIN_RATIO

    return {
        "success": True,
        "club": resolved,
        "saudi_players": saudi,
        "foreign_players_total": foreign_total,
        "total_registered_players": total,
        "homegrown_ratio": round(ratio, 4),
        "homegrown_ratio_pct": round(ratio * 100, 1),
        "minimum_required_ratio_pct": HOMEGROWN_MIN_RATIO * 100,
        "meets_requirement": meets_requirement,
        "reason": (
            f"{resolved} squad is {round(ratio * 100, 1)}% Saudi nationals "
            f"({saudi}/{total}), which "
            + ("meets" if meets_requirement else "falls below")
            + f" the {HOMEGROWN_MIN_RATIO * 100:.0f}% guideline."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 6: currency_converter_saudi_riyal
# ---------------------------------------------------------------------------

def currency_converter_saudi_riyal(amount: float, from_currency: str) -> dict[str, Any]:
    """Convert a EUR/GBP/USD amount to Saudi Riyal (SAR) using RSL reference rates.

    Reference rates (per dossier Section 2): 1 EUR = 4.08 SAR,
    1 GBP = 4.78 SAR, 1 USD = 3.75 SAR.

    Args:
        amount: The amount in the source currency. Must be non-negative.
        from_currency: One of "EUR", "GBP", "USD" (case-insensitive).

    Returns:
        A dict with the converted SAR amount (and a millions-formatted
        convenience field) plus the exchange rate used.
    """
    if not isinstance(amount, (int, float)) or amount < 0:
        return {"success": False, "error": f"Invalid amount={amount!r}. Must be a non-negative number."}

    currency = from_currency.strip().upper()
    if currency not in CURRENCY_RATES_TO_SAR:
        return {
            "success": False,
            "error": f"Unsupported currency '{from_currency}'. Supported: {sorted(CURRENCY_RATES_TO_SAR)}",
        }

    rate = CURRENCY_RATES_TO_SAR[currency]
    converted_sar = amount * rate

    return {
        "success": True,
        "input_amount": amount,
        "from_currency": currency,
        "exchange_rate_to_sar": rate,
        "converted_amount_sar": round(converted_sar, 2),
        "converted_amount_sar_millions": round(converted_sar / 1_000_000, 3),
    }


# ---------------------------------------------------------------------------
# Tool 7: compare_player_stats
# ---------------------------------------------------------------------------

def compare_player_stats(
    target_player_metrics: dict[str, Any], current_player_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Compute a side-by-side statistical comparison between two players.

    Expected metric keys in each input dict (missing keys default to 0):
      - "goals", "assists", "key_passes_per_90", "minutes_played"

    "minutes_per_goal" is derived as minutes_played / goals (None if the
    player has scored 0 goals, to avoid a division by zero).

    Args:
        target_player_metrics: Metrics dict for the prospective signing.
        current_player_metrics: Metrics dict for the existing squad player.

    Returns:
        A dict with per-metric values for both players, the delta
        (target - current), and a plain-language summary.
    """
    if not isinstance(target_player_metrics, dict) or not isinstance(current_player_metrics, dict):
        return {"success": False, "error": "Both metrics arguments must be dicts."}

    def _derive(metrics: dict[str, Any]) -> dict[str, Any]:
        goals = float(metrics.get("goals", 0) or 0)
        assists = float(metrics.get("assists", 0) or 0)
        key_passes = float(metrics.get("key_passes_per_90", metrics.get("key_passes", 0)) or 0)
        minutes_played = float(metrics.get("minutes_played", 0) or 0)
        minutes_per_goal = (minutes_played / goals) if goals > 0 else None
        return {
            "goals": goals,
            "assists": assists,
            "key_passes_per_90": key_passes,
            "minutes_played": minutes_played,
            "minutes_per_goal": round(minutes_per_goal, 1) if minutes_per_goal is not None else None,
        }

    target = _derive(target_player_metrics)
    current = _derive(current_player_metrics)

    def _delta(key: str) -> float | None:
        t, c = target[key], current[key]
        if t is None or c is None:
            return None
        return round(t - c, 2)

    delta = {
        "goals": _delta("goals"),
        "assists": _delta("assists"),
        "key_passes_per_90": _delta("key_passes_per_90"),
        "minutes_per_goal": _delta("minutes_per_goal"),
    }

    notes = []
    if delta["goals"] is not None:
        notes.append(
            f"Target scores {abs(delta['goals']):g} {'more' if delta['goals'] >= 0 else 'fewer'} goals."
        )
    if delta["minutes_per_goal"] is not None:
        better = delta["minutes_per_goal"] < 0
        notes.append(
            f"Target needs {abs(delta['minutes_per_goal']):g} "
            f"{'fewer' if better else 'more'} minutes per goal "
            f"({'more' if better else 'less'} clinical)."
        )
    if delta["assists"] is not None:
        notes.append(
            f"Target records {abs(delta['assists']):g} {'more' if delta['assists'] >= 0 else 'fewer'} assists."
        )
    if delta["key_passes_per_90"] is not None:
        notes.append(
            f"Target creates {abs(delta['key_passes_per_90']):g} "
            f"{'more' if delta['key_passes_per_90'] >= 0 else 'fewer'} key passes per 90."
        )

    return {
        "success": True,
        "target_player": target,
        "current_player": current,
        "delta_target_minus_current": delta,
        "summary": " ".join(notes) if notes else "Insufficient data to compare.",
    }


TOOL_REGISTRY: dict[str, Any] = {
    "calculate_transfer_score": calculate_transfer_score,
    "filter_latest_updates": filter_latest_updates,
    "check_squad_registration": check_squad_registration,
    "filter_under21_foreign_slot": filter_under21_foreign_slot,
    "check_homegrown_player_ratio": check_homegrown_player_ratio,
    "currency_converter_saudi_riyal": currency_converter_saudi_riyal,
    "compare_player_stats": compare_player_stats,
}
