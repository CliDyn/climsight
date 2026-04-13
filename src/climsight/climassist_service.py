"""Lightweight advisory service for the deployable ClimAssist MVP.

This module intentionally avoids the heavy ClimSight runtime:
- no local climate bundles
- no vector store
- no sandboxed Python execution

It uses free forecast data from Open-Meteo and can optionally upgrade the
response quality with an OpenAI-compatible model such as OpenRouter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any

import requests
from geopy.geocoders import Nominatim
from openai import OpenAI


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemma-3-4b-it"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


@dataclass
class AdvisoryRequest:
    lat: float
    lon: float
    crop: str
    stage: str
    question: str
    language: str
    model_name: str | None = None
    api_key: str | None = None


def generate_advisory(request: AdvisoryRequest) -> dict[str, Any]:
    """Generate a lightweight farmer-facing advisory."""
    location_label = _resolve_location_label(request.lat, request.lon)
    forecast = _fetch_forecast(request.lat, request.lon)
    evidence = _build_evidence(location_label, forecast)
    report, provider = _build_report(request, evidence)
    return {
        "provider": provider,
        "location_label": location_label,
        "report": report,
        "evidence": evidence,
        "request": {
            "crop": request.crop,
            "stage": request.stage,
            "language": request.language,
            "question": request.question,
            "lat": request.lat,
            "lon": request.lon,
        },
    }


def get_runtime_status() -> dict[str, Any]:
    """Expose whether the service can use an LLM or will stay heuristic only."""
    return {
        "openrouter_ready": bool(os.getenv("OPENROUTER_API_KEY")),
        "openai_ready": bool(os.getenv("OPENAI_API_KEY")),
        "default_provider": _default_provider_name(),
    }


def _resolve_location_label(lat: float, lon: float) -> str:
    """Best-effort reverse geocoding without making it critical to the flow."""
    try:
        geolocator = Nominatim(user_agent="climassist-mvp")
        location = geolocator.reverse((lat, lon), language="en", zoom=10, timeout=5)
        if location and location.address:
            parts = [part.strip() for part in location.address.split(",") if part.strip()]
            return ", ".join(parts[:3])
    except Exception:
        pass
    return f"{lat:.3f}, {lon:.3f}"


def _fetch_forecast(lat: float, lon: float) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
            ]
        ),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_hours",
                "wind_speed_10m_max",
            ]
        ),
        "forecast_days": 7,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _build_evidence(location_label: str, forecast: dict[str, Any]) -> dict[str, Any]:
    current = forecast.get("current", {})
    daily = forecast.get("daily", {})

    days = []
    for index, day in enumerate(daily.get("time", [])):
        days.append(
            {
                "date": day,
                "temp_max_c": _safe_round(_list_value(daily, "temperature_2m_max", index)),
                "temp_min_c": _safe_round(_list_value(daily, "temperature_2m_min", index)),
                "precipitation_mm": _safe_round(_list_value(daily, "precipitation_sum", index)),
                "precipitation_hours": _safe_round(
                    _list_value(daily, "precipitation_hours", index)
                ),
                "wind_max_kph": _safe_round(_list_value(daily, "wind_speed_10m_max", index)),
            }
        )

    total_rain = round(sum(day["precipitation_mm"] for day in days), 1) if days else 0.0
    hottest_day = max((day["temp_max_c"] for day in days), default=None)
    coolest_night = min((day["temp_min_c"] for day in days), default=None)
    hottest_day_count = sum(1 for day in days if day["temp_max_c"] is not None and day["temp_max_c"] >= 35)
    heavy_rain_days = [day["date"] for day in days if day["precipitation_mm"] is not None and day["precipitation_mm"] >= 20]
    windy_days = [day["date"] for day in days if day["wind_max_kph"] is not None and day["wind_max_kph"] >= 30]

    risks = []
    if total_rain < 10:
        risks.append("Dry week signal: low rainfall expected over the next 7 days.")
    if hottest_day_count >= 2:
        risks.append("Heat stress risk: multiple days at or above 35C.")
    if heavy_rain_days:
        risks.append(f"Heavy rain risk: watch runoff or waterlogging on {', '.join(heavy_rain_days[:3])}.")
    if windy_days:
        risks.append(f"Wind risk: strong winds are forecast on {', '.join(windy_days[:3])}.")
    if not risks:
        risks.append("No major short-term weather shock stands out from the 7-day forecast.")

    return {
        "location_label": location_label,
        "timezone": forecast.get("timezone"),
        "current": {
            "temperature_c": _safe_round(current.get("temperature_2m")),
            "relative_humidity_pct": _safe_round(current.get("relative_humidity_2m")),
            "precipitation_mm": _safe_round(current.get("precipitation")),
            "wind_speed_kph": _safe_round(current.get("wind_speed_10m")),
        },
        "summary": {
            "total_precipitation_mm": total_rain,
            "hottest_day_c": hottest_day,
            "coolest_night_c": coolest_night,
        },
        "daily": days,
        "risk_flags": risks,
        "sources": [
            "Open-Meteo 7-day forecast",
            "User-provided crop and question context",
        ],
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _build_report(request: AdvisoryRequest, evidence: dict[str, Any]) -> tuple[str, str]:
    llm_config = _resolve_llm_config(request)
    if llm_config is None:
        report = _build_heuristic_report(request, evidence)
        return report, "heuristic"

    try:
        report = _build_llm_report(request, evidence, llm_config)
        return report, llm_config["provider"]
    except Exception:
        report = _build_heuristic_report(request, evidence)
        return report, "heuristic"


def _resolve_llm_config(request: AdvisoryRequest) -> dict[str, str] | None:
    api_key = request.api_key or os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return {
            "provider": "openrouter",
            "api_key": api_key,
            "base_url": os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            "model_name": request.model_name or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        }

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return {
            "provider": "openai",
            "api_key": openai_api_key,
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "model_name": request.model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        }

    return None


def _build_llm_report(
    request: AdvisoryRequest,
    evidence: dict[str, Any],
    llm_config: dict[str, str],
) -> str:
    client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["base_url"])

    system_prompt = (
        "You are ClimAssist, a careful climate advisory assistant for smallholder farmers. "
        "Use only the evidence provided. Do not invent seasonal trends, soil facts, or long-term "
        "climate projections that are not in the evidence. Keep the answer practical, grounded, and "
        "explicit about uncertainty. Use markdown with these sections: "
        "## Snapshot, ## What to do now, ## Risks to watch, ## Confidence and limits, ## Sources."
    )

    user_prompt = (
        f"Language: {request.language}\n"
        f"Crop: {request.crop}\n"
        f"Farming stage: {request.stage}\n"
        f"User question: {request.question or 'Give practical advice for the next 7 days.'}\n\n"
        f"Evidence JSON:\n{evidence}\n\n"
        "Additional rules:\n"
        "- If the requested language is Hausa, answer in Hausa.\n"
        "- Keep the sources section short and factual.\n"
        "- Mention that this MVP is grounded mainly in short-term forecast data.\n"
        "- If evidence is insufficient for a claim, say so clearly.\n"
    )

    response = client.chat.completions.create(
        model=llm_config["model_name"],
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise ValueError("Empty model response")
    return content


def _build_heuristic_report(request: AdvisoryRequest, evidence: dict[str, Any]) -> str:
    summary = evidence["summary"]
    current = evidence["current"]
    risks = evidence["risk_flags"]

    actions = _suggest_actions(request.crop, request.stage, summary, risks)
    limitation = (
        "This preview is running in deterministic mode using short-term forecast data only. "
        "It does not yet include full seasonal modeling, local soil measurements, or extension-officer review."
    )
    if request.language.lower() == "hausa":
        limitation = (
            "An nemi amsa a Hausa, amma wannan yanayin na fallback yana bada amsa cikin Turanci. "
            + limitation
        )

    bullet_actions = "\n".join(f"- {action}" for action in actions)
    bullet_risks = "\n".join(f"- {risk}" for risk in risks)
    bullet_sources = "\n".join(f"- {source}" for source in evidence["sources"])

    return (
        f"## Snapshot\n"
        f"- Location: **{evidence['location_label']}**\n"
        f"- Crop: **{request.crop.title()}**\n"
        f"- Stage: **{request.stage.replace('-', ' ').title()}**\n"
        f"- Current temperature: **{_render_metric(current['temperature_c'], 'C')}**\n"
        f"- Current wind: **{_render_metric(current['wind_speed_kph'], ' km/h')}**\n"
        f"- Rain expected over next 7 days: **{_render_metric(summary['total_precipitation_mm'], ' mm')}**\n"
        f"- Hottest day in forecast: **{_render_metric(summary['hottest_day_c'], 'C')}**\n\n"
        f"## What to do now\n"
        f"{bullet_actions}\n\n"
        f"## Risks to watch\n"
        f"{bullet_risks}\n\n"
        f"## Confidence and limits\n"
        f"- {limitation}\n"
        f"- Good for short-term field decisions. Not enough on its own for long-season planning.\n\n"
        f"## Sources\n"
        f"{bullet_sources}\n"
    )


def _suggest_actions(
    crop: str,
    stage: str,
    summary: dict[str, Any],
    risks: list[str],
) -> list[str]:
    stage_key = stage.lower()
    crop_label = crop.lower()
    actions = []

    if "dry week" in " ".join(risks).lower():
        if stage_key in {"pre-planting", "planting"}:
            actions.append(
                f"If possible, avoid committing all {crop_label} seed at once until follow-up rainfall is clearer."
            )
        actions.append("Preserve soil moisture: reduce unnecessary disturbance and keep mulch or residue where available.")

    if any("heat stress" in risk.lower() for risk in risks):
        actions.append("Plan field work early in the morning and check young plants for heat stress.")
        if stage_key in {"flowering", "grain-fill", "fruiting"}:
            actions.append("Monitor flowering or grain-fill closely, because hot days can reduce final yield.")

    if any("heavy rain" in risk.lower() for risk in risks):
        actions.append("Clear drainage paths now and avoid fertilizer application immediately before heavy rain.")

    if any("wind risk" in risk.lower() for risk in risks):
        actions.append("Protect lightweight coverings and watch for lodging risk in taller crops.")

    if not actions:
        actions.append("Use the next 7 days for routine field checks and keep monitoring forecast updates.")
        actions.append("If planting, use a staggered approach instead of putting all seed in on one day.")

    if stage_key == "harvest":
        actions.append("Prioritize drying and safe storage if rainfall windows look short.")

    return actions[:4]


def _default_provider_name() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "heuristic"


def _list_value(container: dict[str, Any], key: str, index: int) -> float | None:
    values = container.get(key, [])
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _safe_round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _render_metric(value: float | None, suffix: str) -> str:
    if value is None:
        return f"n/a{suffix}"
    return f"{value}{suffix}"
