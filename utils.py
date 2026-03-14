"""
utils.py
========
Shared helper functions used by both train_model.py and app.py.

Keeps business logic (feature engineering, interpretation rules,
colour mapping) in one place so app.py stays clean.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ── Feature engineering (mirrors train_model.py exactly) ─────────────────────

def engineer_features_for_inference(input_dict: dict) -> pd.DataFrame:
    """
    Convert the raw input dictionary collected from the Streamlit sidebar
    into a single-row DataFrame with all engineered features.

    Parameters
    ----------
    input_dict : dict
        Keys must match the raw column names captured in app.py.

    Returns
    -------
    pd.DataFrame  – one row, ready to be passed to pipeline.predict()
    """
    row = input_dict.copy()

    # Date features (use harvest_month / harvest_quarter directly;
    # the app captures them as integers so no parsing needed)
    # These are already in input_dict as "harvest_month" and "harvest_quarter".

    # Interaction / engineered features
    row["sugar_ripeness_ratio"] = (
        row["sugar_content_brix"] / (row["ripeness_index"] + 1e-6)
    )
    row["size_index"] = row["weight_g"] / (row["length_cm"] + 1e-6)

    return pd.DataFrame([row])


# ── Quality tier metadata ────────────────────────────────────────────────────

# Maps each quality_category to a display colour (Streamlit-compatible),
# a short emoji, and a business label used in the recommendation panel.
QUALITY_META: dict[str, dict] = {
    "Premium": {
        "colour":      "#2e7d32",   # deep green
        "bg_colour":   "#e8f5e9",
        "emoji":       "🌟",
        "business_label": "Premium",
        "retail_route": "Premium retail & export channels",
        "action":       "Fast-track to premium packing line. Prioritise same-day dispatch.",
        "price_band":   "Top-tier pricing applicable",
        "alert_type":   "success",
    },
    "Good": {
        "colour":      "#1565c0",   # deep blue
        "bg_colour":   "#e3f2fd",
        "emoji":       "✅",
        "business_label": "Good",
        "retail_route": "Standard retail & supermarket supply",
        "action":       "Route to standard retail packing. 3–5 day shelf window.",
        "price_band":   "Standard retail pricing",
        "alert_type":   "info",
    },
    "Processing": {
        "colour":      "#e65100",   # deep orange
        "bg_colour":   "#fff3e0",
        "emoji":       "⚠️",
        "business_label": "Processing",
        "retail_route": "Industrial processing (smoothies, dried fruit, baby food)",
        "action":       "Redirect to processing facility. Do not send to fresh retail.",
        "price_band":   "Discounted — processing grade",
        "alert_type":   "warning",
    },
    "Unripe": {
        "colour":      "#6a1b9a",   # deep purple
        "bg_colour":   "#f3e5f5",
        "emoji":       "🔵",
        "business_label": "Unripe",
        "retail_route": "Ripening room or extended storage",
        "action":       "Transfer to controlled ripening room (18–20 °C, 90% RH). Re-inspect in 48 h.",
        "price_band":   "Hold — no sale until re-graded",
        "alert_type":   "warning",
    },
}


def get_quality_meta(quality_class: str) -> dict:
    """Return metadata for a predicted quality class, with a safe fallback."""
    return QUALITY_META.get(
        quality_class,
        {
            "colour":      "#555555",
            "bg_colour":   "#f5f5f5",
            "emoji":       "❓",
            "business_label": quality_class,
            "retail_route": "Unknown",
            "action":       "Manual inspection required.",
            "price_band":   "To be determined",
            "alert_type":   "info",
        },
    )


# ── Confidence helpers ───────────────────────────────────────────────────────

def confidence_label(prob: float) -> str:
    """Convert a probability to a human-readable confidence label."""
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.75:
        return "High"
    elif prob >= 0.55:
        return "Moderate"
    else:
        return "Low"


def confidence_colour(prob: float) -> str:
    """Return a hex colour that tracks confidence level."""
    if prob >= 0.90:
        return "#2e7d32"
    elif prob >= 0.75:
        return "#1565c0"
    elif prob >= 0.55:
        return "#e65100"
    else:
        return "#b71c1c"


# ── Ripeness helpers ─────────────────────────────────────────────────────────

def ripeness_index_to_category(ripeness_index: float) -> str:
    """
    Derive a human-readable ripeness stage from the continuous index.
    Mirrors the categorical encoding in the original dataset.
    """
    if ripeness_index < 2.5:
        return "Green"
    elif ripeness_index < 4.5:
        return "Turning"
    elif ripeness_index < 6.0:
        return "Ripe"
    else:
        return "Overripe"


# ── Input validation ─────────────────────────────────────────────────────────

def validate_inputs(inputs: dict) -> list[str]:
    """
    Return a list of warning strings for out-of-range or suspicious values.
    An empty list means all inputs look fine.
    """
    warnings: list[str] = []

    if inputs["ripeness_index"] < 1.5:
        warnings.append("Ripeness index is very low — banana may be too green to grade accurately.")

    if inputs["sugar_content_brix"] < 15.5:
        warnings.append("Sugar content is below the typical minimum — double-check the Brix reading.")

    if inputs["firmness_kgf"] > 4.8:
        warnings.append("Firmness is at the upper limit — confirm the kgf reading is correct.")

    weight_to_length = inputs["weight_g"] / max(inputs["length_cm"], 1)
    if weight_to_length < 4.0 or weight_to_length > 12.0:
        warnings.append(
            f"Weight-to-length ratio ({weight_to_length:.1f} g/cm) is unusual — "
            "verify the measurements."
        )

    return warnings


# ── Feature display names ────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "ripeness_index":      "Ripeness Index",
    "sugar_content_brix":  "Sugar Content (Brix)",
    "firmness_kgf":        "Firmness (kgf)",
    "length_cm":           "Length (cm)",
    "weight_g":            "Weight (g)",
    "tree_age_years":      "Tree Age (years)",
    "altitude_m":          "Altitude (m)",
    "rainfall_mm":         "Annual Rainfall (mm)",
    "soil_nitrogen_ppm":   "Soil Nitrogen (ppm)",
    "harvest_month":       "Harvest Month",
    "harvest_quarter":     "Harvest Quarter",
    "variety":             "Variety",
    "region":              "Region",
    "ripeness_category":   "Ripeness Stage",
    "sugar_ripeness_ratio":"Sugar / Ripeness Ratio",
    "size_index":          "Size Index (g/cm)",
}
