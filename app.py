"""
app.py
======
Banana Quality Inspector — Streamlit application.

Launch with:
    streamlit run app.py

Requires the trained pipeline to exist at  model/banana_pipeline.joblib
Run  python train_model.py  once before starting the app.
"""

from __future__ import annotations

import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    engineer_features_for_inference,
    get_quality_meta,
    confidence_label,
    confidence_colour,
    ripeness_index_to_category,
    validate_inputs,
    FEATURE_DISPLAY_NAMES,
)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Banana Quality Inspector",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global typography ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #fffdf7;
}

/* ── Top header banner ── */
.bqi-header {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    border-bottom: 3px solid #f5c518;
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 0 0 16px 16px;
    margin-bottom: 1.5rem;
}
.bqi-header h1 {
    font-family: 'DM Serif Display', serif;
    color: #f5c518;
    font-size: 2.4rem;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.bqi-header p {
    color: #c8c8c8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.bqi-header .badge {
    display: inline-block;
    background: #f5c518;
    color: #1a1a1a;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 0.7rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.4rem;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    border: 1.5px solid;
}

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-right: 6px;
    margin-top: 4px;
}

/* ── Recommendation block ── */
.rec-block {
    background: #1a1a1a;
    color: #f0f0f0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.9rem;
    line-height: 1.7;
}
.rec-block .rec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #f5c518;
    margin-bottom: 0.8rem;
}
.rec-block .rec-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
    align-items: flex-start;
}
.rec-block .rec-key {
    color: #888;
    min-width: 110px;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding-top: 2px;
}
.rec-block .rec-val {
    color: #f0f0f0;
    font-weight: 400;
}

/* ── History table ── */
.history-table th {
    background: #1a1a1a !important;
    color: #f5c518 !important;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: #1e1e1e;
}
section[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stNumberInput > label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #aaa !important;
    letter-spacing: 0.3px;
}
section[data-testid="stSidebar"] hr {
    border-color: #333 !important;
}
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #f5c518 !important;
    margin: 1rem 0 0.4rem;
}

/* ── Divider ── */
.yellow-divider {
    height: 3px;
    background: linear-gradient(90deg, #f5c518, transparent);
    border-radius: 2px;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join("model", "banana_pipeline.joblib")

@st.cache_resource(show_spinner="Loading quality model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Auto-train if model doesn't exist (needed for cloud deployment)
        import train_model as tm
        df_raw       = tm.load_data(tm.DATA_PATH)
        df_processed = tm.engineer_features(df_raw)
        pipeline, num_feats, cat_feats = tm.train_and_evaluate(df_processed)
        tm.save_pipeline(pipeline, num_feats, cat_feats)
    return joblib.load(MODEL_PATH)


Save the file, then go back to GitHub, click on `app.py` → click the **pencil icon** to edit → paste the updated function → click **Commit changes**.

---

## Step 4 — Deploy on Streamlit Cloud

1. Go to **share.streamlit.io**
2. Click **Sign in with GitHub** and authorize it
3. Click **New app**
4. Fill in:
   - **Repository:** `yourname/banana-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**

It will take about 2–3 minutes to install packages and launch. When it's done you'll see a green **Your app is live** message.

---

## Step 5 — Share the link

You'll get a link like:

https://yourname-banana-app.streamlit.app

artefact = load_model()
pipeline             = artefact["pipeline"]
numeric_features     = artefact["numeric_features"]
categorical_features = artefact["categorical_features"]
classes              = artefact["classes"]   # e.g. ['Good','Premium','Processing','Unripe']


# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="bqi-header">
  <div class="badge">Internal Tool · Quality Operations</div>
  <h1>🍌 Banana Quality Inspector</h1>
  <p>
    AI-powered batch grading for quality inspectors &nbsp;|&nbsp;
    Gradient Boosting classifier &nbsp;·&nbsp; 4-class multiclass &nbsp;·&nbsp;
    ~91 % cross-validated accuracy
  </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Inspection inputs ───────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">🍌 Inspection Inputs</div>',
                unsafe_allow_html=True)
    st.markdown("Fill in the measurements for the banana batch you are grading.")
    st.markdown("---")

    # ── Maturity & Sensory ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Maturity & Sensory</div>',
                unsafe_allow_html=True)

    ripeness_index = st.slider(
        "Ripeness Index  (1 = green → 7 = overripe)",
        min_value=1.0, max_value=7.0, value=4.1, step=0.05,
    )

    # Derive the ripeness category label automatically
    ripe_cat = ripeness_index_to_category(ripeness_index)
    st.caption(f"📍 Inferred ripeness stage: **{ripe_cat}**")

    sugar_content_brix = st.slider(
        "Sugar Content (Brix °)",
        min_value=15.0, max_value=22.0, value=18.5, step=0.1,
    )

    firmness_kgf = st.slider(
        "Firmness (kgf)",
        min_value=0.5, max_value=5.0, value=2.7, step=0.05,
    )

    st.markdown("---")

    # ── Physical dimensions ────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Physical Dimensions</div>',
                unsafe_allow_html=True)

    length_cm = st.slider(
        "Length (cm)",
        min_value=10.0, max_value=30.0, value=19.9, step=0.1,
    )

    weight_g = st.slider(
        "Weight (g)",
        min_value=80.0, max_value=250.0, value=164.0, step=1.0,
    )

    st.markdown("---")

    # ── Provenance ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Provenance</div>',
                unsafe_allow_html=True)

    variety = st.selectbox(
        "Variety",
        options=["Cavendish", "Plantain", "Burro", "Manzano",
                 "Lady Finger", "Red Dacca", "Blue Java", "Fehi"],
    )

    region = st.selectbox(
        "Region of Origin",
        options=["Colombia", "Ecuador", "Guatemala", "Costa Rica",
                 "Brazil", "Honduras", "India", "Philippines"],
    )

    st.markdown("---")

    # ── Agronomic conditions ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Agronomic Conditions</div>',
                unsafe_allow_html=True)

    tree_age_years = st.slider(
        "Tree Age (years)",
        min_value=2.0, max_value=20.0, value=10.0, step=0.5,
    )

    altitude_m = st.slider(
        "Altitude (m)",
        min_value=0.0, max_value=1500.0, value=700.0, step=10.0,
    )

    rainfall_mm = st.slider(
        "Annual Rainfall (mm)",
        min_value=1000.0, max_value=3000.0, value=1973.0, step=10.0,
    )

    soil_nitrogen_ppm = st.slider(
        "Soil Nitrogen (ppm)",
        min_value=10.0, max_value=200.0, value=104.0, step=1.0,
    )

    st.markdown("---")

    # ── Harvest date ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Harvest Info</div>',
                unsafe_allow_html=True)

    harvest_month = st.selectbox(
        "Harvest Month",
        options=list(range(1, 13)),
        index=9,   # October default
        format_func=lambda m: datetime.date(2000, m, 1).strftime("%B"),
    )
    harvest_quarter = (harvest_month - 1) // 3 + 1

    st.markdown("---")
    predict_btn = st.button("🔍  Run Quality Prediction", use_container_width=True)


# ── Build input dict & run prediction ─────────────────────────────────────────

raw_inputs = {
    "ripeness_index":      ripeness_index,
    "ripeness_category":   ripe_cat,
    "sugar_content_brix":  sugar_content_brix,
    "firmness_kgf":        firmness_kgf,
    "length_cm":           length_cm,
    "weight_g":            weight_g,
    "variety":             variety,
    "region":              region,
    "tree_age_years":      tree_age_years,
    "altitude_m":          altitude_m,
    "rainfall_mm":         rainfall_mm,
    "soil_nitrogen_ppm":   soil_nitrogen_ppm,
    "harvest_month":       harvest_month,
    "harvest_quarter":     harvest_quarter,
}

input_df = engineer_features_for_inference(raw_inputs)

# Align columns to what the pipeline expects
expected_cols = numeric_features + categorical_features
input_df = input_df.reindex(columns=expected_cols)

# Run prediction
proba_array   = pipeline.predict_proba(input_df)[0]
predicted_idx = int(np.argmax(proba_array))
predicted_cls = classes[predicted_idx]
predicted_prob = float(proba_array[predicted_idx])

meta = get_quality_meta(predicted_cls)
warnings_list = validate_inputs(raw_inputs)


# ── Main content — three columns ──────────────────────────────────────────────

col_left, col_mid, col_right = st.columns([1.15, 1.15, 0.7], gap="medium")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Quality Prediction result
# ─────────────────────────────────────────────────────────────────────────────

with col_left:
    st.markdown('<div class="section-label">Quality Prediction</div>',
                unsafe_allow_html=True)

    # Validation warnings (show before result if any)
    for w in warnings_list:
        st.warning(w, icon="⚠️")

    # Result card
    st.markdown(f"""
    <div class="result-card" style="
        background:{meta['bg_colour']};
        border-color:{meta['colour']};">
      <div style="font-size:0.75rem;font-weight:600;letter-spacing:2px;
                  text-transform:uppercase;color:{meta['colour']};margin-bottom:0.4rem;">
        Predicted Grade
      </div>
      <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;
                  color:{meta['colour']};line-height:1.1;margin-bottom:0.5rem;">
        {meta['emoji']} {predicted_cls}
      </div>
      <div style="font-size:0.9rem;color:#555;font-weight:400;">
        {meta['retail_route']}
      </div>
      <div style="margin-top:0.9rem;">
        <span class="metric-pill"
              style="background:{meta['colour']}22;color:{meta['colour']};">
          Confidence: {predicted_prob*100:.1f}%
        </span>
        <span class="metric-pill"
              style="background:{confidence_colour(predicted_prob)}22;
                     color:{confidence_colour(predicted_prob)};">
          {confidence_label(predicted_prob)}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar chart for all classes
    st.markdown('<div class="section-label" style="margin-top:1rem;">Class Probabilities</div>',
                unsafe_allow_html=True)

    CLASS_COLOURS = {
        "Premium":    "#2e7d32",
        "Good":       "#1565c0",
        "Processing": "#e65100",
        "Unripe":     "#6a1b9a",
    }

    prob_df = pd.DataFrame({
        "Grade":       classes,
        "Probability": (proba_array * 100).round(1),
    }).sort_values("Probability", ascending=True)

    bar_colours = [CLASS_COLOURS.get(g, "#888") for g in prob_df["Grade"]]

    fig_bar = go.Figure(go.Bar(
        x=prob_df["Probability"],
        y=prob_df["Grade"],
        orientation="h",
        marker_color=bar_colours,
        text=[f"{v:.1f}%" for v in prob_df["Probability"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis=dict(range=[0, 110], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        font=dict(family="DM Sans", size=13),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLE COLUMN — Operational Recommendation + Input Summary
# ─────────────────────────────────────────────────────────────────────────────

with col_mid:
    st.markdown('<div class="section-label">Operational Recommendation</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rec-block">
      <div class="rec-title">📋 Dispatch Recommendation</div>
      <div class="rec-row">
        <div class="rec-key">Grade</div>
        <div class="rec-val">{predicted_cls} — {meta['business_label']}</div>
      </div>
      <div class="rec-row">
        <div class="rec-key">Action</div>
        <div class="rec-val">{meta['action']}</div>
      </div>
      <div class="rec-row">
        <div class="rec-key">Route</div>
        <div class="rec-val">{meta['retail_route']}</div>
      </div>
      <div class="rec-row">
        <div class="rec-key">Pricing</div>
        <div class="rec-val">{meta['price_band']}</div>
      </div>
      <div class="rec-row">
        <div class="rec-key">Confidence</div>
        <div class="rec-val">{predicted_prob*100:.1f}% ({confidence_label(predicted_prob)})</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence gauge
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Prediction Confidence</div>',
                unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(predicted_prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "family": "DM Sans"}},
        gauge={
            "axis":      {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar":       {"color": meta["colour"], "thickness": 0.25},
            "bgcolor":   "#f5f5f5",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   55], "color": "#fce4ec"},
                {"range": [55,  75], "color": "#fff9c4"},
                {"range": [75, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line":  {"color": meta["colour"], "width": 3},
                "thickness": 0.8,
                "value": round(predicted_prob * 100, 1),
            },
        },
    ))
    fig_gauge.update_layout(
        margin=dict(l=20, r=20, t=20, b=10),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Input summary
    st.markdown('<div class="section-label" style="margin-top:0.4rem;">Inspection Summary</div>',
                unsafe_allow_html=True)

    summary_items = [
        ("Variety",         variety),
        ("Region",          region),
        ("Ripeness Index",  f"{ripeness_index:.2f}  ({ripe_cat})"),
        ("Sugar (Brix°)",   f"{sugar_content_brix:.1f}"),
        ("Firmness (kgf)",  f"{firmness_kgf:.2f}"),
        ("Length × Weight", f"{length_cm:.1f} cm  ×  {weight_g:.0f} g"),
        ("Tree Age",        f"{tree_age_years:.1f} yr"),
        ("Altitude",        f"{altitude_m:.0f} m"),
        ("Rainfall",        f"{rainfall_mm:.0f} mm / yr"),
        ("Soil N",          f"{soil_nitrogen_ppm:.0f} ppm"),
    ]

    summary_html = "<table style='width:100%;font-size:0.83rem;border-collapse:collapse;'>"
    for label, value in summary_items:
        summary_html += f"""
        <tr>
          <td style='color:#888;padding:4px 8px 4px 0;white-space:nowrap;
                     border-bottom:0.5px solid #eee;'>{label}</td>
          <td style='color:#1a1a1a;padding:4px 0;font-weight:500;
                     border-bottom:0.5px solid #eee;'>{value}</td>
        </tr>"""
    summary_html += "</table>"
    st.markdown(summary_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Engineered feature spotlight
# ─────────────────────────────────────────────────────────────────────────────

with col_right:
    st.markdown('<div class="section-label">Key Quality Drivers</div>',
                unsafe_allow_html=True)

    sugar_ripeness_ratio = raw_inputs["sugar_content_brix"] / max(raw_inputs["ripeness_index"], 0.01)
    size_index           = raw_inputs["weight_g"] / max(raw_inputs["length_cm"], 0.01)

    # Benchmarks derived from the training dataset
    benchmarks = {
        "Ripeness Index":         (ripeness_index,       4.04, 1.0, 7.0),
        "Sugar Content (Brix°)":  (sugar_content_brix,  18.52, 15.0, 22.0),
        "Firmness (kgf)":         (firmness_kgf,         2.71, 0.5, 5.0),
        "Sugar / Ripeness Ratio": (sugar_ripeness_ratio, 4.93, 2.5, 9.0),
        "Size Index (g/cm)":      (size_index,           8.35, 3.0, 14.0),
    }

    for feat_name, (val, avg, lo, hi) in benchmarks.items():
        pct_val = int(min(100, max(0, (val - lo) / (hi - lo) * 100)))
        pct_avg = int(min(100, max(0, (avg - lo) / (hi - lo) * 100)))

        diff     = val - avg
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        diff_col = "#2e7d32" if diff >= 0 else "#c62828"

        st.markdown(f"""
        <div style="margin-bottom:1.1rem;">
          <div style="display:flex;justify-content:space-between;
                      font-size:0.78rem;margin-bottom:3px;">
            <span style="color:#555;font-weight:500;">{feat_name}</span>
            <span style="color:{diff_col};font-weight:600;">{diff_str} vs avg</span>
          </div>
          <div style="background:#eee;border-radius:4px;height:6px;position:relative;">
            <div style="background:{meta['colour']};width:{pct_val}%;
                        height:6px;border-radius:4px;"></div>
            <div style="position:absolute;top:-3px;left:{pct_avg}%;
                        width:2px;height:12px;background:#1a1a1a;
                        border-radius:1px;" title="Dataset average"></div>
          </div>
          <div style="font-size:0.75rem;color:#888;margin-top:2px;">
            {val:.2f} &nbsp;·&nbsp; avg {avg:.2f}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="yellow-divider"></div>', unsafe_allow_html=True)

    # Quick legend
    st.markdown("""
    <div style="font-size:0.75rem;color:#888;line-height:1.8;">
      <b style="color:#1a1a1a;">Bar</b> = your value<br>
      <b style="color:#1a1a1a;">|</b> = dataset average<br>
      <span style="color:#2e7d32;">+</span> = above average &nbsp;
      <span style="color:#c62828;">−</span> = below average
    </div>
    """, unsafe_allow_html=True)


# ── Save to history on button press ──────────────────────────────────────────

if predict_btn:
    record = {
        "Timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
        "Variety":    variety,
        "Region":     region,
        "Ripeness":   f"{ripeness_index:.2f}",
        "Sugar":      f"{sugar_content_brix:.1f}",
        "Firmness":   f"{firmness_kgf:.2f}",
        "Grade":      predicted_cls,
        "Confidence": f"{predicted_prob*100:.1f}%",
    }
    st.session_state.history.insert(0, record)


# ── Prediction History ────────────────────────────────────────────────────────

st.markdown('<div class="yellow-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Prediction History (this session)</div>',
            unsafe_allow_html=True)

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    # Colour the Grade column
    def colour_grade(val):
        colours = {
            "Premium":    "background-color:#e8f5e9;color:#2e7d32;font-weight:600",
            "Good":       "background-color:#e3f2fd;color:#1565c0;font-weight:600",
            "Processing": "background-color:#fff3e0;color:#e65100;font-weight:600",
            "Unripe":     "background-color:#f3e5f5;color:#6a1b9a;font-weight:600",
        }
        return colours.get(val, "")

    styled = (
        history_df.style
        .applymap(colour_grade, subset=["Grade"])
        .set_properties(**{
            "font-size": "0.83rem",
            "text-align": "left",
        })
        .hide(axis="index")
    )

    st.dataframe(styled, use_container_width=True, height=220)

    col_dl, col_clear, _ = st.columns([1, 1, 4])
    with col_dl:
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Export CSV",
            data=csv_data,
            file_name="banana_inspection_log.csv",
            mime="text/csv",
        )
    with col_clear:
        if st.button("🗑️  Clear History"):
            st.session_state.history = []
            st.rerun()
else:
    st.info(
        "No predictions recorded yet. "
        "Adjust the sliders and press **Run Quality Prediction** to begin.",
        icon="📋",
    )


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #ddd;
            font-size:0.78rem;color:#aaa;text-align:center;">
  Banana Quality Inspector &nbsp;·&nbsp; ML2 Group Project &nbsp;·&nbsp;
  Gradient Boosting · scikit-learn &nbsp;·&nbsp;
  For internal quality operations use only
</div>
""", unsafe_allow_html=True)
