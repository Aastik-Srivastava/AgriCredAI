"""
Carbon Credits Dashboard
- Satellite-style map with farmer carbon zones in green
- ML-estimated carbon sequestration 
- Blockchain ledger
- Credit estimator form
All connected to the shared AdvancedDataPipeline (farmers DB).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from config import DATABASE_PATH

# ─── Carbon credit estimation model (cached once) ───────────────────────────

@st.cache_resource
def _build_cc_model():
    np.random.seed(42)
    N = 800
    X = pd.DataFrame({
        "area": np.random.uniform(0.5, 50, N),
        "ndvi": np.random.uniform(0.15, 0.92, N),
        "soil_carbon": np.random.uniform(4, 45, N),
        "rainfall": np.random.uniform(350, 2000, N),
        "type_afforestation": np.random.binomial(1, 0.25, N),
        "type_notill": np.random.binomial(1, 0.25, N),
        "type_covercrop": np.random.binomial(1, 0.25, N),
        "type_rice": np.random.binomial(1, 0.25, N),
        "verified": np.random.binomial(1, 0.9, N),
    })
    y = np.maximum(
        X["area"] * X["ndvi"] * (
            X["type_afforestation"] * 1.4 + X["type_notill"] * 1.1 +
            X["type_covercrop"] * 1.15 + X["type_rice"] * 1.0
        ) * X["verified"] * 0.92
        + 0.0012 * X["rainfall"] - 0.18 * X["soil_carbon"]
        + np.random.normal(0, 1.2, N), 0
    )
    from sklearn.ensemble import RandomForestRegressor
    mdl = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    mdl.fit(X, y)
    return mdl


# Mapping from crop type → [afforestation, notill, covercrop, rice]
CROP_TYPE_MAP = {
    "Rice":      [0, 0, 0, 1],
    "Wheat":     [0, 1, 0, 0],
    "Cotton":    [0, 0, 1, 0],
    "Sugarcane": [0, 0, 1, 0],
    "Soybean":   [0, 1, 0, 0],
    "Maize":     [0, 0, 1, 0],
    "Other":     [0, 0, 0, 0],
}

NDVI_DEFAULTS  = {"Wheat": 0.65, "Rice": 0.72, "Cotton": 0.58,
                  "Sugarcane": 0.78, "Soybean": 0.61, "Maize": 0.66}


def _estimate_credits(model, area, crop, ndvi, soil_c, rain, verified):
    tv    = CROP_TYPE_MAP.get(crop, [0, 0, 0, 0])
    feats = np.array([[area, ndvi, soil_c, rain, *tv, int(verified)]])
    return max(0.0, float(model.predict(feats)[0]))


def carbon_credits_dashboard(pipeline=None):
    """Carbon Credits Intelligence dashboard."""
    st.header("🌿 Carbon Credits Intelligence Platform")
    st.caption("ML-estimated carbon sequestration · green satellite farm map · credit ledger")
    st.markdown("---")

    # ── 1. Pull farmer data ────────────────────────────────────────────────
    try:
        conn = pipeline.conn if pipeline else sqlite3.connect(DATABASE_PATH)
        farmers_raw = conn.execute(
            "SELECT farmer_id, name, latitude, longitude, land_size, crop_type "
            "FROM farmers "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
            "  AND land_size > 0 "
            "ORDER BY land_size DESC LIMIT 300"
        ).fetchall()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    if not farmers_raw:
        st.warning("No farmers with coordinates found. Run 'Seed 2000 farmers' from the Executive Summary first.")
        return

    # ── 2. Estimate credits via ML ─────────────────────────────────────────
    model = _build_cc_model()
    rng   = np.random.default_rng(seed=42)

    rows = []
    for fid, name, lat, lon, area, crop in farmers_raw:
        crop_k = crop if crop in CROP_TYPE_MAP else "Other"
        ndvi   = NDVI_DEFAULTS.get(crop, 0.6) + rng.uniform(-0.06, 0.06)
        soil_c = rng.uniform(8, 30)
        rain   = rng.uniform(450, 1600)
        credits = _estimate_credits(model, area, crop_k, ndvi, soil_c, rain, True)
        value   = credits * 12.0 * 83.0   # $12/tCO2e × ₹83/$
        rows.append({
            "farmer_id": fid, "name": name, "lat": lat, "lon": lon,
            "area_ha":   round(area, 2),     "crop": crop_k,
            "ndvi":      round(ndvi, 3),
            "credits":   round(credits, 2),
            "value_inr": round(value, 0),
            "car_equiv": round(credits / 4.6, 1),
            "trees_yr":  int(credits / 0.021),
        })

    cc_df = pd.DataFrame(rows)
    total_credits = cc_df["credits"].sum()
    total_value   = cc_df["value_inr"].sum()
    avg_credits   = cc_df["credits"].mean()
    connected_farmers = len(cc_df)

    # ── 3. KPI cards ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🌿 Total tCO₂ Sequestered", f"{total_credits:,.1f}")
    k2.metric("💰 Market Value",            f"₹{total_value/1e5:,.1f}L")
    k3.metric("👨‍🌾 Farmers Enrolled",       str(connected_farmers))
    k4.metric("📊 Avg Credits / Farmer",    f"{avg_credits:.2f} tCO₂")

    st.markdown("---")

    # Also link info from portfolio metrics if pipeline available
    if pipeline:
        try:
            m = pipeline.calculate_and_store_portfolio_metrics()
            p1, p2, p3 = st.columns(3)
            p1.metric("📂 Total Portfolio Farmers", f"{m.get('total_farmers', 0):,}")
            p2.metric("🏦 Active Loans", f"{m.get('active_loans', 0):,}")
            p3.metric("📉 Portfolio Default Rate", f"{m.get('default_rate', 0):.1f}%")
        except Exception:
            pass
        st.markdown("---")

    # ── 4. Spatial Map ───────────────────────────────────────────────────
    st.subheader("🗺️ Farm Interactive Map")
    st.caption("Circle size = farm area (ha) · Color indicates metric. (Token-free layer)")

    metric_choice = st.radio("Display Metric on Map:", ["Carbon Credits (tCO₂)", "NDVI (Vegetation Index)"], horizontal=True)

    max_area = cc_df["area_ha"].max() or 1.0
    cc_df["marker_size"] = (cc_df["area_ha"] / max_area * 35 + 6).clip(6, 42)

    GREEN_SCALE = [
        [0.00, "#002200"],  # very dark evergreen
        [0.25, "#004d00"],
        [0.50, "#009933"],
        [0.75, "#00cc44"],
        [1.00, "#99ff66"],  # bright lime at top
    ]

    NDVI_SCALE = [
        [0.0, "#a50026"],
        [0.2, "#d73027"],
        [0.4, "#f46d43"],
        [0.6, "#fee08b"],
        [0.8, "#66bd63"],
        [1.0, "#006837"]
    ]

    is_ndvi = "NDVI" in metric_choice
    active_color = "ndvi" if is_ndvi else "credits"
    active_scale = NDVI_SCALE if is_ndvi else GREEN_SCALE
    active_title = "NDVI" if is_ndvi else "tCO₂ Credits"

    col_max = cc_df[active_color].max()
    col_min = cc_df[active_color].min()
    if col_max == col_min:
        col_max = col_min + 1.0

    fig_sat = px.scatter_mapbox(
        cc_df, lat="lat", lon="lon",
        color=active_color,
        size="marker_size",
        hover_name="name",
        hover_data={
            "crop":      True,
            "area_ha":   ":.2f",
            "credits":   ":.2f",
            "value_inr": ":.0f",
            "ndvi":      ":.3f",
            "car_equiv": True,
            "lat": False, "lon": False, "marker_size": False
        },
        color_continuous_scale=active_scale,
        zoom=4.0,
        center={"lat": 22.5, "lon": 82.0},
        mapbox_style="carto-darkmatter", # Fix: No mapbox token needed
    )
    fig_sat.update_layout(
        height=620, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title=active_title,
            len=0.75,
            tickvals=[col_min, col_min + (col_max-col_min)*0.5, col_max],
            ticktext=[f"{col_min:.2f} (Low)", "Med", f"{col_max:.2f} (High)"],
        ),
    )
    st.plotly_chart(fig_sat, use_container_width=True)

    # ── 5. Credit estimator form ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧮 Credit Estimator — on-the-fly ML prediction")
    col_form, col_res = st.columns([1, 1])

    with col_form:
        with st.form("cc_estimate"):
            in_area = st.number_input("Farm Area (ha)", 0.5, 200.0, value=5.0, step=0.5)
            in_crop = st.selectbox("Crop / Practice", list(CROP_TYPE_MAP.keys()))
            in_ndvi = st.slider("Vegetation Index (NDVI)", 0.10, 0.95, 0.62, step=0.01)
            in_soil = st.number_input("Baseline Soil Carbon (t/ha)", 1.0, 80.0, value=15.0)
            in_rain = st.number_input("Annual Rainfall (mm)", 200, 2500, value=800)
            in_ver  = st.checkbox("Third-party Verified Practices", value=True)
            submitted = st.form_submit_button("📊 Estimate Credits", type="primary")

    with col_res:
        if submitted:
            pred = _estimate_credits(model, in_area, in_crop, in_ndvi,
                                     in_soil, in_rain, in_ver)
            val  = pred * 12.0 * 83.0
            st.subheader("📋 Estimation Results")
            r1, r2 = st.columns(2)
            r1.metric("Carbon Credits",  f"{pred:.2f} tCO₂e")
            r1.metric("Market Value",    f"₹{val:,.0f}")
            r2.metric("Cars Off-Road",   f"{pred/4.6:.1f} yr")
            r2.metric("Trees Equiv./yr", f"{int(pred/0.021):,}")
            if pred > 0.5:
                st.success("✅ Project qualifies for carbon credit certification.")
                st.info(f"💡 At current market rates (₹996/tCO₂e): **₹{val:,.0f}** annually.")
            elif pred > 0:
                st.warning("⚠️ Low credit potential. Consider verified cover-cropping or no-till methods.")
            else:
                st.error("❌ This combination generates near-zero credits under the ML model.")
        else:
            st.info("Fill in the fields and click **Estimate Credits** to run the ML model.")

    # ── 6. Distribution charts ──────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌾 Credits by Crop Type")
        crop_grp = cc_df.groupby("crop")["credits"].agg(
            Total="sum", Farmers="count", PerFarmer="mean"
        ).reset_index().sort_values("Total", ascending=True)
        fig_crop = px.bar(
            crop_grp, x="Total", y="crop", orientation="h",
            color="Total", color_continuous_scale=GREEN_SCALE,
            labels={"Total": "Total tCO₂e", "crop": ""},
            hover_data={"Farmers": True, "PerFarmer": ":.2f"},
        )
        fig_crop.update_layout(height=330, coloraxis_showscale=False)
        st.plotly_chart(fig_crop, use_container_width=True)

    with c2:
        st.subheader("📊 Credit Distribution")
        fig_hist = px.histogram(
            cc_df, x="credits", nbins=35,
            color_discrete_sequence=["#00cc44"],
            labels={"credits": "tCO₂e per farmer", "count": "Farmers"},
        )
        fig_hist.update_layout(height=330, bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── 7. Top earners table ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Top Carbon Credit Earners")
    top10 = cc_df.nlargest(12, "credits")[
        ["name", "crop", "area_ha", "ndvi", "credits", "value_inr", "car_equiv"]
    ].copy()
    top10.columns = ["Farmer", "Crop", "Area (ha)", "NDVI",
                     "tCO₂ Credits", "Value (₹)", "Cars Equivalent"]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    # ── 8. Blockchain ledger (if file exists) ──────────────────────────────
    try:
        cc_conn   = sqlite3.connect("carbon_credits.db")
        ledger_df = pd.read_sql_query(
            "SELECT * FROM credits ORDER BY timestamp DESC LIMIT 200", cc_conn
        )
        cc_conn.close()
        if not ledger_df.empty:
            st.markdown("---")
            st.subheader("🔗 Blockchain Carbon Ledger")
            st.dataframe(ledger_df, use_container_width=True, hide_index=True)
    except Exception:
        pass   # ledger DB may not exist yet — silently skip

    st.markdown("---")
    st.success(f"✅ Carbon intelligence computed for {connected_farmers} farmers "
               f"· ML model: RandomForest (120 trees) "
               f"· last generated: {datetime.now().strftime('%H:%M:%S')}")
