"""
Weather Risk Monitor — real OpenWeatherMap data per agricultural region.
Called from advanced_app.py as weather_risk_monitor(pipeline).
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime
from config import WEATHER_API_KEY, DATABASE_PATH
from weather_alert_system import WeatherAlertSystem

# Key Indian agricultural regions with their primary crops
AGRI_CITIES = [
    {"name": "Punjab (Ludhiana)",      "lat": 30.9010, "lon": 75.8573, "crop": "Wheat",     "state": "Punjab"},
    {"name": "Maharashtra (Pune)",     "lat": 18.5204, "lon": 73.8567, "crop": "Sugarcane", "state": "Maharashtra"},
    {"name": "Karnataka (Bangalore)",  "lat": 12.9716, "lon": 77.5946, "crop": "Rice",      "state": "Karnataka"},
    {"name": "Tamil Nadu (Chennai)",   "lat": 13.0827, "lon": 80.2707, "crop": "Cotton",    "state": "Tamil Nadu"},
    {"name": "UP (Lucknow)",           "lat": 26.8467, "lon": 80.9462, "crop": "Wheat",     "state": "Uttar Pradesh"},
    {"name": "West Bengal (Kolkata)",  "lat": 22.5726, "lon": 88.3639, "crop": "Rice",      "state": "West Bengal"},
    {"name": "Gujarat (Ahmedabad)",    "lat": 23.0225, "lon": 72.5714, "crop": "Cotton",    "state": "Gujarat"},
    {"name": "Rajasthan (Jaipur)",     "lat": 26.9124, "lon": 75.7873, "crop": "Soybean",   "state": "Rajasthan"},
    {"name": "Madhya Pradesh (Bhopal)","lat": 23.2599, "lon": 77.4126, "crop": "Soybean",   "state": "MP"},
    {"name": "Telangana (Hyderabad)",  "lat": 17.3850, "lon": 78.4867, "crop": "Maize",     "state": "Telangana"},
]


@st.cache_data(ttl=900, show_spinner=False)
def _owm_current(lat: float, lon: float) -> dict:
    """Fetch current weather JSON from OpenWeatherMap (cached 15 min)."""
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon,
                    "appid": WEATHER_API_KEY, "units": "metric"},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _risk_score(data: dict, crop: str) -> float:
    """Weighted 0-1 risk score from current weather."""
    if not data:
        return 0.5
    temp     = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind     = data["wind"]["speed"]
    rain_1h  = (data.get("rain") or {}).get("1h", 0)

    frost_crops = {"Rice", "Cotton", "Sugarcane", "Soybean", "Maize"}
    temp_r  = (0.9 if temp > 42 else 0.7 if temp > 38 else
               (0.8 if crop in frost_crops else 0.2) if temp < 4 else
               0.4 if temp < 10 else 0.1)
    hum_r   = 0.65 if humidity > 90 else 0.35 if humidity > 80 else 0.05
    wind_r  = 0.75 if wind > 20 else 0.35 if wind > 13 else 0.05
    rain_r  = 0.85 if rain_1h > 20 else 0.45 if rain_1h > 8 else 0.05

    return min(temp_r * 0.35 + hum_r * 0.25 + wind_r * 0.20 + rain_r * 0.20, 1.0)


def weather_risk_monitor(pipeline=None):
    """Real-data weather risk monitoring dashboard."""
    st.header("🌤️ Live Weather Risk Monitoring System")
    st.caption("📡 OpenWeatherMap API — refreshed every 15 min. All data is live.")
    st.markdown("---")

    # ── Fetch live weather ─────────────────────────────────────────────────
    with st.spinner("Fetching live conditions from OpenWeatherMap…"):
        rows = []
        for city in AGRI_CITIES:
            raw  = _owm_current(city["lat"], city["lon"])
            risk = _risk_score(raw, city["crop"])
            if raw:
                rows.append({
                    "region":      city["name"],
                    "state":       city["state"],
                    "crop":        city["crop"],
                    "lat":         city["lat"],
                    "lon":         city["lon"],
                    "temp":        raw["main"]["temp"],
                    "feels_like":  raw["main"]["feels_like"],
                    "humidity":    raw["main"]["humidity"],
                    "pressure":    raw["main"]["pressure"],
                    "wind_speed":  raw["wind"]["speed"],
                    "condition":   raw["weather"][0]["description"].title(),
                    "rain_1h":     (raw.get("rain") or {}).get("1h", 0),
                    "risk_score":  risk,
                    "risk_cat":   ("🔴 High" if risk > 0.6 else
                                   "🟡 Medium" if risk > 0.3 else "🟢 Low"),
                    "live":        True,
                })
            else:
                rows.append({
                    "region": city["name"], "state": city["state"],
                    "crop": city["crop"],   "lat": city["lat"],
                    "lon": city["lon"],
                    "temp": None, "feels_like": None, "humidity": None,
                    "pressure": None,       "wind_speed": None,
                    "condition": "API unavailable",
                    "rain_1h": 0.0,         "risk_score": 0.5,
                    "risk_cat": "⚠️ Unknown","live": False,
                })

    df   = pd.DataFrame(rows)
    live = df["live"].sum()

    # ── KPI bar ────────────────────────────────────────────────────────────
    total_farmers = 0
    if pipeline:
        try:
            m = pipeline.calculate_and_store_portfolio_metrics()
            total_farmers = m.get("total_farmers", 0)
        except Exception:
            pass

    high  = int((df["risk_cat"].str.startswith("🔴")).sum())
    med   = int((df["risk_cat"].str.startswith("🟡")).sum())
    low   = int((df["risk_cat"].str.startswith("🟢")).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👨‍🌾 Farmers in DB", f"{total_farmers:,}" if total_farmers else "N/A")
    c2.metric("🔴 High-Risk Regions",   str(high))
    c3.metric("🟡 Medium-Risk Regions", str(med))
    c4.metric("🟢 Safe Regions",         str(low))
    st.markdown("---")

    # ── Live map ───────────────────────────────────────────────────────────
    st.subheader("🗺️ Live Agricultural Weather-Risk Map")
    df_live = df[df["live"]].copy()
    if not df_live.empty:
        fig = px.scatter_mapbox(
            df_live, lat="lat", lon="lon",
            color="risk_score", size=[28] * len(df_live),
            hover_name="region",
            hover_data={
                "crop": True, "state": True,
                "temp": ":.1f", "humidity": ":.0f",
                "wind_speed": ":.1f", "condition": True, "risk_cat": True,
                "lat": False, "lon": False, "risk_score": False, "live": False,
            },
            color_continuous_scale="RdYlGn_r",
            zoom=4.3, center={"lat": 22, "lon": 80},
        )
        fig.update_layout(
            mapbox_style="open-street-map", height=520,
            margin={"r": 0, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title="Risk",
                tickvals=[0, 0.33, 0.66, 1.0],
                ticktext=["Low", "Moderate", "High", "Critical"],
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Could not reach OpenWeatherMap API. "
                 "Verify `WEATHER_API_KEY` in `.env` and your internet connection.")

    # ── Regional cards ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Real-Time Regional Conditions")
    grid = st.columns(2)
    for i, row in df.iterrows():
        rc = row["risk_cat"]
        border = ("#dc3545" if "High" in rc else
                  "#ffc107" if "Medium" in rc else "#28a745")
        t  = f"{row['temp']:.1f}°C"   if row["temp"]       is not None else "N/A"
        h  = f"{row['humidity']:.0f}%" if row["humidity"]   is not None else "N/A"
        w  = f"{row['wind_speed']:.1f} m/s" if row["wind_speed"] is not None else "N/A"
        r1h= f"{row['rain_1h']:.1f} mm/h" if row["live"] else "—"
        with grid[i % 2]:
            st.markdown(f"""
            <div style='border-left:5px solid {border}; padding:11px 14px;
                        border-radius:8px; margin-bottom:10px; background:#111a11'>
              <b style='font-size:15px'>{row['region']}</b>
              &nbsp;<span style='color:{border}; font-size:12px'>{rc}</span><br>
              <span style='font-size:13px; color:#ccc'>{row['condition']}</span><br>
              🌡 {t} &nbsp; 💧 {h} &nbsp; 💨 {w} &nbsp; 🌧 {r1h} &nbsp; 🌾 {row['crop']}<br>
              <span style='font-size:11px; opacity:0.55'>
                {'🟢 Live · OpenWeatherMap' if row['live'] else '⚠️ API unavailable'}
              </span>
            </div>""", unsafe_allow_html=True)

    # ── Alert engine ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚨 Farmer Alert Engine")
    _, bcol = st.columns([3, 1])
    run_alerts = bcol.button("🔄 Run Alert Scan", type="primary",
                             key="weather_alert_btn")

    if run_alerts or "weather_alerts_cache" not in st.session_state:
        with st.spinner("Analysing weather vs. all DB farmers…"):
            try:
                ws = WeatherAlertSystem()
                result = ws.run_once_mvp()
            except Exception as e:
                result = {"status": "error", "alerts": [], "summary": str(e)}
            st.session_state["weather_alerts_cache"] = result

    ar = st.session_state.get("weather_alerts_cache", {"alerts": [], "summary": ""})
    st.caption(f"Last scan: {ar.get('summary', '—')}")

    alerts = ar.get("alerts", [])
    if alerts:
        for i, alert in enumerate(alerts[:10], 1):
            sev  = alert.get("severity", "low")
            icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "🟢"
            with st.expander(
                f"{icon} #{i} — {alert.get('farmer_name', 'Unknown')} "
                f"· {alert.get('type', '').replace('_', ' ').title()}",
                expanded=(i <= 2)
            ):
                st.write(f"**Severity:** {sev.title()}")
                st.write(f"**Message:**  {alert.get('message', '—')}")
                st.write(f"**Action:**   {alert.get('recommended_action', '—')}")
                st.progress(alert.get("risk_level", 0),
                            text=f"Risk {alert.get('risk_level', 0):.0%}")
    else:
        st.info("🌤️ No critical alerts. All monitored farmers are in safe conditions.")

    # ── Temperature bar chart ──────────────────────────────────────────────
    df_bar = df[df["live"]].sort_values("temp")
    if not df_bar.empty:
        st.markdown("---")
        st.subheader("🌡️ Temperature Snapshot by Region")
        fig_bar = px.bar(
            df_bar, x="temp", y="region", orientation="h",
            color="risk_score", color_continuous_scale="RdYlGn_r",
            labels={"temp": "Temperature (°C)", "region": ""},
            text_auto=".1f",
        )
        fig_bar.update_layout(height=370, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Humidity vs temp scatter ───────────────────────────────────────────
    df_sc = df[df["live"]].copy()
    if len(df_sc) >= 3:
        st.subheader("💧 Humidity × Temperature Risk Scatter")
        fig_sc = px.scatter(
            df_sc, x="temp", y="humidity",
            color="risk_score", size="wind_speed",
            hover_name="region",
            color_continuous_scale="RdYlGn_r",
            labels={"temp": "Temp (°C)", "humidity": "Humidity (%)",
                    "wind_speed": "Wind (m/s)"},
            range_color=[0, 1],
        )
        fig_sc.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Status footer ──────────────────────────────────────────────────────
    st.markdown("---")
    if live == len(AGRI_CITIES):
        st.success(f"🟢 All {live} regions live · OpenWeatherMap API · "
                   f"Updated {datetime.now().strftime('%H:%M:%S')}")
    elif live > 0:
        st.warning(f"🟡 {live}/{len(AGRI_CITIES)} regions live · "
                   f"Updated {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.error("🔴 No live data — check WEATHER_API_KEY in `.env`")
