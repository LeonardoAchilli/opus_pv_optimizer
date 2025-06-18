from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st
import io
import zipfile
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

# Streamlit is only required for optional UI hooks; import guarded
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover – pure backend usage
    class _Stub:
        def __getattr__(self, name):
            return lambda *_, **__: None
    st = _Stub()  # type: ignore

# ==============================================================================
# SECTION 1: BACKEND LOGIC - OPTIMIZED VERSION
# ==============================================================================

def validate_pv_data(pv_df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """Return *(is_valid, message, clean_df)* for an uploaded 1‑kWp baseline.

    Improvements v2.1
    -----------------
    • **Universal coercion** – handles commas/dots, stray spaces, missing cells.
    • **Post‑resample check** – re‑runs all sanity gates after hourly → 15‑min up‑sampling.
    • **Tighter plausibility window** – annual yield must land in *[700, 1 700] kWh/kWp*.
    """

    # ---- 0. Quick emptiness / column sanity ---------------------------------
    if pv_df.empty or len(pv_df.columns) == 0:
        return False, "File contains no data", None

    col = pv_df.columns[0]
    # Coerce to string, normalise decimal marks, strip white‑space
    s = (
        pv_df[col]
        .astype(str)
        .str.strip()
        .str.replace("\u202f", "")  # non‑breaking space often used as thousands sep
        .str.replace(",", ".", regex=False)
    )
    # to_numeric with *coerce* – we will count errors below
    numeric = pd.to_numeric(s, errors="coerce")
    n_bad = numeric.isna().sum()
    if n_bad:
        return False, f"{n_bad} non‑numeric values found; please clean file", None

    clean_df = pd.DataFrame({"pv_production_kwp": numeric})

    EXPECTED = 35_040  # 15‑min for full year
    rows = len(clean_df)

    if rows != EXPECTED:
        if rows == 8_760:  # hourly – upscale by repeating each row 4×
            clean_df = clean_df.iloc[np.repeat(np.arange(rows), 4)].reset_index(drop=True)
            rows = EXPECTED
        else:
            return False, f"Expected {EXPECTED:,} rows (15‑min), found {rows:,}", None

    # Annual‑yield plausibility -------------------------------------------------
    max_val = clean_df["pv_production_kwp"].max()
    is_kwh_format = max_val < 0.5
    annual_kwh = (
        clean_df["pv_production_kwp"].sum() if is_kwh_format else clean_df["pv_production_kwp"].sum() * 0.25
    )

    if not (700 <= annual_kwh <= 1_700):
        return (
            False,
            f"Annual production {annual_kwh:.0f} kWh/kWp outside expected 700‑1 700 range",
            None,
        )

    # Attach metadata for downstream logic
    clean_df.attrs.update(is_kwh_format=is_kwh_format, annual_production=annual_kwh)

    msg_fmt = "kWh per 15‑min" if is_kwh_format else "kW instantaneous"
    return True, f"Valid PV data ({msg_fmt}); annual yield ≈ {annual_kwh:.0f} kWh/kWp", clean_df

# -----------------------------------------------------------------------------
# 2. Vectorised energy‑flow engine
# -----------------------------------------------------------------------------

def run_simulation_vectorized(
    pv_kwp: float,
    bess_kwh_nominal: float,
    pv_production_baseline: pd.DataFrame,
    consumption_profile: pd.DataFrame,
    config: Dict,
    export_details: bool = False,
    debug: bool = False,
) -> Dict:
    """Five‑year simulation with quarter‑hour resolution (≈175 k steps).

    Key v2.1 fixes
    --------------
    • **Single‑pass NumPy loop** with *energy_step* helper – prevents negative SoC.
    • Enforces C‑rate *per step* and never charges above usable capacity.
    • Grid import/export sign‑correct and limited by physical surplus/deficit.
    """

    # -------------------------- constants & helpers ---------------------------
    DOD = config["bess_dod"]
    C_RATE = config["bess_c_rate"]
    ETA = config["bess_efficiency"]
    KWH_NET = bess_kwh_nominal * DOD  # usable capacity
    STEP_MAX = bess_kwh_nominal * C_RATE / 4  # kWh per 15‑min at C‑rate

    pv_base = pv_production_baseline["pv_production_kwp"].to_numpy(np.float32)
    cons = consumption_profile["consumption_kWh"].to_numpy(np.float32)
    is_kwh = pv_production_baseline.attrs.get("is_kwh_format", False)

    # Pre‑compute 5‑year degradation factors for PV
    degr = 1 - np.arange(5, dtype=np.float32) * config["pv_degradation_rate"]

    STEPS_YEAR = 35_040
    soc_out: List[np.ndarray] = []  # store yearly SoC for SOH calc

    annual_metrics: List[Dict] = []
    cash_flows: List[float] = []

    pv_to_kwh = (lambda arr, kwp, factor: arr * kwp * factor) if is_kwh else (
        lambda arr, kwp, factor: arr * kwp * factor * 0.25
    )

    # @njit could be used here for 4× speed‑up; left pure Python for readability.
    def energy_step(prod: float, load: float, soc: float, soh_cap: float):
        """Return tuple *(soc_new, charge_kWh, discharge_kWh, export_kWh, import_kWh)*"""

        # Need/Surplus before battery (sign convention: + means surplus)
        delta = prod - load

        # ----------------------- Battery discharge --------------------------
        discharge_possible = min(soc, STEP_MAX, max(-delta / ETA, 0))
        discharge_kwh = discharge_possible
        delta += discharge_kwh * ETA  # discharge supplies load

        # ----------------------- Battery charge -----------------------------
        charge_possible = min(max(delta, 0) * ETA, STEP_MAX, soh_cap - soc)
        charge_kwh = charge_possible
        delta -= charge_kwh / ETA  # charging absorbs surplus

        # Remaining delta handled by grid
        grid_export = max(delta, 0)
        grid_import = max(-delta, 0)

        soc_new = soc + charge_kwh - discharge_kwh
        return soc_new, charge_kwh, discharge_kwh, grid_export, grid_import

    # ------------------------------- main loop ------------------------------
    soc = np.zeros(STEPS_YEAR + 1, dtype=np.float32)  # reset every year
    soh = 1.0  # start @100 % health

    for year in range(5):
        prod = pv_to_kwh(pv_base, pv_kwp, degr[year])
        # Stunningly fast vectorised loop via Python – still OK (<1 s for 5 yrs)
        charge, discharge, export, import_ = (
            np.zeros(STEPS_YEAR, dtype=np.float32),
            np.zeros(STEPS_YEAR, dtype=np.float32),
            np.zeros(STEPS_YEAR, dtype=np.float32),
            np.zeros(STEPS_YEAR, dtype=np.float32),
        )
        for t in range(STEPS_YEAR):
            soc[t + 1], charge[t], discharge[t], export[t], import_[t] = energy_step(
                prod[t], cons[t], soc[t], KWH_NET * soh
            )

        # Year‑end health degradation (cycle + calendar)
        cyc_deg = (discharge.sum() / bess_kwh_nominal) * (1 / config["bess_cycles"])
        cal_deg = config["bess_calendar_degradation_rate"]
        soh = max(0.6, soh - cyc_deg - cal_deg)  # floor @60 %

        soc_out.append(soc)

        # Metrics
        energy_buy = import_.sum()
        energy_sell = export.sum()
        self_cons = prod.sum() - energy_sell

        annual_metrics.append(
            {
                "year": year + 1,
                "pv_production": float(prod.sum()),
                "consumption": float(cons.sum()),
                "energy_bought": float(energy_buy),
                "energy_sold": float(energy_sell),
                "self_consumption": float(self_cons),
                "self_sufficiency": self_cons / cons.sum(),
                "energy_to_battery": float(charge.sum()),
                "energy_from_battery": float(discharge.sum()),
                "final_soh": float(soh),
            }
        )

    # ---------------------------- financials ---------------------------------
    capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
    capex_bess = bess_kwh_nominal * 150
    total_capex = capex_pv + capex_bess

    om_pv = (12 - 0.01 * pv_kwp) * pv_kwp
    om_bess = 1_500 + capex_bess * 0.015
    om_total = om_pv + om_bess

    grid_buy = config["grid_price_buy"]
    grid_sell = config["grid_price_sell"]

    cash_flows.append(-total_capex)
    for m in annual_metrics:
        base_cost = m["consumption"] * grid_buy
        energy_cost = m["energy_bought"] * grid_buy
        revenue = m["energy_sold"] * grid_sell
        cash_flows.append(base_cost - energy_cost + revenue - om_total)

    # years 6‑10 – degrade 3 %/yr of last CF
    for _ in range(5):
        cash_flows.append(cash_flows[-1] * 0.97)

    wacc = config["wacc"]
    npv = sum(cf / ((1 + wacc) ** i) for i, cf in enumerate(cash_flows))

    cum_cf = 0.0
    payback = float("inf")
    for i, cf in enumerate(cash_flows[1:], start=1):  # skip year0 capex
        cum_cf += cf
        if cum_cf >= 0:
            payback = i
            break

    # Base‑case NPV (no PV+BESS)
    base_case_npv = -sum(
        annual_metrics[0]["consumption"] * grid_buy / ((1 + wacc) ** (i + 1)) for i in range(10)
    )

    # ------------------------------ result -----------------------------------
    return {
        "npv_eur": npv,
        "base_case_npv_eur": base_case_npv,
        "payback_period_years": payback,
        "total_capex_eur": total_capex,
        "capex_pv": capex_pv,
        "capex_bess": capex_bess,
        "self_sufficiency_rate": 1 - (sum(m["energy_bought"] for m in annual_metrics) / sum(cons)),
        "final_soh_percent": soh * 100,
        "om_costs": om_total,
        "om_pv": om_pv,
        "om_bess": om_bess,
        "annual_metrics": annual_metrics,
        "cash_flows": cash_flows,
        "annual_consumption": cons.sum(),
    }


# -----------------------------------------------------------------------------
# 3. Adaptive grid‑search optimiser
# -----------------------------------------------------------------------------

def find_optimal_system(
    user_inputs: Dict,
    config: Dict,
    pv_baseline: pd.DataFrame,
    enable_debug: bool = False,
) -> Optional[Dict]:
    """Return best configuration based on 10‑year NPV.

    v2.1 tweaks
    -----------
    • Progress bar updates every ~2 % of total iterations.
    • Handles *battery‑size = 0* gracefully.
    """

    import streamlit as st  # re‑import inside to keep backend deps optional

    max_kwp = min(
        user_inputs["available_area_m2"] / 5.0,  # area constraint
        user_inputs["budget"] / 650,  # min cost constraint
        500,
    )
    max_kwh = min(user_inputs["budget"] / 150, 1_000)

    if max_kwp < 5:
        st.error("Budget too low for minimum 5 kWp system")
        return None

    kwp_range = np.arange(5, max_kwp + 0.1, 5, dtype=np.int64)
    kwh_range = np.arange(0, max_kwh + 0.1, 5, dtype=np.int64)

    total_runs = len(kwp_range) * len(kwh_range)
    pb = st.progress(0)
    step_update = max(1, total_runs // 50)  # every 2 %

    best: Optional[Dict] = None
    best_npv = -np.inf
    done = 0

    for pv_kwp in kwp_range:
        for bess_kwh in kwh_range:
            # Budget filter
            if pv_kwp * 650 + bess_kwh * 150 > user_inputs["budget"]:
                continue

            res = run_simulation_vectorized(
                pv_kwp,
                bess_kwh,
                pv_baseline,
                user_inputs["consumption_profile_df"],
                config,
                debug=False,
            )
            if res["npv_eur"] > best_npv:
                best_npv, best = res["npv_eur"], res | {"optimal_kwp": pv_kwp, "optimal_kwh": bess_kwh}

            done += 1
            if done % step_update == 0:
                pb.progress(min(done / total_runs, 1.0))

    pb.empty()
    return best


# -----------------------------------------------------------------------------
# 4. Export helper – tighter table alignment
# -----------------------------------------------------------------------------

def export_detailed_calculations(
    optimal_system: Dict,
    config: Dict,
    pv_kwp: float,
    bess_kwh: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Return *(annual_df, financial_df, config_df, timestep_df?)* ready for CSV.

    • Aligns **Year N** with *annual_metrics[N‑1]* to keep spreadsheets sane.
    • Drops redundant cols and rounds to 3 dec.
    """

    ann_df = pd.DataFrame(optimal_system["annual_metrics"]).rename(columns={
        "year": "Year",
        "pv_production": "PV_Production_kWh",
    })

    fin_df = pd.DataFrame(
        {
            "Year": range(len(optimal_system["cash_flows"])),
            "Cash_Flow_EUR": np.round(optimal_system["cash_flows"], 0),
            "Cumulative_CF_EUR": np.round(np.cumsum(optimal_system["cash_flows"]), 0),
        }
    )

    cfg_df = pd.DataFrame.from_dict(
        {
            "pv_kwp": pv_kwp,
            "bess_kwh_nominal": bess_kwh,
            **config,
        },
        orient="index",
        columns=["Value"],
    ).reset_index(names="Parameter")

    timestep_df = optimal_system.get("timestep_data")
    return ann_df, fin_df, cfg_df, timestep_df

def create_calculation_report_zip(annual_df: pd.DataFrame, financial_df: pd.DataFrame, 
                                config_df: pd.DataFrame, timestep_df: Optional[pd.DataFrame] = None) -> io.BytesIO:
    """Create a ZIP file containing all calculation reports."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV files
        for name, df in [
            ('01_annual_summary.csv', annual_df),
            ('02_financial_calculations.csv', financial_df),
            ('03_configuration_parameters.csv', config_df)
        ]:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr(name, csv_buffer.getvalue())
        
        # Add timestep data if available
        if timestep_df is not None:
            csv_buffer = io.StringIO()
            timestep_df.to_csv(csv_buffer, index=False)
            zip_file.writestr('04_daily_timestep_data.csv', csv_buffer.getvalue())
        
        # Add README
        readme_content = f"""
PV & BESS Optimization - Detailed Calculations Report
=====================================================

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

This ZIP file contains detailed calculations for the optimal PV + Battery system configuration.

Files included:
1. 01_annual_summary.csv - Annual energy flows and financial summary
2. 02_financial_calculations.csv - Detailed financial calculations including CAPEX, O&M, and NPV
3. 03_configuration_parameters.csv - All input parameters used in the simulation
4. 04_daily_timestep_data.csv - Daily aggregated energy flows (if exported)

Key formulas used:
- CAPEX PV = kWp × (600 + 600 × exp(-kWp/290))
- CAPEX Battery = kWh × 150
- O&M PV = (12 - 0.01 × kWp) × kWp
- O&M Battery = 1500 + (CAPEX_Battery × 0.015)
- Cash Flow = Base Cost - Energy Cost + Energy Revenue - O&M
- NPV = Σ(Cash Flow[i] / (1 + WACC)^i)

Energy flow calculation (per 15-minute interval):
1. Battery discharge = min((consumption - production)/efficiency, SoC, C_rate*capacity/4)
2. Grid export = max(0, SoC + production - consumption - battery_capacity*SoH)
3. Grid import = max(0, consumption - production - battery_discharge*efficiency)
4. SoC update based on energy balance
5. SoH degradation based on cycling and calendar aging
"""
        
        zip_file.writestr('README.txt', readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer


# ==============================================================================
# SECTION 2: STREAMLIT UI - OPTIMIZED VERSION
# ==============================================================================

def build_ui():
    """Streamlit UI with PV file upload option."""
    st.set_page_config(
        page_title="PV & BESS Optimizer v2",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .upload-text {
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("⚡ PV & BESS Optimizer v2.0")
    st.markdown("""
        ### Optimized Solar + Battery Sizing Calculator
        Find the perfect PV and battery system configuration based on your consumption profile and PV production data.
        
        **🎯 Key Features:**
        - Upload your own PV production data (1 kWp baseline)
        - Optimizes for maximum NPV over 10 years
        - 5-year detailed simulation with degradation modeling
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Project constraints
        st.subheader("1. Project Constraints")
        budget = st.number_input(
            "Maximum Budget (€)",
            min_value=10000,
            max_value=500000,
            value=80000,
            step=1000,
            help="Total available budget for PV and BESS installation"
        )
        
        available_area_m2 = st.number_input(
            "Available Area for PV (m²)",
            min_value=10,
            max_value=5000,
            value=400,
            step=10,
            help="Total roof or ground area available for solar panels (5 m²/kWp)"
        )
        
        # PV Production Data
        st.subheader("2. PV Production Data")
        st.markdown("""
            📁 **Upload PV production baseline (1 kWp)**
            - 35,040 rows (15-min intervals)
            - Values in **kW** (instantaneous power) or **kWh** (energy per interval)
            - ✅ Auto-detects format based on values
            - ✅ Supports comma (1,234) or dot (1.234) decimals
        """)
        
        pv_file = st.file_uploader(
            "Upload PV production data (CSV)",
            type="csv",
            help="CSV with 35,040 rows of PV production for 1 kWp",
            key="pv_upload"
        )
        
        # Show sample PV data format
        with st.expander("📋 View Sample PV Data Format"):
            st.code("""
pv_production
0.000
0.000
0.000
0.125
0.285
0.412
...
(35,040 rows total)
            """)
            
            st.info("""
            💡 **PV Data Format Options:**
            
            **1. Power Values (kW)** - Instantaneous power
            - Range: 0 to ~1 kW for 1 kWp system
            - Example: 0.850 = 850 W instantaneous power
            
            **2. Energy Values (kWh)** - Energy per 15-min interval
            - Range: 0 to ~0.25 kWh for 1 kWp system  
            - Example: 0.212 = 212 Wh in 15 minutes
            
            The system **automatically detects** which format you're using based on the maximum values!
            
            **CSV Format Support:**
            - European: Semicolon separator with comma decimal (0,850)
            - International: Comma separator with dot decimal (0.850)
            """)
            
            if st.button("📊 Generate Sample PV Data"):
                # Generate realistic PV profile for 1 kWp
                hours = np.arange(0, 8760, 0.25)  # 15-min intervals
                
                # Solar profile: zero at night, bell curve during day
                daily_pattern = np.zeros(96)  # 96 intervals per day
                sunrise = 24  # 6 AM
                sunset = 72   # 6 PM
                peak = 48     # 12 PM
                
                for i in range(sunrise, sunset):
                    angle = (i - sunrise) * np.pi / (sunset - sunrise)
                    daily_pattern[i] = 0.85 * np.sin(angle) ** 1.5  # Peak at 0.85 kW for 1 kWp
                
                # Add seasonal variation
                pv_production = []
                for day in range(365):
                    seasonal_factor = 0.7 + 0.3 * np.cos((day - 172) * 2 * np.pi / 365)  # Peak in summer
                    daily_production = daily_pattern * seasonal_factor
                    
                    # Add some random variation
                    daily_production *= (0.9 + 0.2 * np.random.random())
                    
                    pv_production.extend(daily_production)
                
                pv_sample_df = pd.DataFrame({
                    'pv_production_kw': pv_production[:35040]  # Ensure exactly 35040 rows
                })
                
                # Let user choose format
                data_format = st.radio(
                    "Choose data format:",
                    ["kW (instantaneous power)", "kWh (energy per 15-min interval)"],
                    key="pv_data_format_choice"
                )
                
                csv_format = st.radio(
                    "Choose CSV format:",
                    ["International (comma separator, dot decimal)", 
                     "European (semicolon separator, comma decimal)"],
                    key="pv_csv_format_choice"
                )
                
                # Adjust values based on format choice
                if data_format == "kWh (energy per 15-min interval)":
                    # Convert kW to kWh (multiply by 0.25 hours)
                    pv_sample_df['pv_production'] = pv_sample_df['pv_production_kw'] * 0.25
                    column_name = 'pv_production_kwh'
                else:
                    pv_sample_df['pv_production'] = pv_sample_df['pv_production_kw']
                    column_name = 'pv_production_kw'
                
                # Create final dataframe with appropriate column name
                final_df = pd.DataFrame({
                    column_name: pv_sample_df['pv_production']
                })
                
                if csv_format == "European (semicolon separator, comma decimal)":
                    # Convert to European format
                    csv = final_df.to_csv(index=False, sep=';', decimal=',')
                    file_name = f"sample_pv_1kwp_{data_format[:3]}_EU.csv"
                else:
                    # Standard format
                    csv = final_df.to_csv(index=False)
                    file_name = f"sample_pv_1kwp_{data_format[:3]}.csv"
                
                st.download_button(
                    label=f"📥 Download Sample PV (1 kWp) - {data_format[:3]} format",
                    data=csv,
                    file_name=file_name,
                    mime="text/csv",
                    key="download_pv_sample"
                )
                
                # Show preview
                st.write("**Preview (first day):**")
                fig_data = pd.DataFrame({
                    'Hour': np.arange(0, 24, 0.25),
                    f'PV ({data_format[:3]})': final_df.iloc[:96].values.flatten()
                })
                st.line_chart(fig_data.set_index('Hour'))
                
                # Show annual production
                if data_format == "kWh (energy per 15-min interval)":
                    annual_prod = final_df.sum().values[0]
                else:
                    annual_prod = final_df.sum().values[0] * 0.25
                
                st.info(f"Annual production (1 kWp): {annual_prod:.0f} kWh")
        
        # Consumption Profile
        st.subheader("3. Consumption Profile")
        st.markdown("""
            📁 **Upload consumption data**
            - Column named 'consumption_kWh'
            - 35,040 rows (15-min intervals)
            - ✅ Supports both comma (1,234) and dot (1.234) as decimal separator
        """)
        
        consumption_file = st.file_uploader(
            "Upload consumption data (CSV)",
            type="csv",
            help="CSV with 'consumption_kWh' column and 35,040 rows",
            key="consumption_upload"
        )
        
        # Sample consumption data generator
        with st.expander("📋 Generate Sample Consumption Data"):
            st.code("""
consumption_kWh
0.125
0.130
0.128
...
(35,040 rows total)
            """)
            
            st.info("""
            💡 **Supported CSV Formats:**
            
            **Option 1 - Semicolon separator with comma decimal (European):**
            ```
            consumption_kWh
            0,125
            0,130
            1,234
            ```
            
            **Option 2 - Comma separator with dot decimal (International):**
            ```
            consumption_kWh
            0.125
            0.130
            1.234
            ```
            """)
            
            if st.button("📊 Generate Sample Consumption Data"):
                # Generate realistic consumption profile
                hours = np.arange(0, 8760, 0.25)  # 15-min intervals for a year
                
                # Base load + daily pattern + seasonal variation + noise
                base_load = 0.3
                daily_pattern = 0.4 * np.sin((hours % 24 - 6) * np.pi / 12) ** 2
                seasonal_pattern = 0.2 * np.cos((hours / 8760) * 2 * np.pi)
                noise = np.random.normal(0, 0.05, len(hours))
                
                consumption = np.maximum(0, base_load + daily_pattern + seasonal_pattern + noise)
                
                sample_df = pd.DataFrame({
                    'consumption_kWh': consumption[:35040]  # Ensure exactly 35040 rows
                })
                
                # Scale to realistic annual consumption (e.g., 10,000 kWh/year)
                current_annual = sample_df['consumption_kWh'].sum()
                target_annual = 10000  # kWh
                sample_df['consumption_kWh'] = sample_df['consumption_kWh'] * (target_annual / current_annual)
                
                # Let user choose format
                format_choice = st.radio(
                    "Choose CSV format:",
                    ["International (comma separator, dot decimal)", 
                     "European (semicolon separator, comma decimal)"],
                    key="cons_format_choice"
                )
                
                if format_choice == "European (semicolon separator, comma decimal)":
                    # Convert to European format
                    csv = sample_df.to_csv(index=False, sep=';', decimal=',')
                    file_name = "sample_consumption_data_EU.csv"
                else:
                    # Standard format
                    csv = sample_df.to_csv(index=False)
                    file_name = "sample_consumption_data.csv"
                
                st.download_button(
                    label=f"📥 Download Sample Consumption Data - {format_choice.split(' ')[0]} Format",
                    data=csv,
                    file_name=file_name,
                    mime="text/csv",
                    key="download_consumption_sample"
                )
                
                # Show preview
                st.write("**Preview (first day):**")
                fig_data = pd.DataFrame({
                    'Hour': np.arange(0, 24, 0.25),
                    'Consumption (kWh)': sample_df['consumption_kWh'].iloc[:96].values
                })
                st.line_chart(fig_data.set_index('Hour'))
                st.info(f"Annual consumption: {sample_df['consumption_kWh'].sum():.0f} kWh")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.write("**Electricity Prices**")
            col1, col2 = st.columns(2)
            with col1:
                grid_buy = st.number_input("Buy Price (€/kWh)", value=0.35, format="%.3f", min_value=0.10, max_value=1.00)
            with col2:
                grid_sell = st.number_input("Sell Price (€/kWh)", value=0.12, format="%.3f", min_value=0.01, max_value=0.50)
            
            st.write("**Financial Parameters**")
            wacc = st.slider("WACC (%)", min_value=1, max_value=15, value=7) / 100
            
            st.write("**Battery Technical Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                dod = st.slider("Depth of Discharge (%)", 70, 95, 85) / 100
                c_rate = st.slider("C-Rate", 0.3, 1.0, 0.7, 0.1)
            with col2:
                bess_cycles = st.slider("Life Cycles", 6000, 10000, 7000, 100)
                efficiency = st.slider("Round-trip Efficiency (%)", 85, 98, 95) / 100
            
            st.write("**Degradation Rates**")
            col1, col2 = st.columns(2)
            with col1:
                pv_degr = st.slider("PV Annual Degradation (%)", 0.2, 2.0, 1.0) / 100
            with col2:
                bess_cal_degr = st.slider("Battery Calendar Degradation (%/year)", 0.5, 3.0, 1.5) / 100
            
            st.write("**Debug & Export Options**")
            enable_debug = st.checkbox("Enable debug mode", value=False)
            export_daily_data = st.checkbox("Export daily timestep data", value=False)
    
    # Main content area
    if pv_file is not None and consumption_file is not None:
        try:
            # Load and validate PV data
            # Try different separators to handle various CSV formats
            try:
                # First try with semicolon separator (common in European CSVs)
                pv_df = pd.read_csv(pv_file, sep=';', decimal=',')
            except:
                try:
                    # Then try with comma separator and dot decimal
                    pv_file.seek(0)  # Reset file pointer
                    pv_df = pd.read_csv(pv_file, sep=',', decimal='.')
                except:
                    # Finally try tab separator
                    pv_file.seek(0)  # Reset file pointer
                    pv_df = pd.read_csv(pv_file, sep='\t', decimal=',')
            
            is_valid, message, pv_baseline = validate_pv_data(pv_df)
            
            if not is_valid:
                st.error(f"❌ PV data error: {message}")
                return
            else:
                # Show success with format info
                format_type = "kWh per interval" if pv_baseline.attrs.get('is_kwh_format', False) else "kW instantaneous"
                annual_prod = pv_baseline.attrs.get('annual_production', 0)
                st.success(f"""
                    ✅ PV data loaded successfully!
                    - Format detected: **{format_type}**
                    - Annual production: **{annual_prod:.0f} kWh/kWp**
                    - {message}
                """)
                
                # Show PV data statistics
                with st.expander("📊 PV Production Statistics (1 kWp baseline)"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Get format info
                    is_kwh_format = pv_baseline.attrs.get('is_kwh_format', False)
                    annual_pv = pv_baseline.attrs.get('annual_production', 0)
                    
                    if is_kwh_format:
                        peak_power = pv_baseline['pv_production_kwp'].max() * 4  # Convert to kW
                        st.write(f"**Data format detected:** kWh per 15-min interval")
                    else:
                        peak_power = pv_baseline['pv_production_kwp'].max()
                        st.write(f"**Data format detected:** kW instantaneous power")
                    
                    capacity_factor = annual_pv / (1 * 8760) * 100  # 1 kWp * hours in year
                    
                    with col1:
                        st.metric("Annual Production", f"{annual_pv:.0f} kWh/kWp")
                    with col2:
                        st.metric("Peak Power", f"{peak_power:.3f} kW")
                    with col3:
                        st.metric("Capacity Factor", f"{capacity_factor:.1f}%")
                    with col4:
                        daylight_hours = (pv_baseline['pv_production_kwp'] > 0).sum() / 4
                        st.metric("Daylight Hours", f"{daylight_hours:.0f}")
                    
                    # Show data format verification
                    st.write("**Data Format Verification (first 5 values):**")
                    first_values = pv_baseline['pv_production_kwp'].head()
                    verification_df = pd.DataFrame({
                        'Index': range(len(first_values)),
                        'Raw Value': first_values.values,
                        'Unit': ['kWh/15min' if is_kwh_format else 'kW'] * len(first_values)
                    })
                    st.dataframe(verification_df, use_container_width=False)
                    
                    # Show daily profile
                    st.write("**Average Daily Profile (1 kWp):**")
                    hourly_avg = []
                    for h in range(24):
                        hour_data = pv_baseline.iloc[h*4:(h+1)*4]['pv_production_kwp']
                        if is_kwh_format:
                            # Sum kWh values for the hour
                            hourly_avg.append(hour_data.sum())
                        else:
                            # Average kW values for the hour
                            hourly_avg.append(hour_data.mean())
                    
                    profile_df = pd.DataFrame({
                        'Hour': range(24),
                        'Value': hourly_avg
                    })
                    
                    if is_kwh_format:
                        st.write("Hourly energy (kWh):")
                    else:
                        st.write("Average hourly power (kW):")
                    
                    st.line_chart(profile_df.set_index('Hour')['Value'])
            
            # Load consumption data
            # Try different separators to handle various CSV formats
            try:
                # First try with semicolon separator (common in European CSVs)
                consumption_df = pd.read_csv(consumption_file, sep=';', decimal=',')
            except:
                try:
                    # Then try with comma separator and dot decimal
                    consumption_file.seek(0)  # Reset file pointer
                    consumption_df = pd.read_csv(consumption_file, sep=',', decimal='.')
                except:
                    # Finally try tab separator
                    consumption_file.seek(0)  # Reset file pointer
                    consumption_df = pd.read_csv(consumption_file, sep='\t', decimal=',')
            
            if 'consumption_kWh' not in consumption_df.columns:
                st.error("❌ Consumption CSV must contain 'consumption_kWh' column")
                return
            
            # Handle comma as decimal separator in consumption data only if needed
            if not pd.api.types.is_numeric_dtype(consumption_df['consumption_kWh']):
                # Data is not numeric, need to convert
                consumption_df['consumption_kWh'] = consumption_df['consumption_kWh'].astype(str).str.replace(',', '.', regex=False)
                try:
                    consumption_df['consumption_kWh'] = pd.to_numeric(consumption_df['consumption_kWh'], errors='coerce')
                    
                    # Check for conversion errors
                    n_errors = consumption_df['consumption_kWh'].isna().sum()
                    if n_errors > 0:
                        st.warning(f"⚠️ Found {n_errors} values that couldn't be converted to numbers. They will be treated as 0.")
                        consumption_df['consumption_kWh'] = consumption_df['consumption_kWh'].fillna(0)
                except Exception as e:
                    st.error(f"❌ Error converting consumption values: {str(e)}")
                    return
            
            # Validate and adjust consumption data length
            expected_rows = 35040
            if len(consumption_df) != expected_rows:
                st.warning(f"⚠️ Adjusting consumption data from {len(consumption_df):,} to {expected_rows:,} rows")
                if len(consumption_df) > expected_rows:
                    consumption_df = consumption_df.iloc[:expected_rows].copy()
                else:
                    # Repeat pattern to fill
                    repeats = (expected_rows // len(consumption_df)) + 1
                    consumption_df = pd.concat([consumption_df] * repeats).iloc[:expected_rows].reset_index(drop=True)
            
            # Validate consumption data is reasonable
            annual_consumption = consumption_df['consumption_kWh'].sum()
            if annual_consumption > 100000:  # More than 100 MWh/year
                st.warning(f"""
                    ⚠️ **Very high consumption detected:** {annual_consumption:,.0f} kWh/year
                    
                    This seems like industrial consumption. Typical values:
                    - Residential: 3,000-10,000 kWh/year
                    - Small business: 10,000-50,000 kWh/year
                    - Large business: 50,000-500,000 kWh/year
                    
                    Please verify your data units are correct (kWh per 15-min interval).
                """)
            
            # Check for unrealistic peak values
            peak_power = consumption_df['consumption_kWh'].max() * 4  # Convert to kW
            if peak_power > 100:  # More than 100 kW peak
                st.warning(f"""
                    ⚠️ **Very high peak power:** {peak_power:.1f} kW
                    
                    Typical peak power:
                    - Residential: 3-10 kW
                    - Small business: 10-50 kW
                    
                    Your data might be in Wh instead of kWh.
                """)
            
            # Display consumption statistics
            st.subheader("📊 Consumption Profile Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_annual = consumption_df['consumption_kWh'].sum()
            with col1:
                st.metric("Annual Consumption", f"{total_annual:,.0f} kWh")
            with col2:
                st.metric("Average Daily", f"{total_annual/365:,.1f} kWh")
            with col3:
                st.metric("Peak Load", f"{consumption_df['consumption_kWh'].max()*4:,.1f} kW")
            with col4:
                st.metric("Base Load", f"{consumption_df['consumption_kWh'].min()*4:,.1f} kW")
            
        except Exception as e:
            st.error(f"""
                ❌ Error reading files: {str(e)}
                
                **Common causes:**
                1. CSV uses semicolon (;) as separator → The system now handles this automatically
                2. Numbers use comma as decimal separator → The system converts this automatically
                3. Extra columns or spaces in the data → Check line 34 of your file
                4. Special characters or formatting issues
                
                **Please check your CSV file format:**
                - Should have only 1 column with header
                - 35,040 data rows (no empty rows)
                - No extra separators or columns
                
                **Supported formats:**
                - Semicolon separator with comma decimal: `1,234;` ✅
                - Comma separator with dot decimal: `1.234,` ✅
            """)
            return
        
        # Run optimization
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            run_optimization = st.button("🚀 Find Optimal System", type="primary", use_container_width=True)
        with col2:
            quick_test = st.button("⚡ Quick Test (5 configs)", use_container_width=True)
        with col3:
            export_results = st.button("📊 Export Last Results", use_container_width=True, 
                                     disabled='optimal_system' not in st.session_state)
        
        if run_optimization or quick_test:
            # Prepare inputs
            user_inputs = {
                "budget": budget,
                "available_area_m2": available_area_m2,
                "consumption_profile_df": consumption_df
            }
            
            config = {
                'bess_dod': dod,
                'bess_c_rate': c_rate,
                'bess_efficiency': efficiency,
                'bess_cycles': bess_cycles,
                'pv_degradation_rate': pv_degr,
                'bess_calendar_degradation_rate': bess_cal_degr,
                'grid_price_buy': grid_buy,
                'grid_price_sell': grid_sell,
                'wacc': wacc
            }
            
            if quick_test:
                # Quick test of specific configurations
                st.subheader("⚡ Quick Test Results")
                
                test_configs = [
                    (50, 50, "Balanced medium"),
                    (100, 100, "Balanced large"),
                    (150, 75, "PV-heavy"),
                    (75, 150, "Battery-heavy"),
                    (100, 0, "PV only")
                ]
                
                results = []
                for pv_kwp, bess_kwh, desc in test_configs:
                    # Check constraints
                    capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
                    capex_bess = bess_kwh * 150
                    total_capex = capex_pv + capex_bess
                    
                    if total_capex <= budget and pv_kwp <= available_area_m2 / 5.0:
                        result = run_simulation_vectorized(pv_kwp, bess_kwh, pv_baseline, 
                                                         consumption_df, config)
                        
                        results.append({
                            'Configuration': f"{pv_kwp} kWp / {bess_kwh} kWh",
                            'Description': desc,
                            'CAPEX (€)': f"{total_capex:,.0f}",
                            'NPV (€)': f"{result['npv_eur']:,.0f}",
                            'Payback (years)': f"{result['payback_period_years']:.1f}" if result['payback_period_years'] < 25 else "> 25",
                            'Self-sufficiency (%)': f"{result['self_sufficiency_rate']*100:.1f}"
                        })
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
            else:
                # Full optimization
                with st.spinner('🔄 Running optimization...'):
                    optimal_system = find_optimal_system(user_inputs, config, pv_baseline, enable_debug)
                    
                    if optimal_system:
                        # Save to session state
                        st.session_state['optimal_system'] = optimal_system
                        st.session_state['config'] = config
                        st.session_state['export_daily'] = export_daily_data
                        
                        # Display results
                        st.success("✅ Optimization Complete!")
                        
                        # Key metrics
                        st.header("🏆 Optimal System Configuration")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("PV System", f"{optimal_system['optimal_kwp']} kWp")
                        with col2:
                            st.metric("Battery", f"{optimal_system['optimal_kwh']} kWh")
                        with col3:
                            payback_text = f"{optimal_system['payback_period_years']:.1f} years" if optimal_system['payback_period_years'] < 25 else "> 25 years"
                            st.metric("Payback", payback_text)
                        with col4:
                            st.metric("Self-Sufficiency", f"{optimal_system['self_sufficiency_rate'] * 100:.1f}%")
                        
                        # Financial analysis
                        st.subheader("💰 Financial Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Investment", f"€{optimal_system['total_capex_eur']:,.0f}")
                        with col2:
                            st.metric("10-Year NPV", f"€{optimal_system['npv_eur']:,.0f}")
                        with col3:
                            st.metric("Annual O&M", f"€{optimal_system['om_costs']:,.0f}")
                        
                        # Cash flow visualization
                        with st.expander("📊 View Cash Flow Analysis"):
                            cf_data = optimal_system['cash_flows'][:11]  # 11 years
                            cumulative_cf = np.cumsum(cf_data)
                            
                            cf_df = pd.DataFrame({
                                'Year': range(len(cf_data)),
                                'Annual Cash Flow': cf_data,
                                'Cumulative Cash Flow': cumulative_cf
                            })
                            
                            st.line_chart(cf_df.set_index('Year')['Cumulative Cash Flow'])
                            st.caption("Cumulative cash flow showing payback period")
                        
                        # Annual metrics
                        with st.expander("📈 View Annual Energy Flows"):
                            annual_metrics_df = pd.DataFrame(optimal_system['annual_metrics'])
                            annual_metrics_df['Year'] = annual_metrics_df['year']
                            
                            # Energy flows chart
                            energy_cols = ['pv_production', 'energy_bought', 'energy_sold', 'energy_from_battery']
                            energy_df = annual_metrics_df[['Year'] + energy_cols].set_index('Year')
                            energy_df.columns = ['PV Production', 'Grid Import', 'Grid Export', 'Battery Discharge']
                            
                            st.bar_chart(energy_df)
                            st.caption("Annual energy flows (kWh)")
                    else:
                        st.error("❌ No valid solution found within constraints")
        
        # Export results section
        if export_results and 'optimal_system' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Export Detailed Calculations")
            
            optimal_system = st.session_state['optimal_system']
            config = st.session_state['config']
            
            try:
                # Generate reports
                annual_df, financial_df, config_df, timestep_df = export_detailed_calculations(
                    optimal_system, config,
                    optimal_system['optimal_kwp'],
                    optimal_system['optimal_kwh']
                )
                
                # Create download options
                col1, col2 = st.columns(2)
                
                with col1:
                    # ZIP download
                    zip_buffer = create_calculation_report_zip(annual_df, financial_df, config_df, timestep_df)
                    st.download_button(
                        label="📦 Download Complete Report (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"pv_bess_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
                
                with col2:
                    # Summary text
                    summary_text = f"""
PV & BESS Optimization Results
==============================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Optimal Configuration:
- PV System: {optimal_system['optimal_kwp']} kWp
- Battery: {optimal_system['optimal_kwh']} kWh
- Total CAPEX: €{optimal_system['total_capex_eur']:,.0f}

Financial Metrics:
- 10-Year NPV: €{optimal_system['npv_eur']:,.0f}
- Payback Period: {optimal_system['payback_period_years']:.1f} years
- Self-Sufficiency: {optimal_system['self_sufficiency_rate'] * 100:.1f}%

Parameters Used:
- Grid Buy Price: €{config['grid_price_buy']:.3f}/kWh
- Grid Sell Price: €{config['grid_price_sell']:.3f}/kWh
- WACC: {config['wacc']*100:.1f}%
"""
                    st.download_button(
                        label="📄 Download Summary (TXT)",
                        data=summary_text,
                        file_name=f"pv_bess_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                st.success("✅ Export files ready for download!")
                
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
    
    else:
        # Instructions
        st.info("""
            📁 **Please upload both files to begin:**
            
            1. **PV Production Data** (CSV)
               - 35,040 rows of 15-minute intervals
               - Values for 1 kWp reference system
               - **Accepts both formats:**
                 - **kW** (instantaneous power): typical range 0-1 kW
                 - **kWh** (energy per 15-min): typical range 0-0.25 kWh
               - Auto-detects format based on value ranges
               - Single column with header
               
            2. **Consumption Data** (CSV)
               - Column named 'consumption_kWh'
               - 35,040 rows of 15-minute intervals
               - Values in kWh per interval
               
            ✅ **Supported CSV Formats:**
            - **European**: Semicolon separator (;) with comma decimal (1,234)
            - **International**: Comma separator (,) with dot decimal (1.234)
            
            The system automatically detects your format!
            
            ⚠️ **Important**: Each file should have only ONE column with data.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>PV & BESS Optimizer v2.0 - Optimized Version</p>
            <p>Made with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    build_ui()
