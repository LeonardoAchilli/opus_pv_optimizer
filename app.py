import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import io

# ==============================================================================
# SECTION 1: BACKEND LOGIC
# ==============================================================================

# ==============================================================================
# SECTION 1: BACKEND LOGIC
# ==============================================================================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_pvgis_data(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetches 15-minute interval PV generation data from PVGIS v5.2 using JSON format."""
    # Updated to PVGIS API v5.2 with JSON output
    api_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    
    # Parameters for JSON output - much more reliable than CSV
    params = {
        'lat': latitude,
        'lon': longitude,
        'outputformat': 'json',  # Changed to JSON
        'pvcalculation': 1,
        'peakpower': 1,
        'loss': 0,
        'angle': 35,  # tilt angle
        'aspect': 0,  # azimuth (0 = south)
        'raddatabase': 'PVGIS-SARAH2',
        'startyear': 2020,
        'endyear': 2020,
        'usehorizon': 1,
        'mountingplace': 'free',
        'pvtechchoice': 'crystSi',
        'trackingtype': 0
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract hourly data
        if 'outputs' in data and 'hourly' in data['outputs']:
            hourly_data = data['outputs']['hourly']
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
            df = df.set_index('time')
            
            # Convert power from W to kW
            if 'P' in df.columns:
                df['P_kW'] = df['P'] / 1000.0
            else:
                return None
            
            # Resample to 15-minute intervals if we have hourly data
            if len(df) < 35040:  # If we have hourly data (8760 hours in a year)
                df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                # Ensure exactly 35040 points
                if len(df_resampled) > 35040:
                    df_resampled = df_resampled.iloc[:35040]
                elif len(df_resampled) < 35040:
                    # Pad with zeros if needed
                    padding = 35040 - len(df_resampled)
                    pad_index = pd.date_range(start=df_resampled.index[-1] + pd.Timedelta(minutes=15), 
                                             periods=padding, freq='15min')
                    pad_df = pd.DataFrame({'P_kW': [0] * padding}, index=pad_index)
                    df_resampled = pd.concat([df_resampled, pad_df])
                return df_resampled
            
            return df[['P_kW']]
            
        else:
            return None
            
    except requests.exceptions.HTTPError as e:
        # Try with ERA5 database as fallback
        params['raddatabase'] = 'PVGIS-ERA5'
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'outputs' in data and 'hourly' in data['outputs']:
                hourly_data = data['outputs']['hourly']
                df = pd.DataFrame(hourly_data)
                df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                df = df.set_index('time')
                
                if 'P' in df.columns:
                    df['P_kW'] = df['P'] / 1000.0
                    
                    if len(df) < 35040:
                        df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                        # Ensure exactly 35040 points
                        if len(df_resampled) > 35040:
                            df_resampled = df_resampled.iloc[:35040]
                        elif len(df_resampled) < 35040:
                            padding = 35040 - len(df_resampled)
                            pad_index = pd.date_range(start=df_resampled.index[-1] + pd.Timedelta(minutes=15), 
                                                     periods=padding, freq='15min')
                            pad_df = pd.DataFrame({'P_kW': [0] * padding}, index=pad_index)
                            df_resampled = pd.concat([df_resampled, pad_df])
                        return df_resampled
                    
                    return df[['P_kW']]
                else:
                    return None
            else:
                return None
                
        except Exception as e2:
            # Last resort: try without specifying database
            params.pop('raddatabase', None)
            
            try:
                response = requests.get(api_url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'outputs' in data and 'hourly' in data['outputs']:
                    hourly_data = data['outputs']['hourly']
                    df = pd.DataFrame(hourly_data)
                    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                    df = df.set_index('time')
                    
                    if 'P' in df.columns:
                        df['P_kW'] = df['P'] / 1000.0
                        
                        if len(df) < 35040:
                            df_resampled = df[['P_kW']].resample('15min').interpolate(method='linear')
                            # Ensure exactly 35040 points
                            if len(df_resampled) > 35040:
                                df_resampled = df_resampled.iloc[:35040]
                            elif len(df_resampled) < 35040:
                                padding = 35040 - len(df_resampled)
                                pad_index = pd.date_range(start=df_resampled.index[-1] + pd.Timedelta(minutes=15), 
                                                         periods=padding, freq='15min')
                                pad_df = pd.DataFrame({'P_kW': [0] * padding}, index=pad_index)
                                df_resampled = pd.concat([df_resampled, pad_df])
                            return df_resampled
                        
                        return df[['P_kW']]
                    
            except Exception as e3:
                return None
                
    except json.JSONDecodeError as e:
        return None
        
    except Exception as e:
        return None


def run_simulation_vectorized(pv_kwp, bess_kwh_nominal, pvgis_baseline_data, consumption_profile, config):
    """Vectorized simulation with exact energy flow formulas as specified."""
    # Extract configuration parameters
    dod = config['bess_dod']
    c_rate = config['bess_c_rate']
    efficiency = config['bess_efficiency']  # Single efficiency for both charge and discharge
    pv_degr_rate = config['pv_degradation_rate']
    calendar_degradation = config['bess_calendar_degradation_rate']
    tot_cycles = config['bess_cycles']
    
    # Calculate battery parameters
    kwh_netti = bess_kwh_nominal * dod  # Usable capacity
    max_charge_discharge_per_step = bess_kwh_nominal * c_rate / 4  # C-rate limit for 15-min step
    
    # Convert data to numpy arrays
    pv_base = pvgis_baseline_data['P_kW'].to_numpy(dtype=np.float32)
    cons = consumption_profile['consumption_kWh'].to_numpy(dtype=np.float32)
    
    # Debug: Check consumption data
    annual_consumption = cons.sum()
    if annual_consumption == 0:
        st.warning(f"⚠️ Warning: Annual consumption is zero! Check consumption data.")
        st.write(f"Consumption profile shape: {cons.shape}")
        st.write(f"First 10 values: {cons[:10]}")
    
    # Calculate base case annual cost - FIXED
    base_case_annual_cost = annual_consumption * config['grid_price_buy']
    
    # Simulation parameters
    steps_per_year = 35040  # 96 steps/day * 365 days
    
    # Initialize tracking variables
    annual_metrics = []
    total_grid_import = 0
    total_consumption = annual_consumption * 5  # 5 years total
    
    # Pre-calculate PV degradation factors
    pv_degradation_factors = (1 - pv_degr_rate) ** np.arange(5)
    
    # Run 5-year simulation
    for year in range(5):
        # Apply PV degradation
        pv_production = pv_base * pv_kwp * pv_degradation_factors[year] * 0.25  # kWh per 15-min
        
        # Initialize arrays for this year
        soc = np.zeros(steps_per_year + 1, dtype=np.float32)
        soh = np.zeros(steps_per_year + 1, dtype=np.float32)
        kwh_scaricati = np.zeros(steps_per_year, dtype=np.float32)
        kwh_caricati = np.zeros(steps_per_year, dtype=np.float32)
        immissione = np.zeros(steps_per_year, dtype=np.float32)
        acquisto = np.zeros(steps_per_year, dtype=np.float32)
        
        # Set initial SoH
        if year == 0:
            soh[0] = 1.0
        else:
            soh[0] = annual_metrics[-1]['final_soh']
        
        # Simulate each 15-minute step
        for t in range(steps_per_year):
            produzione = pv_production[t]
            consumo = cons[t]
            
            # Current max SoC based on battery health
            max_soc_current = kwh_netti * soh[t]
            
            # 1. Calculate kWh_scaricati (battery discharge)
            if consumo > produzione:
                # Energy needed from battery
                energy_deficit = (consumo - produzione) / efficiency
                # Apply limits: available SoC and C-rate
                kwh_scaricati[t] = min(energy_deficit, soc[t], max_charge_discharge_per_step)
            else:
                kwh_scaricati[t] = 0
            
            # 2. Calculate grid export BEFORE updating SoC (Immissione)
            if produzione > consumo:
                # Energy that cannot be stored in battery
                excess_after_battery = soc[t] + produzione - consumo - max_soc_current
                immissione[t] = max(0, excess_after_battery)
            else:
                immissione[t] = 0
            
            # 3. Calculate grid import (Acquisto)
            # Formula as specified: max(0, consumo - produzione - SoC(t-1))
            acquisto[t] = max(0, consumo - produzione - soc[t])
            
            # 4. Now update SoC and track energy charged
            if produzione >= consumo:
                # Excess energy: charge battery
                delta_soc = (produzione - consumo) * efficiency
                # Actual charge limited by capacity and C-rate
                actual_charge = min(delta_soc, max_soc_current - soc[t], max_charge_discharge_per_step)
                soc[t+1] = soc[t] + actual_charge
                # Track energy sent to battery (before efficiency)
                kwh_caricati[t] = actual_charge / efficiency
            else:
                # Deficit: discharge battery (already calculated as kwh_scaricati)
                soc[t+1] = soc[t] - kwh_scaricati[t]
                kwh_caricati[t] = 0
            
            # 5. Calculate SoH degradation
            if t > 0:  # Use previous step's discharge for degradation
                cycle_degradation = (kwh_scaricati[t-1] / bess_kwh_nominal) * (0.2 * 1.15 / tot_cycles)
            else:
                cycle_degradation = 0
            
            calendar_deg_per_step = calendar_degradation / steps_per_year
            soh[t+1] = soh[t] - calendar_deg_per_step - cycle_degradation
            soh[t+1] = max(0, soh[t+1])  # Ensure SoH doesn't go negative
        
        # Calculate annual totals
        yearly_pv_production = pv_production.sum()
        yearly_consumption = annual_consumption  # Use the actual annual consumption
        yearly_energy_bought = acquisto.sum()
        yearly_energy_sold = immissione.sum()
        yearly_battery_discharge = (kwh_scaricati * efficiency).sum()  # Energy delivered to load
        yearly_battery_charge = kwh_caricati.sum()  # Energy sent to battery
        
        yearly_self_consumption = yearly_consumption - yearly_energy_bought
        
        # Store detailed metrics
        annual_metrics.append({
            'year': year + 1,
            'pv_production': yearly_pv_production,
            'consumption': yearly_consumption,
            'energy_bought': yearly_energy_bought,
            'energy_sold': yearly_energy_sold,
            'energy_to_battery': yearly_battery_charge,
            'energy_from_battery': yearly_battery_discharge,
            'self_consumption': yearly_self_consumption,
            'self_sufficiency': yearly_self_consumption / yearly_consumption if yearly_consumption > 0 else 0,
            'final_soh': soh[steps_per_year],
            'avg_soc': soc[1:steps_per_year+1].mean(),
            'max_soc': soc[1:steps_per_year+1].max(),
            'min_soc': soc[1:steps_per_year+1].min()
        })
        
        total_grid_import += yearly_energy_bought
    
    # Calculate financial metrics
    capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
    capex_bess = bess_kwh_nominal * 150
    total_capex = capex_pv + capex_bess
    
    # O&M costs
    om_pv = (12 - 0.01 * pv_kwp) * pv_kwp
    om_bess = 1500 + (capex_bess * 0.015)
    total_om = om_pv + om_bess
    
    # Calculate cash flows using differential formula
    cash_flows = [-total_capex]  # Year 0
    
    for metrics in annual_metrics:
        # CF = (consumo × buy) - (energia_prelevata × buy) + (energia_immessa × sell) - O&M
        base_cost = metrics['consumption'] * config['grid_price_buy']
        energy_cost = metrics['energy_bought'] * config['grid_price_buy']
        energy_revenue = metrics['energy_sold'] * config['grid_price_sell']
        
        annual_cf = base_cost - energy_cost + energy_revenue - total_om
        cash_flows.append(annual_cf)
    
    # Project cash flows for years 6-10
    if len(cash_flows) > 1:
        last_cf = cash_flows[-1]
        for _ in range(5):
            projected_cf = last_cf * 0.97  # 3% degradation
            cash_flows.append(projected_cf)
            last_cf = projected_cf
    
    # Calculate NPV
    wacc = config['wacc']
    npv = sum(cash_flows[i] / ((1 + wacc) ** i) for i in range(len(cash_flows)))
    
    # Calculate payback period
    cumulative_cf = 0
    payback_period = float('inf')
    for i, cf in enumerate(cash_flows):
        cumulative_cf += cf
        if cumulative_cf > 0 and i > 0:
            prev_cumulative = cumulative_cf - cf
            fraction = -prev_cumulative / cf
            payback_period = i - 1 + fraction
            break
    
    # Calculate base case NPV
    base_case_npv = -sum(base_case_annual_cost / ((1 + wacc) ** (i + 1)) for i in range(10))
    
    # Overall self-sufficiency rate
    self_sufficiency_rate = (total_consumption - total_grid_import) / total_consumption if total_consumption > 0 else 0
    
    return {
        "npv_eur": npv,
        "base_case_npv_eur": base_case_npv,
        "payback_period_years": payback_period,
        "total_capex_eur": total_capex,
        "capex_pv": capex_pv,
        "capex_bess": capex_bess,
        "self_sufficiency_rate": self_sufficiency_rate,
        "final_soh_percent": annual_metrics[-1]['final_soh'] * 100,
        "om_costs": total_om,
        "om_pv": om_pv,
        "om_bess": om_bess,
        "base_case_annual_cost": base_case_annual_cost,
        "annual_metrics": annual_metrics,
        "cash_flows": cash_flows,
        "annual_consumption": annual_consumption  # Add for debugging
    }

def find_optimal_system(user_inputs, config, pvgis_baseline):
    """Finds the optimal PV and BESS combination with improved search algorithm."""
    # Calculate maximum feasible sizes
    max_kwp_from_area = user_inputs['available_area_m2'] / 5.0  # 5 m²/kWp
    max_kwp_from_budget = user_inputs['budget'] / 650  # Minimum cost estimate
    max_kwp = min(max_kwp_from_area, max_kwp_from_budget)
    
    max_kwh = user_inputs['budget'] / 150  # Minimum battery cost
    
    # Ensure we have valid search ranges
    if max_kwp < 5:
        st.error(f"Budget too low! Minimum PV system costs ~€3,250 (5 kWp). Your budget allows max {max_kwp:.1f} kWp")
        return None
    
    # Define search ranges with adaptive step sizes
    kwp_step = max(5, int(max_kwp / 20))  # More granular search
    kwh_step = max(5, int(max_kwh / 20))
    
    pv_search_range = range(kwp_step, int(max_kwp) + kwp_step, kwp_step)
    bess_search_range = range(0, int(max_kwh) + kwh_step, kwh_step)
    
    # Debug information
    st.info(f"Searching PV sizes: {kwp_step} to {int(max_kwp)} kWp (step: {kwp_step})")
    st.info(f"Searching battery sizes: 0 to {int(max_kwh)} kWh (step: {kwh_step})")
    
    # Initialize search variables
    best_result = None
    min_payback = float('inf')
    results_matrix = []
    valid_solutions = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    total_sims = len(pv_search_range) * len(bess_search_range)
    sim_count = 0
    update_frequency = max(1, total_sims // 20)  # Update progress bar 20 times max
    
    # Search for optimal combination
    for pv_kwp in pv_search_range:
        battery_found_within_budget = False
        
        for bess_kwh in bess_search_range:
            sim_count += 1
            
            # Update progress bar less frequently
            if sim_count % update_frequency == 0 or sim_count == total_sims:
                progress_bar.progress(min(sim_count / total_sims, 1.0))
            
            # Check budget constraint with correct CAPEX calculation
            current_capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
            current_capex_bess = bess_kwh * 150
            
            if (current_capex_pv + current_capex_bess) > user_inputs['budget']:
                break  # Skip remaining battery sizes for this PV size
            
            battery_found_within_budget = True
            valid_solutions += 1
            
            # Run vectorized simulation
            result = run_simulation_vectorized(pv_kwp, bess_kwh, pvgis_baseline, 
                                             user_inputs['consumption_profile_df'], config)
            
            # Store result for analysis
            result['pv_kwp'] = pv_kwp
            result['bess_kwh'] = bess_kwh
            results_matrix.append(result)
            
            # Track best result - prioritize by NPV if all paybacks are infinite
            if result['payback_period_years'] < min_payback:
                min_payback = result['payback_period_years']
                best_result = result.copy()
                best_result['optimal_kwp'] = pv_kwp
                best_result['optimal_kwh'] = bess_kwh
            elif min_payback == float('inf') and best_result is None:
                # If no finite payback found yet, use the one with best NPV
                best_result = result.copy()
                best_result['optimal_kwp'] = pv_kwp
                best_result['optimal_kwh'] = bess_kwh
            elif min_payback == float('inf') and result['npv_eur'] > best_result.get('npv_eur', -float('inf')):
                # Update if this has better NPV
                best_result = result.copy()
                best_result['optimal_kwp'] = pv_kwp
                best_result['optimal_kwh'] = bess_kwh
        
        # If no battery size fits within budget for this PV size, skip larger PV sizes
        if not battery_found_within_budget and bess_search_range:
            break
    
    progress_bar.empty()
    
    # Show summary with more details
    if valid_solutions > 0:
        st.success(f"✅ Analyzed {valid_solutions} valid configurations")
        
        # Show sample results for debugging
        if results_matrix and st.checkbox("Show sample results for debugging"):
            st.write("**First 5 configurations analyzed:**")
            for i, res in enumerate(results_matrix[:5]):
                with st.expander(f"Config {i+1}: PV={res['pv_kwp']}kWp, BESS={res['bess_kwh']}kWh"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Financial:**")
                        st.write(f"- CAPEX: €{res['total_capex_eur']:,.0f}")
                        st.write(f"- NPV (Differential): €{res['npv_eur']:,.0f}")
                        st.write(f"- Payback: {res['payback_period_years']:.1f} years")
                        st.write(f"- Annual O&M: €{res['om_costs']:,.0f}")
                    with col2:
                        st.write(f"**Performance:**")
                        st.write(f"- Self-sufficiency: {res['self_sufficiency_rate']*100:.1f}%")
                        if 'annual_metrics' in res and res['annual_metrics']:
                            year1 = res['annual_metrics'][0]
                            st.write(f"- PV Production Y1: {year1['pv_production']:,.0f} kWh")
                            st.write(f"- Grid Import Y1: {year1['energy_bought']:,.0f} kWh")
                            st.write(f"- Grid Export Y1: {year1['energy_sold']:,.0f} kWh")
                    
                    # Show Year 1 cash flow calculation
                    if 'cash_flows' in res and len(res['cash_flows']) > 1:
                        st.write("**Year 1 Cash Flow Breakdown:**")
                        cf1 = res['cash_flows'][1]
                        if 'annual_metrics' in res:
                            m1 = res['annual_metrics'][0]
                            base = m1['consumption'] * config['grid_price_buy']
                            cost = m1['energy_bought'] * config['grid_price_buy']
                            revenue = m1['energy_sold'] * config['grid_price_sell']
                            
                            st.write(f"Base cost: €{base:,.0f}")
                            st.write(f"- Energy cost: €{cost:,.0f}")
                            st.write(f"+ Energy revenue: €{revenue:,.0f}")
                            st.write(f"- O&M: €{res['om_costs']:,.0f}")
                            st.write(f"= Cash Flow: €{cf1:,.0f}")
    else:
        st.warning("No valid configurations found within constraints")
    
    # Add results matrix to best result for visualization
    if best_result:
        best_result['all_results'] = results_matrix
    
    return best_result


def build_ui():
    """Streamlit UI with enhanced features and error handling."""
    st.set_page_config(
        page_title="PV & BESS Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stMetric {
            background-color: #262626;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("⚡ Optimal PV & BESS Sizing Calculator")
    st.markdown("""
        ### Find the perfect solar + battery system for your needs
        This tool optimizes Photovoltaic (PV) and Battery Energy Storage System (BESS) sizing 
        based on your consumption data, location, and budget constraints.
    """)
    
    # Sidebar inputs
    with st.sidebar:
        st.header("🔧 Configuration")
        
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
            help="Total roof or ground area available for solar panels"
        )
        
        st.subheader("2. Location")
        st.caption("PVGIS covers Europe, Africa, and most of Asia")
        
        # Location presets
        location_preset = st.selectbox(
            "Quick location selection:",
            ["Custom", "Rome, Italy", "Berlin, Germany", "Madrid, Spain", 
             "Athens, Greece", "Cairo, Egypt", "Istanbul, Turkey"]
        )
        
        # Set coordinates based on selection
        if location_preset == "Rome, Italy":
            default_lat, default_lon = 41.9028, 12.4964
        elif location_preset == "Berlin, Germany":
            default_lat, default_lon = 52.5200, 13.4050
        elif location_preset == "Madrid, Spain":
            default_lat, default_lon = 40.4168, -3.7038
        elif location_preset == "Athens, Greece":
            default_lat, default_lon = 37.9838, 23.7275
        elif location_preset == "Cairo, Egypt":
            default_lat, default_lon = 30.0444, 31.2357
        elif location_preset == "Istanbul, Turkey":
            default_lat, default_lon = 41.0082, 28.9784
        else:
            default_lat, default_lon = 41.9028, 12.4964
        
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=default_lat, format="%.4f", 
                                min_value=-90.0, max_value=90.0,
                                help="Decimal degrees North")
        with col2:
            lon = st.number_input("Longitude", value=default_lon, format="%.4f",
                                min_value=-180.0, max_value=180.0,
                                help="Decimal degrees East")
        
        # Test location button
        if st.button("🧪 Test Location", help="Check if solar data is available for this location"):
            with st.spinner("Testing location..."):
                test_data = get_pvgis_data(lat, lon)
                if test_data is not None and not test_data.empty:
                    st.success(f"✅ Location valid! Solar data available for {lat:.2f}°N, {lon:.2f}°E")
                else:
                    st.error("❌ No solar data available for this location")
        
        st.subheader("3. Consumption Profile")
        uploaded_file = st.file_uploader(
            "Upload 15-minute consumption data (CSV)",
            type="csv",
            help="""
            The CSV must contain:
            - Column named 'consumption_kWh'
            - 35,040 rows (1 year of 15-min data)
            - Values in kWh per 15-minute interval
            """
        )
        
        # Advanced settings (collapsible)
        with st.expander("⚙️ Advanced Settings"):
            st.write("**Electricity Prices**")
            st.info("💡 Italian market avg: Buy €0.30-0.40/kWh, Sell €0.10-0.15/kWh")
            grid_buy = st.number_input("Grid Buy Price (€/kWh)", value=0.35, format="%.3f", min_value=0.10, max_value=1.00)
            grid_sell = st.number_input("Grid Sell Price (€/kWh)", value=0.12, format="%.3f", min_value=0.01, max_value=0.50)
            st.warning("⚠️ Electricity prices have a major impact on results. Use accurate local prices!")
            
            st.write("**Financial Parameters**")
            wacc = st.slider("WACC (%)", min_value=1, max_value=15, value=7) / 100
            
            st.write("**Battery Parameters**")
            dod = st.slider("Depth of Discharge (%)", 70, 95, 85) / 100
            c_rate = st.slider("C-Rate", 0.3, 1.0, 0.7, 0.1)
            bess_cycles = st.slider("Battery Life Cycles", 6000, 10000, 7000, 100)
            
            st.write("**Technical Parameters**")
            efficiency = st.slider("Charge/Discharge Efficiency (%)", 85, 98, 95) / 100
            pv_degr = st.slider("PV Annual Degradation (%)", 0.2, 2.0, 1.0) / 100
            bess_cal_degr = st.slider("Battery Calendar Degradation (%/year)", 0.5, 3.0, 1.5) / 100
    
    expected_rows = 35040
    # Main content area
    if uploaded_file is not None:
        try:
            # Read and validate consumption data
            consumption_df = pd.read_csv(uploaded_file)
            
            if 'consumption_kWh' not in consumption_df.columns:
                st.error("❌ Error: CSV must contain a column named 'consumption_kWh'.")
                return
            
            actual_rows = len(consumption_df)
            
            # Validate consumption data length
            if len(consumption_df) != expected_rows:
                st.warning(f"""
                    ⚠️ Warning: Expected {expected_rows:,} rows but found {actual_rows:,} rows.
                    The simulation assumes 1 year of 15-minute data.
                    Adjusting data to match expected length...
                """)
                # Trim or pad the data to match expected length
                if len(consumption_df) > expected_rows:
                    consumption_df = consumption_df.iloc[:expected_rows].copy()
                    st.info(f"✂️ Trimmed data to {expected_rows:,} rows")
                else:
                    # Repeat the pattern to fill missing data
                    repeats_needed = (expected_rows // len(consumption_df)) + 1
                    consumption_df = pd.concat([consumption_df] * repeats_needed).iloc[:expected_rows].reset_index(drop=True)
                    st.info(f"📋 Repeated pattern to reach {expected_rows:,} rows")
            
            # Display consumption statistics
            st.subheader("📊 Consumption Profile Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_annual = consumption_df['consumption_kWh'].sum()
            with col1:
                st.metric("Annual Consumption", f"{total_annual:,.0f} kWh")
            with col2:
                st.metric("Average Daily", f"{total_annual/365:,.1f} kWh")
            with col3:
                st.metric("Peak 15-min", f"{consumption_df['consumption_kWh'].max():,.2f} kWh")
            with col4:
                st.metric("Min 15-min", f"{consumption_df['consumption_kWh'].min():,.2f} kWh")
            
            # Visualization of consumption pattern
            with st.expander("📈 View Consumption Pattern"):
                # Daily average profile
                consumption_df['hour'] = range(len(consumption_df))
                consumption_df['hour'] = (consumption_df['hour'] % 96) / 4  # Convert to hours
                hourly_avg = consumption_df.groupby('hour')['consumption_kWh'].mean() * 4  # kW
                
                st.line_chart(
                    hourly_avg,
                    use_container_width=True,
                    height=300
                )
                st.caption("Average daily consumption profile (kW)")
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            return
        
        # Run optimization button
        if st.button("🚀 Find Optimal System", type="primary", use_container_width=True):
            # Prepare inputs
            user_inputs = {
                "budget": budget,
                "available_area_m2": available_area_m2,
                "consumption_profile_df": consumption_df
            }
            
            config = {
                'bess_dod': dod if 'dod' in locals() else 0.85,
                'bess_c_rate': c_rate if 'c_rate' in locals() else 0.7,
                'bess_efficiency': efficiency if 'efficiency' in locals() else 0.95,
                'bess_cycles': bess_cycles if 'bess_cycles' in locals() else 7000,
                'pv_degradation_rate': pv_degr if 'pv_degr' in locals() else 0.01,
                'bess_calendar_degradation_rate': bess_cal_degr if 'bess_cal_degr' in locals() else 0.015,
                'grid_price_buy': grid_buy if 'grid_buy' in locals() else 0.35,
                'grid_price_sell': grid_sell if 'grid_sell' in locals() else 0.12,
                'wacc': wacc if 'wacc' in locals() else 0.07
            }
            
            # Run optimization
            optimal_system = None
            with st.spinner('🔄 Fetching solar data and running optimization...'):
                # Get PVGIS data
                pvgis_baseline = get_pvgis_data(lat, lon)
                
                if pvgis_baseline is not None and not pvgis_baseline.empty:
                    st.success("✅ Solar data retrieved successfully!")
                    
                    # Debug info
                    with st.expander("🔍 Debug Information & Calculation Logic"):
                        st.write(f"PVGIS data points: {len(pvgis_baseline)}")
                        st.write(f"Consumption data points: {len(consumption_df)}")
                        st.write(f"Budget: €{budget:,.0f}")
                        st.write(f"Available area: {available_area_m2} m²")
                        st.write(f"Max PV from area: {available_area_m2 / 5.0:.1f} kWp")
                        st.write(f"Max PV from budget: {budget / 650:.1f} kWp")
                        
                        st.markdown("---")
                        st.write("**Simulation Logic (Step-by-Step):**")
                        
                        st.write("1. **For each 15-minute interval:**")
                        st.code("""
# Step-by-step energy flow calculation

# 1. Battery discharge (if consumption > production)
if consumo > produzione:
    kwh_scaricati = min((consumo - produzione)/efficienza, SoC(t-1), C_rate*capacity/4)
else:
    kwh_scaricati = 0

# 2. State of Charge update
if produzione >= consumo:
    delta_SoC = (produzione - consumo) * efficienza
else:
    delta_SoC = (produzione - consumo) / efficienza
    
SoC = min(max(SoC(t-1) + delta_SoC, 0), kWh_netti * SoH(t-1))

# 3. Grid export (excess that cannot be stored)
if produzione > consumo:
    immissione = max(0, SoC(t-1) + produzione - consumo - kWh_netti * SoH(t-1))
else:
    immissione = 0

# 4. Grid import (deficit not covered by battery)
acquisto = max(0, consumo - produzione - SoC(t-1))

# 5. SoH degradation
SoH = SoH(t-1) - calendar_deg/35040 - (kwh_scaricati(t-1)/capacity) * (0.2 * 1.15 / cycles)
""")
                        
                        st.write("2. **Annual aggregation:**")
                        st.code("""
annual_grid_import = sum(grid_import for all intervals)
annual_grid_export = sum(grid_export for all intervals)
annual_consumption = sum(consumption for all intervals)
""")
                        
                        st.write("3. **Cash Flow calculation (your Excel formula):**")
                        st.code("""
CF[0] = -CAPEX
CF[n] = -O&M + (grid_export × sell_price) - (grid_import × buy_price) + (consumption × buy_price)

Which equals:
CF[n] = (consumption × buy_price) - (grid_import × buy_price) + (grid_export × sell_price) - O&M
""")
                        
                        st.write("4. **NPV calculation:**")
                        st.code("NPV = Σ(CF[i] / (1 + WACC)^i) for i = 0 to 10")
                        
                        st.write("5. **Cost formulas used:**")
                        st.code("""
# CAPEX
CAPEX_PV = pv_kwp × (600 + 600 × exp(-pv_kwp / 290))  # Non-linear pricing
CAPEX_BESS = bess_kwh × 150  # Linear pricing

# O&M
O&M_PV = (12 - 0.01 × pv_kwp) × pv_kwp  # Economies of scale
O&M_BESS = 1500 + (CAPEX_BESS × 0.015)  # Fixed + percentage
""")
                        
                        st.write("6. **Degradation models:**")
                        st.code("""
# PV degradation (annual)
pv_output_year_n = pv_output_year_1 × (1 - pv_degradation_rate)^(n-1)

# Battery degradation (per step)
calendar_degradation_per_step = annual_calendar_deg / 35040
cycle_degradation_per_step = (kwh_scaricati / battery_capacity) × (0.2 × 1.15 / total_cycles)
SoH_new = SoH_old - calendar_degradation_per_step - cycle_degradation_per_step

# Note: Cycle degradation uses discharge from PREVIOUS step (t-1)
""")
                        
                    # Ensure data consistency
                    if len(pvgis_baseline) != len(consumption_df):
                        st.error(f"Data mismatch: PVGIS has {len(pvgis_baseline)} points, consumption has {len(consumption_df)} points")
                        # Try to align data
                        min_len = min(len(pvgis_baseline), len(consumption_df))
                        pvgis_baseline = pvgis_baseline.iloc[:min_len]
                        consumption_df = consumption_df.iloc[:min_len]
                        st.warning(f"Trimmed both datasets to {min_len} points")
                    
                    # Run optimization with error handling
                    try:
                        optimal_system = find_optimal_system(user_inputs, config, pvgis_baseline)
                        
                        if optimal_system is None:
                            st.error("❌ No valid solution found!")
                            st.warning("""
                            **Possible reasons:**
                            - Budget too low for any viable system
                            - No combination meets the constraints
                            
                            **Try:**
                            - Increasing the budget (€150,000+)
                            - Checking your consumption data
                            """)
                        
                    except Exception as e:
                        st.error(f"❌ Error during optimization: {str(e)}")
                        with st.expander("Show full error"):
                            st.exception(e)
                
                else:
                    st.error("❌ Could not retrieve solar data. Please check your location or try again later.")
                    st.info("""
                    💡 **Tips:**
                    - PVGIS covers Europe, Africa, and most of Asia
                    - Americas and Oceania are not covered
                    - Try coordinates like: Rome (41.9, 12.5), Berlin (52.5, 13.4), Cairo (30.0, 31.2)
                    """)
            
            # Display results
            if optimal_system:
                st.success("✅ Optimization Complete!")
                st.markdown("---")
                
                # Key results
                st.header("🏆 Optimal System Configuration")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "PV System Size",
                        f"{optimal_system['optimal_kwp']} kWp",
                        help="Optimal photovoltaic system capacity"
                    )
                
                with col2:
                    st.metric(
                        "Battery Size",
                        f"{optimal_system['optimal_kwh']} kWh",
                        help="Optimal battery storage capacity"
                    )
                
                with col3:
                    if optimal_system['payback_period_years'] == float('inf'):
                        st.metric(
                            "Payback Period",
                            "> 25 years",
                            help="System doesn't achieve positive payback within analysis period"
                        )
                    else:
                        st.metric(
                            "Payback Period",
                            f"{optimal_system['payback_period_years']:.1f} years",
                            help="Time to recover initial investment"
                        )
                
                with col4:
                    st.metric(
                        "Self-Sufficiency",
                        f"{optimal_system['self_sufficiency_rate'] * 100:.1f}%",
                        help="Percentage of consumption covered by PV+BESS"
                    )
                
                # Financial details with base case comparison
                st.subheader("💰 Financial Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Investment",
                        f"€{optimal_system['total_capex_eur']:,.0f}",
                        help="Total capital expenditure"
                    )
                
                with col2:
                    st.metric(
                        "10-Year NPV (Differential)",
                        f"€{optimal_system['npv_eur']:,.0f}",
                        help="Net Present Value compared to buying all energy from grid"
                    )
                
                with col3:
                    st.metric(
                        "Annual O&M",
                        f"€{optimal_system['om_costs']:,.0f}",
                        help="Operation & Maintenance costs per year"
                    )
                
                # Base case comparison
                with st.expander("📊 Base Case Comparison"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Base Case Annual Cost",
                            f"€{optimal_system.get('base_case_annual_cost', 0):,.0f}",
                            help="Annual cost of buying all energy from grid"
                        )
                    
                    with col2:
                        st.metric(
                            "Base Case 10-Year NPV",
                            f"€{optimal_system.get('base_case_npv_eur', 0):,.0f}",
                            help="NPV of continuing to buy all energy from grid"
                        )
                    
                    with col3:
                        # Il benefit è già l'NPV differenziale
                        if optimal_system.get('base_case_annual_cost', 0) > 0:
                            roi = (optimal_system['npv_eur'] / optimal_system['total_capex_eur']) * 100
                            st.metric(
                                "ROI (10 years)",
                                f"{roi:.1f}%",
                                help="Return on Investment over 10 years"
                            )
                        else:
                            st.metric(
                                "Total Benefit",
                                f"€{optimal_system['npv_eur']:,.0f}",
                                help="Total financial benefit vs base case"
                            )
                    
                    st.info("""
                    **NPV Calculation Method (Differential Approach):**
                    
                    The NPV is calculated using differential cash flows compared to the base case:
                    
                    📊 **Cash Flow Formula:**
                    - Year 0: CF₀ = -CAPEX
                    - Years 1-10: CFₙ = (Base Cost - System Cost) - O&M
                    
                    Where:
                    - Base Cost = Annual consumption × Grid buy price
                    - System Cost = Energy bought × Buy price - Energy sold × Sell price
                    - NPV = Σ(CFₙ / (1 + WACC)ⁿ)
                    
                    This represents the net financial benefit of installing the PV+BESS system
                    compared to continuing to buy all energy from the grid.
                    """)
                
                # System health
                st.subheader("🔋 System Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Battery Health (Year 5)",
                        f"{optimal_system['final_soh_percent']:.1f}%",
                        help="Battery State of Health after 5 years"
                    )
                
                # Annual savings chart
                  # Annual savings e cash flow analysis
                if 'cash_flows' in optimal_system and len(optimal_system['cash_flows']) > 1:
                    # La metrica dei risparmi ora usa i cash flow. CF[0] è il CAPEX, CF[1] è il primo anno.
                    year_1_net_benefit = optimal_system['cash_flows'][1] 
                    year_5_net_benefit = optimal_system['cash_flows'][5]

                    with col2:
                        st.metric(
                            "Year 1 Net Benefit",
                            f"€{year_1_net_benefit:,.0f}",
                            f"{year_5_net_benefit - year_1_net_benefit:+,_} vs Year 5"
                        )

                    # Visualizzazione dei flussi di cassa
                    with st.expander("📊 View Cash Flow Analysis"):
                        # I cash_flows sono già calcolati per 10 anni (11 elementi: anno 0 + anni 1-10)
                        cf_data = optimal_system['cash_flows']
                        
                        # Crea DataFrame per la visualizzazione
                        years = list(range(0, len(cf_data)))
                        cumulative_cf = np.cumsum(cf_data)
                        
                        cf_df = pd.DataFrame({
                            'Year': years,
                            'Annual Cash Flow (€)': cf_data,
                            'Cumulative Cash Flow (€)': cumulative_cf
                        }).set_index('Year')

                        st.write("**Differential Cash Flow Analysis (10 Years):**")
                        st.dataframe(
                            cf_df.style.format({
                                'Annual Cash Flow (€)': '€{:,.0f}',
                                'Cumulative Cash Flow (€)': '€{:,.0f}'
                            }),
                            use_container_width=True
                        )

                        st.line_chart(
                            cf_df['Cumulative Cash Flow (€)'],
                            height=300
                        )
                        st.caption("Cumulative cash flow showing payback period (when line crosses zero)")

                        # Metriche chiave
                        st.write("**Key Financial Metrics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Investment", f"€{optimal_system['total_capex_eur']:,.0f}")
                        with col2:
                            # Calcola il beneficio medio sui primi 5 anni operativi (da indice 1 a 5)
                            avg_benefit = sum(cf_data[1:6])/5
                            st.metric("Avg Annual Benefit (Y1-5)", f"€{avg_benefit:,.0f}")
                        with col3:
                            if optimal_system['payback_period_years'] != float('inf'):
                                st.metric("Break-even Point", f"{optimal_system['payback_period_years']:.1f} years")
                            else:
                                st.metric("Break-even Point", "> 10 years")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if optimal_system['payback_period_years'] == float('inf'):
                    st.warning("""
                    ⚠️ **No positive payback achieved** - The system doesn't pay for itself within the analysis period.
                    
                    This might be due to:
                    - Low electricity prices
                    - Low consumption relative to investment
                    - High system costs
                    
                    Consider:
                    - Checking if electricity prices are correct for your area
                    - Waiting for battery prices to decrease
                    - Installing PV-only system without battery
                    - Increasing electricity sell price if you have special feed-in tariffs
                    """)
                elif optimal_system['optimal_kwh'] == 0:
                    st.info("🔍 **No battery recommended** - The analysis suggests a PV-only system provides the best financial return for your situation.")
                elif optimal_system['payback_period_years'] > 8:
                    st.warning("⚠️ **Long payback period** - Consider if the environmental benefits justify the investment.")
                else:
                    st.success("✅ **Excellent investment** - The system shows strong financial returns with reasonable payback period.")
                
                # Export results
                st.markdown("---")
                payback_text = f"{optimal_system['payback_period_years']:.1f} years" if optimal_system['payback_period_years'] != float('inf') else "> 25 years"
                base_case_info = f"""
                Base Case Analysis (No PV/BESS System):
                - Annual electricity cost: €{optimal_system.get('base_case_annual_cost', 0):,.0f}
                - 10-year NPV of costs: €{optimal_system.get('base_case_npv_eur', 0):,.0f}
                
                The NPV shown above (€{optimal_system['npv_eur']:,.0f}) represents the net financial
                benefit of installing the PV+BESS system compared to continuing to buy all energy
                from the grid (differential NPV).
                """
                
                results_text = f"""
                PV & BESS Optimization Results
                ==============================
                Location: {lat:.4f}°N, {lon:.4f}°E
                Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
                
                Optimal Configuration:
                - PV System: {optimal_system['optimal_kwp']} kWp
                - Battery: {optimal_system['optimal_kwh']} kWh
                - Total CAPEX: €{optimal_system['total_capex_eur']:,.0f}
                
                Financial Metrics (Differential Analysis):
                - Payback Period: {payback_text}
                - 10-Year NPV: €{optimal_system['npv_eur']:,.0f}
                - Annual O&M: €{optimal_system['om_costs']:,.0f}
                
                {base_case_info}
                
                Performance:
                - Self-Sufficiency: {optimal_system['self_sufficiency_rate'] * 100:.1f}%
                - Battery SoH (Year 5): {optimal_system['final_soh_percent']:.1f}%
                
                Economic Parameters Used:
                - Grid Buy Price: €{config['grid_price_buy']:.3f}/kWh
                - Grid Sell Price: €{config['grid_price_sell']:.3f}/kWh
                - WACC: {config['wacc']*100:.1f}%
                
                Note: NPV is calculated using differential cash flows compared to the base case
                of continuing to purchase all energy from the grid.
                """
                
                st.download_button(
                    label="📥 Download Results",
                    data=results_text,
                    file_name=f"pv_bess_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
    else:
        # Instructions when no file is uploaded
        st.info("""
            📁 **Please upload a consumption data file to begin.**
            
            Your CSV file should contain:
            - A column named `consumption_kWh`
            - 35,040 rows (one full year of 15-minute interval data)
            - Consumption values in kWh per 15-minute period
            
            Example format:
            ```
            consumption_kWh
            0.125
            0.130
            0.128
            ...
            ```
            
            **Note**: 35,040 rows = 96 intervals/day × 365 days
        """)
        
        # Sample data generator
        if st.button("📊 Generate Sample Data"):
            # Create realistic consumption profile
            # 35040 = 96 intervals per day * 365 days
            hours = np.arange(0, 8760, 0.25)  # 15-min intervals for a year
            
            # Base load + daily pattern + seasonal variation + noise
            base_load = 0.3
            daily_pattern = 0.4 * np.sin((hours % 24 - 6) * np.pi / 12) ** 2
            seasonal_pattern = 0.2 * np.cos((hours / 8760) * 2 * np.pi)
            noise = np.random.normal(0, 0.05, len(hours))
            
            consumption = np.maximum(0, base_load + daily_pattern + seasonal_pattern + noise)
            
            sample_df = pd.DataFrame({
                'consumption_kWh': consumption
            })
            
            st.success(f"Generated {len(sample_df)} consumption data points")
            
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Consumption Data",
                data=csv,
                file_name="sample_consumption_data.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>Powered by PVGIS API v5.2 | Solar data © European Commission</p>
            <p>Made with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    build_ui()
