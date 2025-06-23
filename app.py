import json
import numpy as np
import pandas as pd
import streamlit as st
import io
import zipfile
from typing import Dict, Tuple, Optional, List

# ==============================================================================
# SECTION 1: BACKEND LOGIC - OPTIMIZED VERSION
# ==============================================================================

def process_pv_data(pv_df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Processa i dati di produzione PV dal CSV caricato.
    Assume che l'utente fornisca dati in kWh per intervallo di 15 minuti.
    
    Returns:
        Tuple of (is_valid, message, processed_dataframe)
    """
    # Check if dataframe is empty
    if pv_df.empty:
        return False, "File is empty", None
    
    # Check if dataframe has columns
    if len(pv_df.columns) == 0:
        return False, "No columns found in CSV", None
    
    # Use the first column as PV production data
    pv_column = pv_df.columns[0]
    
    # Create clean dataframe with standardized column name
    clean_df = pd.DataFrame({
        'pv_production_kwp': pv_df[pv_column]
    })
    
    # Check number of rows
    expected_rows = 35040
    actual_rows = len(clean_df)
    
    if actual_rows != expected_rows:
        return False, f"Expected {expected_rows:,} rows (15-min intervals for 1 year), found {actual_rows:,}", None
    
    # Check if all values are zero
    if clean_df['pv_production_kwp'].sum() == 0:
        return False, "All PV production values are zero", None
    
    # Calculate annual production (assuming kWh per 15-min interval)
    annual_production = clean_df['pv_production_kwp'].sum()
    
    # Check if annual production is reasonable
    if annual_production < 500:
        return False, f"Annual production too low: {annual_production:.0f} kWh/kWp. Expected 800-1500 kWh/kWp for Europe", None
    
    if annual_production > 2000:
        return False, f"Annual production too high: {annual_production:.0f} kWh/kWp. Expected 800-1500 kWh/kWp for Europe", None
    
    # Store annual production for reference
    clean_df.attrs['annual_production'] = annual_production
    
    return True, f"Valid PV data - Annual production: {annual_production:.0f} kWh/kWp", clean_df


def process_consumption_data(consumption_df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Processa i dati di consumo dal CSV caricato.
    
    Returns:
        Tuple of (is_valid, message, processed_dataframe)
    """
    # Check if dataframe is empty
    if consumption_df.empty:
        return False, "File is empty", None
    
    # Check required column exists
    if 'consumption_kWh' not in consumption_df.columns:
        return False, "CSV must contain 'consumption_kWh' column", None
    
    # Create clean dataframe
    clean_df = pd.DataFrame({
        'consumption_kWh': consumption_df['consumption_kWh']
    })
    
    # Check number of rows
    expected_rows = 35040
    actual_rows = len(clean_df)
    
    if actual_rows != expected_rows:
        return False, f"Expected {expected_rows:,} rows (15-min intervals for 1 year), found {actual_rows:,}", None
    
    # Check if all values are zero
    if clean_df['consumption_kWh'].sum() == 0:
        return False, "All consumption values are zero", None
    
    # Calculate annual consumption for reference
    annual_consumption = clean_df['consumption_kWh'].sum()
    clean_df.attrs['annual_consumption'] = annual_consumption
    
    return True, f"Valid consumption data - Annual consumption: {annual_consumption:.0f} kWh", clean_df


def run_simulation_vectorized(pv_kwp: float, bess_kwh_nominal: float, pv_production_baseline: pd.DataFrame, 
                            consumption_profile: pd.DataFrame, config: Dict, 
                            export_details: bool = False, debug: bool = False) -> Dict:
    """
    Optimized vectorized simulation with exact energy flow formulas.
    Assumes PV data is in kWh per 15-min interval.
    """
    # Extract configuration parameters
    dod = config['bess_dod']
    c_rate = config['bess_c_rate']
    efficiency = config['bess_efficiency']
    pv_degr_rate = config['pv_degradation_rate']
    calendar_degradation = config['bess_calendar_degradation_rate']
    tot_cycles = config['bess_cycles']
    
    # Calculate battery parameters
    kwh_netti = bess_kwh_nominal * dod  # Usable capacity
    max_charge_discharge_per_step = bess_kwh_nominal * c_rate / 4  # C-rate limit for 15-min step
    
    # Convert data to numpy arrays for faster computation
    pv_base = pv_production_baseline['pv_production_kwp'].to_numpy(dtype=np.float32)
    cons = consumption_profile['consumption_kWh'].to_numpy(dtype=np.float32)
    
    # Debug information
    if debug:
        st.write("### 🔍 Debug Information - Simulation Input")
        st.write(f"**PV System:** {pv_kwp} kWp")
        st.write(f"**Battery:** {bess_kwh_nominal} kWh (usable: {kwh_netti:.1f} kWh)")
        st.write(f"**PV baseline data points:** {len(pv_base)}")
        
        annual_prod_1kwp = pv_base.sum()
        
        st.write(f"**PV baseline stats (1 kWp):**")
        st.write(f"  - Annual production: {annual_prod_1kwp:.0f} kWh")
        st.write(f"  - Peak value: {pv_base.max():.3f} kWh/interval")
        st.write(f"  - Non-zero values: {(pv_base > 0).sum()} of {len(pv_base)}")
    
    # Calculate base case annual cost
    annual_consumption = cons.sum()
    base_case_annual_cost = annual_consumption * config['grid_price_buy']
    
    # Simulation parameters
    steps_per_year = 35040  # 96 steps/day * 365 days
    
    # Initialize tracking variables
    annual_metrics = []
    total_grid_import = 0
    total_consumption = annual_consumption * 5  # 5 years total
    
    # Pre-calculate PV degradation factors
    pv_degradation_factors = (1 - pv_degr_rate) ** np.arange(5)
    
    # Store detailed timestep data if requested
    timestep_data = [] if export_details else None
    
    # Run 5-year simulation
    for year in range(5):
        # Apply PV degradation and scale by system size
        # Data is in kWh per 15-min interval
        pv_production = pv_base * pv_kwp * pv_degradation_factors[year]
        
        if debug and year == 0:
            st.write(f"**Year 1 PV production:**")
            st.write(f"  - Total: {pv_production.sum():.0f} kWh")
            st.write(f"  - Degradation factor: {pv_degradation_factors[year]:.3f}")
        
        # Initialize arrays for this year
        soc = np.zeros(steps_per_year + 1, dtype=np.float32)
        soh = np.zeros(steps_per_year + 1, dtype=np.float32)
        kwh_scaricati = np.zeros(steps_per_year, dtype=np.float32)
        kwh_caricati = np.zeros(steps_per_year, dtype=np.float32)
        immissione = np.zeros(steps_per_year, dtype=np.float32)
        acquisto = np.zeros(steps_per_year, dtype=np.float32)
        
        # Set initial SoH
        soh[0] = 1.0 if year == 0 else annual_metrics[-1]['final_soh']
        
        # Simulate each 15-minute step
        for t in range(steps_per_year):
            produzione = pv_production[t]
            consumo = cons[t]
            
            # Current max SoC based on battery health
            max_soc_current = kwh_netti * soh[t]
            
            # 1. Calculate battery discharge
            if consumo > produzione:
                energy_deficit = (consumo - produzione) / efficiency
                kwh_scaricati[t] = min(energy_deficit, soc[t], max_charge_discharge_per_step)
            else:
                kwh_scaricati[t] = 0
            
            # 2. Calculate grid export
            if produzione > consumo:
                excess_after_battery = soc[t] + produzione - consumo - max_soc_current
                immissione[t] = max(0, excess_after_battery)
            else:
                immissione[t] = 0
            
            # 3. Calculate grid import
            energy_from_battery = kwh_scaricati[t] * efficiency
            acquisto[t] = max(0, consumo - produzione - energy_from_battery)
            
            # 4. Update SoC
            if produzione >= consumo:
                delta_soc = (produzione - consumo) * efficiency
                actual_charge = min(delta_soc, max_soc_current - soc[t], max_charge_discharge_per_step)
                soc[t+1] = soc[t] + actual_charge
                kwh_caricati[t] = actual_charge / efficiency
            else:
                soc[t+1] = soc[t] - kwh_scaricati[t]
                kwh_caricati[t] = 0
            
            # 5. Calculate SoH degradation
            if t > 0:
                cycle_degradation = (kwh_scaricati[t-1] / bess_kwh_nominal) * (0.2 * 1.15 / tot_cycles)
            else:
                cycle_degradation = 0
            
            calendar_deg_per_step = calendar_degradation / steps_per_year
            soh[t+1] = max(0, soh[t] - calendar_deg_per_step - cycle_degradation)
            
            # Store timestep data if requested (daily aggregation)
            if export_details and t % 96 == 0:
                day_slice = slice(t, min(t + 96, steps_per_year))
                timestep_data.append({
                    'year': year + 1,
                    'day': t // 96 + 1,
                    'pv_production_kwh': pv_production[day_slice].sum(),
                    'consumption_kwh': cons[day_slice].sum(),
                    'battery_charge_kwh': kwh_caricati[day_slice].sum(),
                    'battery_discharge_kwh': (kwh_scaricati[day_slice] * efficiency).sum(),
                    'grid_import_kwh': acquisto[day_slice].sum(),
                    'grid_export_kwh': immissione[day_slice].sum(),
                    'avg_soc_kwh': soc[t:t+96].mean(),
                    'soh_percent': soh[t] * 100
                })
        
        # Calculate annual totals
        yearly_metrics = {
            'year': year + 1,
            'pv_production': pv_production.sum(),
            'consumption': annual_consumption,
            'energy_bought': acquisto.sum(),
            'energy_sold': immissione.sum(),
            'energy_to_battery': kwh_caricati.sum(),
            'energy_from_battery': (kwh_scaricati * efficiency).sum(),
            'self_consumption': annual_consumption - acquisto.sum(),
            'self_sufficiency': (annual_consumption - acquisto.sum()) / annual_consumption if annual_consumption > 0 else 0,
            'final_soh': soh[steps_per_year],
            'avg_soc': soc[1:steps_per_year+1].mean(),
            'max_soc': soc[1:steps_per_year+1].max(),
            'min_soc': soc[1:steps_per_year+1].min()
        }
        
        annual_metrics.append(yearly_metrics)
        total_grid_import += yearly_metrics['energy_bought']
    
    # Calculate financial metrics
    capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
    capex_bess = bess_kwh_nominal * 150
    total_capex = capex_pv + capex_bess
    
    # O&M costs
    om_pv = (12 - 0.01 * pv_kwp) * pv_kwp
    om_bess = 1500 + (capex_bess * 0.015)
    total_om = om_pv + om_bess
    
    # Calculate cash flows
    cash_flows = [-total_capex]  # Year 0
    
    for metrics in annual_metrics:
        base_cost = metrics['consumption'] * config['grid_price_buy']
        energy_cost = metrics['energy_bought'] * config['grid_price_buy']
        energy_revenue = metrics['energy_sold'] * config['grid_price_sell']
        
        annual_cf = base_cost - energy_cost + energy_revenue - total_om
        cash_flows.append(annual_cf)
    
    # Project cash flows for years 6-10
    if cash_flows:
        last_cf = cash_flows[-1]
        for _ in range(5):
            last_cf *= 0.97  # 3% degradation
            cash_flows.append(last_cf)
    
    # Calculate NPV and payback
    wacc = config['wacc']
    npv = sum(cf / ((1 + wacc) ** i) for i, cf in enumerate(cash_flows))
    
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
    
    result = {
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
        "annual_consumption": annual_consumption
    }
    
    if export_details:
        result["timestep_data"] = timestep_data
    
    return result


def find_optimal_system(user_inputs: Dict, config: Dict, pv_baseline: pd.DataFrame, 
                       enable_debug: bool = False) -> Optional[Dict]:
    """
    Optimized system search with adaptive grid search.
    """
    # Calculate maximum feasible sizes
    max_kwp_from_area = user_inputs['available_area_m2'] / 5.0  # 5 m²/kWp
    max_kwp_from_budget = user_inputs['budget'] / 650  # Minimum cost estimate
    max_kwp = min(max_kwp_from_area, max_kwp_from_budget, 500)  # Cap at 500 kWp
    
    max_kwh = min(user_inputs['budget'] / 150, 1000)  # Cap at 1000 kWh
    
    # Ensure valid search ranges
    if max_kwp < 5:
        st.error(f"Budget too low! Minimum PV system costs ~€3,250 (5 kWp)")
        return None
    
    # Adaptive step sizes based on search space
    n_pv_steps = min(20, int(max_kwp / 5))
    n_bess_steps = min(20, int(max_kwh / 5))
    
    kwp_step = max(5, int(max_kwp / n_pv_steps))
    kwh_step = max(5, int(max_kwh / n_bess_steps))
    
    pv_search_range = np.arange(kwp_step, max_kwp + kwp_step, kwp_step)
    bess_search_range = np.arange(0, max_kwh + kwh_step, kwh_step)
    
    # Show search space
    st.info(f"🔍 Searching {len(pv_search_range)} PV sizes × {len(bess_search_range)} battery sizes = {len(pv_search_range) * len(bess_search_range)} configurations")
    
    # Initialize search
    best_result = None
    best_npv = -float('inf')
    results_matrix = []
    valid_solutions = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    total_sims = len(pv_search_range) * len(bess_search_range)
    sim_count = 0
    
    # Search for optimal combination
    for i, pv_kwp in enumerate(pv_search_range):
        for j, bess_kwh in enumerate(bess_search_range):
            sim_count += 1
            
            # Update progress
            if sim_count % max(1, total_sims // 50) == 0:
                progress_bar.progress(min(sim_count / total_sims, 1.0))
            
            # Check budget constraint
            capex_pv = pv_kwp * (600 + 600 * np.exp(-pv_kwp / 290))
            capex_bess = bess_kwh * 150
            
            if (capex_pv + capex_bess) > user_inputs['budget']:
                continue
            
            valid_solutions += 1
            
            # Run simulation
            result = run_simulation_vectorized(
                pv_kwp, bess_kwh, pv_baseline, 
                user_inputs['consumption_profile_df'], 
                config, debug=(enable_debug and valid_solutions == 1)
            )
            
            result['pv_kwp'] = pv_kwp
            result['bess_kwh'] = bess_kwh
            results_matrix.append(result)
            
            # Track best result
            if result['npv_eur'] > best_npv:
                best_npv = result['npv_eur']
                best_result = result.copy()
                best_result['optimal_kwp'] = pv_kwp
                best_result['optimal_kwh'] = bess_kwh
    
    progress_bar.empty()
    
    if valid_solutions > 0:
        st.success(f"✅ Analyzed {valid_solutions} valid configurations")
        
        # Add results matrix for visualization
        if best_result:
            best_result['all_results'] = results_matrix
            
            # Show top 5 configurations
            sorted_results = sorted(results_matrix, key=lambda x: x['npv_eur'], reverse=True)[:5]
            
            with st.expander("🏆 Top 5 Configurations by NPV"):
                for idx, res in enumerate(sorted_results):
                    st.write(f"**#{idx+1}:** {res['pv_kwp']} kWp / {res['bess_kwh']} kWh - NPV: €{res['npv_eur']:,.0f}")
    else:
        st.warning("No valid configurations found within constraints")
    
    return best_result


def export_detailed_calculations(optimal_system: Dict, config: Dict, 
                               pv_kwp: float, bess_kwh: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Export detailed calculations to CSV format."""
    
    # 1. Annual Summary
    annual_data = []
    for i, metrics in enumerate(optimal_system['annual_metrics']):
        year_idx = metrics['year']
        
        base_cost = metrics['consumption'] * config['grid_price_buy']
        energy_cost = metrics['energy_bought'] * config['grid_price_buy']
        energy_revenue = metrics['energy_sold'] * config['grid_price_sell']
        net_energy_cost = energy_cost - energy_revenue
        savings = base_cost - net_energy_cost
        
        annual_data.append({
            'Year': year_idx,
            'PV_Production_kWh': metrics['pv_production'],
            'Consumption_kWh': metrics['consumption'],
            'Grid_Import_kWh': metrics['energy_bought'],
            'Grid_Export_kWh': metrics['energy_sold'],
            'Battery_Charge_kWh': metrics['energy_to_battery'],
            'Battery_Discharge_kWh': metrics['energy_from_battery'],
            'Self_Consumption_kWh': metrics['self_consumption'],
            'Self_Sufficiency_%': metrics['self_sufficiency'] * 100,
            'Battery_SoH_%': metrics['final_soh'] * 100,
            'Base_Cost_EUR': base_cost,
            'Energy_Buy_Cost_EUR': energy_cost,
            'Energy_Sell_Revenue_EUR': energy_revenue,
            'Net_Energy_Cost_EUR': net_energy_cost,
            'O&M_Cost_EUR': optimal_system['om_costs'],
            'Annual_Savings_EUR': savings,
            'Cash_Flow_EUR': optimal_system['cash_flows'][year_idx] if year_idx < len(optimal_system['cash_flows']) else 0
        })
    
    annual_df = pd.DataFrame(annual_data)
    
    # 2. Financial Details
    financial_data = []
    
    # CAPEX breakdown
    financial_data.extend([
        {
            'Category': 'CAPEX',
            'Item': 'PV System',
            'Formula': f'{pv_kwp} × (600 + 600 × exp(-{pv_kwp}/290))',
            'Value_EUR': optimal_system['capex_pv']
        },
        {
            'Category': 'CAPEX',
            'Item': 'Battery System',
            'Formula': f'{bess_kwh} × 150',
            'Value_EUR': optimal_system['capex_bess']
        },
        {
            'Category': 'CAPEX',
            'Item': 'Total CAPEX',
            'Formula': 'PV + Battery',
            'Value_EUR': optimal_system['total_capex_eur']
        }
    ])
    
    # O&M breakdown
    financial_data.extend([
        {
            'Category': 'O&M (Annual)',
            'Item': 'PV O&M',
            'Formula': f'(12 - 0.01 × {pv_kwp}) × {pv_kwp}',
            'Value_EUR': optimal_system['om_pv']
        },
        {
            'Category': 'O&M (Annual)',
            'Item': 'Battery O&M',
            'Formula': f'1500 + ({optimal_system["capex_bess"]} × 0.015)',
            'Value_EUR': optimal_system['om_bess']
        },
        {
            'Category': 'O&M (Annual)',
            'Item': 'Total O&M',
            'Formula': 'PV O&M + Battery O&M',
            'Value_EUR': optimal_system['om_costs']
        }
    ])
    
    # NPV details
    for i, cf in enumerate(optimal_system['cash_flows'][:11]):  # First 11 entries (year 0-10)
        discount_factor = 1 / ((1 + config['wacc']) ** i)
        discounted_cf = cf * discount_factor
        
        financial_data.append({
            'Category': f'Cash Flow Year {i}',
            'Item': 'Annual Cash Flow',
            'Formula': 'Base Cost - Energy Cost + Energy Revenue - O&M' if i > 0 else 'Initial Investment',
            'Value_EUR': cf
        })
        financial_data.append({
            'Category': f'Cash Flow Year {i}',
            'Item': 'Discounted Cash Flow',
            'Formula': f'{cf:.2f} × {discount_factor:.4f}',
            'Value_EUR': discounted_cf
        })
    
    financial_data.extend([
        {
            'Category': 'Final Results',
            'Item': 'NPV',
            'Formula': 'Sum of all discounted cash flows',
            'Value_EUR': optimal_system['npv_eur']
        },
        {
            'Category': 'Final Results',
            'Item': 'Payback Period',
            'Formula': 'Year when cumulative CF > 0',
            'Value_EUR': optimal_system['payback_period_years']
        }
    ])
    
    financial_df = pd.DataFrame(financial_data)
    
    # 3. Configuration Parameters
    config_data = [
        {'Parameter': 'PV_Size_kWp', 'Value': pv_kwp, 'Unit': 'kWp'},
        {'Parameter': 'Battery_Size_kWh', 'Value': bess_kwh, 'Unit': 'kWh'},
        {'Parameter': 'Battery_DoD', 'Value': config['bess_dod'] * 100, 'Unit': '%'},
        {'Parameter': 'Battery_C_Rate', 'Value': config['bess_c_rate'], 'Unit': 'C'},
        {'Parameter': 'Battery_Efficiency', 'Value': config['bess_efficiency'] * 100, 'Unit': '%'},
        {'Parameter': 'Battery_Cycles', 'Value': config['bess_cycles'], 'Unit': 'cycles'},
        {'Parameter': 'PV_Degradation_Rate', 'Value': config['pv_degradation_rate'] * 100, 'Unit': '%/year'},
        {'Parameter': 'Battery_Calendar_Degradation', 'Value': config['bess_calendar_degradation_rate'] * 100, 'Unit': '%/year'},
        {'Parameter': 'Grid_Buy_Price', 'Value': config['grid_price_buy'], 'Unit': 'EUR/kWh'},
        {'Parameter': 'Grid_Sell_Price', 'Value': config['grid_price_sell'], 'Unit': 'EUR/kWh'},
        {'Parameter': 'WACC', 'Value': config['wacc'] * 100, 'Unit': '%'},
        {'Parameter': 'Annual_Consumption', 'Value': optimal_system['annual_consumption'], 'Unit': 'kWh'}
    ]
    
    config_df = pd.DataFrame(config_data)
    
    # 4. Timestep data if available
    timestep_df = None
    if 'timestep_data' in optimal_system and optimal_system['timestep_data']:
        timestep_df = pd.DataFrame(optimal_system['timestep_data'])
    
    return annual_df, financial_df, config_df, timestep_df


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

Note: All PV production values are assumed to be in kWh per 15-minute interval.
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
            background-color: #0D0C54;
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
            max_value=2500000,
            value=500000,
            step=1000,
            help="Total available budget for PV and BESS installation"
        )
        
        available_area_m2 = st.number_input(
            "Available Area for PV (m²)",
            min_value=10,
            max_value=10000,
            value=400,
            step=10,
            help="Total roof or ground area available for solar panels (5 m²/kWp)"
        )
        
        # PV Production Data
        st.subheader("2. PV Production Data")
        st.markdown("""
            📁 **Upload PV production baseline (1 kWp)**
            - 35,040 rows (15-min intervals)
            - Values in **kWh** (energy per interval)
            - Expected annual production: 800-1500 kWh/kWp
        """)
        
        pv_file = st.file_uploader(
            "Upload PV production data (CSV)",
            type="csv",
            help="CSV with 35,040 rows of PV production for 1 kWp",
            key="pv_upload"
        )
        
        
        # Consumption Profile
        st.subheader("3. Consumption Profile")
        st.markdown("""
            📁 **Upload consumption data**
            - Column named 'consumption_kWh'
            - 35,040 rows (15-min intervals)
            - Values in kWh per interval
        """)
        
        consumption_file = st.file_uploader(
            "Upload consumption data (CSV)",
            type="csv",
            help="CSV with 'consumption_kWh' column and 35,040 rows",
            key="consumption_upload"
        )
        
  
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
            pv_df = pd.read_csv(pv_file)
            is_valid, message, pv_baseline = process_pv_data(pv_df)
            
            if not is_valid:
                st.error(f"❌ PV data error: {message}")
                return
            else:
                annual_prod = pv_baseline.attrs.get('annual_production', 0)
                st.success(f"""
                    ✅ PV data loaded successfully!
                    - Annual production: **{annual_prod:.0f} kWh/kWp**
                    - {message}
                """)
                
                # Show PV data statistics
                with st.expander("📊 PV Production Statistics (1 kWp baseline)"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Get annual production
                    annual_pv = pv_baseline.attrs.get('annual_production', 0)
                    
                    # Calculate peak power (convert from kWh per 15-min to kW)
                    peak_power = pv_baseline['pv_production_kwp'].max() * 4
                    
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
                    
                    # Show daily profile
                    st.write("**Average Daily Profile (1 kWp):**")
                    hourly_avg = []
                    for h in range(24):
                        hour_data = pv_baseline.iloc[h*4:(h+1)*4]['pv_production_kwp']
                        # Sum kWh values for the hour
                        hourly_avg.append(hour_data.sum())
                    
                    profile_df = pd.DataFrame({
                        'Hour': range(24),
                        'Energy (kWh)': hourly_avg
                    })
                    
                    st.line_chart(profile_df.set_index('Hour')['Energy (kWh)'])
            
            # Load and validate consumption data
            consumption_df = pd.read_csv(consumption_file)
            is_valid, message, consumption_baseline = process_consumption_data(consumption_df)
            
            if not is_valid:
                st.error(f"❌ Consumption data error: {message}")
                return
            else:
                st.success(f"✅ Consumption data loaded successfully! {message}")
            
            # Display consumption statistics
            st.subheader("📊 Consumption Profile Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_annual = consumption_baseline.attrs['annual_consumption']
            with col1:
                st.metric("Annual Consumption", f"{total_annual:,.0f} kWh")
            with col2:
                st.metric("Average Daily", f"{total_annual/365:,.1f} kWh")
            with col3:
                st.metric("Peak Load", f"{consumption_baseline['consumption_kWh'].max()*4:,.1f} kW")
            with col4:
                st.metric("Base Load", f"{consumption_baseline['consumption_kWh'].min()*4:,.1f} kW")
            
        except Exception as e:
            st.error(f"""
                ❌ Error reading files: {str(e)}
                
                **Please check your CSV file format:**
                - Should have proper column headers
                - 35,040 data rows (no empty rows)
                - No extra separators or columns
                - Values should be numeric
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
                "consumption_profile_df": consumption_baseline
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
                                                         consumption_baseline, config)
                        
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
               - Values in **kWh** (energy per 15-min interval)
               - Expected annual production: 800-1500 kWh/kWp
               - Single column with header
               
            2. **Consumption Data** (CSV)
               - Column named 'consumption_kWh'
               - 35,040 rows of 15-minute intervals
               - Values in kWh per interval
               
            ⚠️ **Important**: Ensure your data is properly formatted with numeric values.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>PV & BESS Optimizer v2.0 - Simplified Version</p>
            <p>Made with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    build_ui()
