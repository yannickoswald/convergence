import copy
import numpy as np
from scenario_class import Scenario

# We must put the base params here so the worker file knows about them
base_scenario_params = {
    "end_year": 2100,
    "carbon_budget": 1150 * 0.95 - 2 * 35,
    "gdp_assumption": "constant_ratio",
    "pop_growth_assumption": "semi_log_model",
    "tech_evolution_assumption": "plausible",
    "tech_hysteresis_assumption": "off",
    "steady_state_high_income_assumption": "off",
    "k": 0.05,
    "t0": 2060,
    "population_hysteresis_assumption": "off",
    "run_until_2100": "on",
    "cdr_level_2100": 5,
    "emission_elasticity_assumption": "constant",
}

def run_single_scenario_with_deciles(param_tuple):
    """
    Runs a single iteration of the scenario and returns the extracted trajectory rows.
    """
    # 1. FIX: Unpack all 5 variables here
    inc, cdr, degrowth, fir, elast = param_tuple

    params = copy.deepcopy(base_scenario_params)
    params["income_goal"] = inc
    params["cdr_assumption"] = cdr
    # 2. FIX: Assign the new degrowth variable to the correct parameter
    params["tech_hysteresis_assumption"] = degrowth 
    params["final_improvement_rate"] = fir
    params["base_elasticity"] = elast

    scenario = Scenario(params)
    scenario.run()

    scenario_rows = []

    for country_name, country in scenario.countries.items():
        for year in country.gdp_trajectory.keys():
            if year > params["end_year"]:
                continue

            pop = country.population_trajectory.get(year, np.nan)
            gdp = country.gdp_trajectory.get(year, np.nan)
            gdppc = country.gdppc_trajectory.get(year, np.nan)
            tot_emissions = country.emissions_trajectory.get(year, np.nan)
            gini = country.gini_coefficient_trajectory.get(year, np.nan)

            for d in range(1, 11):
                d_key = f'decile{d}'

                decile_income = country.decile_trajectories.get(d_key, {}).get(year, np.nan)
                decile_emissions_pc = country.decile_emissions_trajectories_pc.get(d_key, {}).get(year, np.nan)
                decile_absolute_emissions = country.decile_emissions_trajectories.get(d_key, {}).get(year, np.nan)

                row = {
                    "Income_Goal": inc,
                    "CDR_Assumption": cdr,
                    "Degrowth_Assumption": degrowth, # 3. FIX: Add it to your output rows too!
                    "Final_Improvement_Rate": fir,
                    "Elasticity": elast,
                    "Country": country_name,
                    "Country_Code": getattr(country, 'code', 'UNKNOWN'),
                    "Year": year,
                    "Decile": d,
                    "Decile_Income": decile_income,
                    "Decile_Emissions_pc": decile_emissions_pc,
                    "Decile_Emissions_Absolute": decile_absolute_emissions,
                    "Macro_Population": pop,
                    "Macro_GDP": gdp,
                    "Macro_GDP_pc": gdppc,
                    "Macro_Total_Emissions": tot_emissions,
                    "Gini": gini
                }
                scenario_rows.append(row)

    return scenario_rows
