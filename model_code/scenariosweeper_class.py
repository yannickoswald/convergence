import matplotlib.pyplot as plt
from scenario_class import Scenario
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# ==============================================================================
# THE WORKER FUNCTION (Placed OUTSIDE the class so CPU cores can easily load it)
# ==============================================================================
def _run_sweeper_iteration(scenario_params):
    """
    Runs a single scenario iteration independently. 
    Because this is a top-level function, Python can easily send it to other CPU cores.
    """
    scenario = Scenario(scenario_params)
    scenario.compute_country_scenario_params()

    # Convert scenario_params dictionary to a tuple of tuples (key, value pairs)
    scenario_key = tuple(sorted(scenario_params.items()))

    # Calculate the global average necessary growth rate
    global_growth_rate = scenario.compute_average_global_growth_rate()
    
    # Run the scenario
    scenario.run()

    # Calculate emissions
    carbon_budget = scenario_params["carbon_budget"]
    total_emission = scenario.sum_cumulative_emissions()
    total_emissions_gigatonnes = total_emission / 1e9
    total_emissions_ratio = total_emissions_gigatonnes / carbon_budget

    # Extract ONLY the scalar trajectory variables needed for the Sweeper contours
    final_emissions = scenario.compute_ending_global_emissions()
    population_below_income_goal = scenario.get_population_below_income_goal()

    # ❌ WE REMOVED THE HEAVY TRAJECTORY EXTRACTIONS HERE TO SAVE MASSIVE AMOUNTS OF RAM:
    # gini_coefficient_national = scenario.store_national_gini_coefficients()
    # national_gdp_trajectories = scenario.store_national_gdp_trajectories()

    return (
        scenario_key, 
        total_emissions_ratio, 
        global_growth_rate, 
        final_emissions, 
        population_below_income_goal
    )

# ==============================================================================
# THE SWEEPER CLASS
# ==============================================================================
class ScenarioSweeper:
    
    def __init__(self, end_year_values, income_goal_values, carbon_budget_values,
                       gdp_assumption_values, pop_growth_assumption_values,
                       tech_evolution_assumption_values, tech_hysteresis_assumption_values,
                       steady_state_high_income_assumption_values, sigmoid_parameters_values,
                       final_improvement_rate, population_hysteresis_assumption_values,
                       run_until_2100, cdr_assumption, cdr_level_2100, elasticity_assumption_values, elasticity_value):

        # Store the parameter values to be swept through
        self.end_year_values = end_year_values
        self.income_goal_values = income_goal_values
        self.carbon_budget_values = carbon_budget_values
        self.gdp_assumption_values = gdp_assumption_values
        self.pop_growth_assumption_values = pop_growth_assumption_values 
        self.tech_evolution_assumption_values = tech_evolution_assumption_values 
        self.tech_hysteresis_assumption_values = tech_hysteresis_assumption_values 
        self.steady_state_high_income_assumption_values = steady_state_high_income_assumption_values 
        self.population_hysteresis_assumption_values = population_hysteresis_assumption_values 
        self.sigmoid_parameters_values = sigmoid_parameters_values
        self.final_improvement_rate = final_improvement_rate
        self.run_until_2100 = run_until_2100 
        self.cdr_assumption = cdr_assumption 
        self.cdr_level_2100 = cdr_level_2100 
        self.elasticity_assumption_values = elasticity_assumption_values
        self.elasticity_value = elasticity_value

        # Output dictionaries
        self.total_emissions = {}
        self.growth_rate_global = {}
        self.gini_coefficient_change_rate_global = {}
        self.final_emissions = {}
        
        # We leave these empty since we are no longer extracting them
        self.gini_coefficient_national = {}
        self.national_gdp_trajectories = {}
        
        self.population_below_income_goal = {}

    def run_scenarios(self):
        """
        Iterates over all scenario parameters and runs them in parallel using ProcessPoolExecutor.
        """
        tech_hysteresis_assumption = self.tech_hysteresis_assumption_values[0] 
        gdp_assumption = self.gdp_assumption_values[0] 
        pop_growth_assumption = self.pop_growth_assumption_values[0] 
        tech_evolution_assumption = self.tech_evolution_assumption_values[0]  
        steady_state_high_income_assumption = self.steady_state_high_income_assumption_values[0]  
        population_hysteresis_assumption = self.population_hysteresis_assumption_values[0]
        run_until_2100 = self.run_until_2100  
        cdr_assumption = self.cdr_assumption  
        cdr_level_2100 = self.cdr_level_2100  
        sigmoid_parameters = self.sigmoid_parameters_values  
        final_improvement_rate = self.final_improvement_rate  
        elasticity_assumption = self.elasticity_assumption_values[0]
        elasticity_value = self.elasticity_value[0]

        tasks = []
        
        # Build task list
        for carbon_budget in self.carbon_budget_values: 
            for end_year in self.end_year_values: 
                for income_goal in self.income_goal_values: 
                    for cdr_level in self.cdr_level_2100: 
                        for carbon_intensity_rate in self.final_improvement_rate: 
                            
                            scenario_params = {
                                "end_year": end_year,
                                "income_goal": income_goal,
                                "carbon_budget": carbon_budget,
                                "gdp_assumption": gdp_assumption,
                                "pop_growth_assumption": pop_growth_assumption,
                                "tech_evolution_assumption": tech_evolution_assumption,
                                "tech_hysteresis_assumption": tech_hysteresis_assumption,
                                "steady_state_high_income_assumption": steady_state_high_income_assumption,
                                "kappa": sigmoid_parameters[0],
                                "t0": sigmoid_parameters[1],
                                "final_improvement_rate": carbon_intensity_rate,
                                "population_hysteresis_assumption": population_hysteresis_assumption,
                                "run_until_2100": run_until_2100[0], 
                                "cdr_assumption": cdr_assumption[0], 
                                "cdr_level_2100": cdr_level,
                                "elasticity_assumption": elasticity_assumption, # Options: "off", "constant", "income_dependent"
                                "elasticity_value": elasticity_value     
                            }
                            tasks.append(scenario_params)

        total_runs = len(tasks)
        print(f"Total scenario combinations to run: {total_runs}")

        # Protect the Mac from thermal throttling by leaving cores free
        safe_cores = max(1, multiprocessing.cpu_count() - 2)

        # Run in parallel
        with ProcessPoolExecutor(max_workers=safe_cores) as executor:
            futures = {executor.submit(_run_sweeper_iteration, p): p for p in tasks}
            
            # Use as_completed so memory is freed the instant a core finishes its math
            for future in tqdm(as_completed(futures), total=total_runs, desc="Sweeping Scenarios"):
                try:
                    result = future.result()
                    
                    # Unpack the NEW stripped-down outputs (5 items instead of 7)
                    (scenario_key, total_emiss_ratio, global_growth, final_emiss, pop_below) = result
                    
                    # Save back to class attributes
                    self.total_emissions[scenario_key] = total_emiss_ratio
                    self.growth_rate_global[scenario_key] = global_growth
                    self.final_emissions[scenario_key] = final_emiss
                    self.population_below_income_goal[scenario_key] = pop_below
                    
                except Exception as e:
                    failed_params = futures[future]
                    print(f"Scenario failed for parameters {failed_params}: {e}")

        return (self.total_emissions, self.growth_rate_global, 
                self.gini_coefficient_change_rate_global, self.final_emissions, 
                self.gini_coefficient_national, self.national_gdp_trajectories, 
                self.population_below_income_goal)

    def create_scenario(self, params):
        return Scenario(params)

    # ==============================================================================
    # PLOT FUNCTIONS (Left exactly untouched)
    # ==============================================================================

    def plot_total_emissions_trade_off(self, dependent_var, variables_considered, fixed_color_scale, annotations_plot, colorscaleon, ax=None):
        # Default index
        ax_index = None

        if ax is not None and isinstance(ax, str) and re.match(r'ax\d+', ax):
            ax_index = int(re.search(r'\d+', ax).group())

        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        x_values_set = set()
        y_values_set = set()

        name_mapping = {"end_year": "Convergence Year",
                        "income_goal": "Income Goal $PPPpc", 
                        "carbon_budget": "Carbon Budget", 
                        "gdp_assumption": "GDP Assumption"}

        for key in dependent_var.keys():
            params_dict = {k: v for k, v in key}  
            x_values_set.add(params_dict[variables_considered[0]])
            y_values_set.add(params_dict[variables_considered[1]])

        x_values = sorted(x_values_set)
        y_values = sorted(y_values_set)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.zeros(X.shape)

        for key, value in dependent_var.items():
            params_dict = {k: v for k, v in key}  
            x_index = x_values.index(params_dict[variables_considered[0]])
            y_index = y_values.index(params_dict[variables_considered[1]])
            Z[y_index, x_index] = value
        
        colors_below = ['#4575b4', '#91bfdb']  
        colors_above = ['#fdae61', '#d73027']  

        def combine_cmaps(cmap_below, cmap_above, threshold, data):
            min_val, max_val = np.min(data), np.max(data)
            
            if threshold >= max_val:
                return cmap_below
            if threshold <= min_val:
                return cmap_above

            norm_thr = (threshold - min_val) / (max_val - min_val)
            norm_thr = np.clip(norm_thr, 0.0, 1.0)
            below_colors = cmap_below(np.linspace(0, 1, int(256 * norm_thr)))
            above_colors = cmap_above(np.linspace(0, 1, 256 - int(256 * norm_thr)))
            all_colors = np.vstack((below_colors, above_colors))
            return mcolors.LinearSegmentedColormap.from_list('combined_cmap', all_colors)
            
        cmap_below = mcolors.LinearSegmentedColormap.from_list("below", colors_below)
        cmap_above = mcolors.LinearSegmentedColormap.from_list("above", colors_above)

        combined_cmap = combine_cmaps(cmap_below, cmap_above, 1, Z)

        contourf_kwargs = {
            "levels": np.linspace(np.min(Z), np.max(Z), 200),  
            "cmap": combined_cmap
        }

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
            current_axis = ax

        contour = ax.contourf(X, Y, Z, **contourf_kwargs)
        if colorscaleon:
            colorbar = fig.colorbar(contour, ax=ax)
            colorbar.set_label(f'Ratio cum. emissions to 2\u00B0C budget', rotation=270, labelpad=15, fontsize=10)
        
        ax.set_xlabel(name_mapping[variables_considered[0]])
        ax.set_ylabel(name_mapping[variables_considered[1]])
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values], rotation=45)
        ax.set_yticks(y_values)
        ax.set_yticklabels([str(y) for y in y_values])
        ax.set_xlim(min(x_values), max(x_values))
        ax.set_ylim(min(y_values), max(y_values))

        contour_line_0 = ax.contour(X, Y, Z, levels=[1], colors='white', linestyles='dashed')
        
        if ax_index != 5:    
            pass 
        else:
            pass

        contour_line_1 = ax.contour(X, Y, Z, levels=[1.1858190709], colors='white', linestyles='dashed')
        def custom_fmt2(x):
            return '2°C 50%'
        ax.clabel(contour_line_1, fmt=custom_fmt2, inline=True, fontsize=8)

        paths_2_degree_budget = contour_line_0.get_paths() if not hasattr(contour_line_0, 'collections') else contour_line_0.collections[0].get_paths()
        paths_2_degree_budget_50pct = contour_line_1.get_paths() if not hasattr(contour_line_1, 'collections') else contour_line_1.collections[0].get_paths()

        def extract_coordinates(paths):
            coords_list = []
            for path in paths:
                vertices = path.vertices
                coords_list.append(vertices)  
            return coords_list

        coords_2_degree67 = extract_coordinates(paths_2_degree_budget)
        coords_2_degree50 = extract_coordinates(paths_2_degree_budget_50pct)

        try:
            x_pos_2100 = x_values.index(2100)
            y_pos_20000 = y_values.index(20000)
            x_coord_2100 = x_values[x_pos_2100]
            y_coord_20000 = y_values[y_pos_20000]
            ax.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)
            ax.annotate("Denmark\n2100\nscenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-30, 40), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
        except ValueError:
            pass

        try:
            x_pos = x_values.index(2100)
            y_pos = y_values.index(20000)
            z_value = Z[y_pos, x_pos]
            ax.scatter(x_values[x_pos], y_values[y_pos], color='red', s=100, zorder=5) 
            ax.annotate(f"Overshoot {z_value:.2f}", (x_values[x_pos], y_values[y_pos]), textcoords="offset points", xytext=(-40, 0), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'))
        except ValueError as e:
            pass

        ax.scatter(2060, 10000, s=100, c='blue', marker='o', zorder = 5, label='Costa Rica')
        ax.annotate("Costa Rica\n2060\nscenario", (2060, 10000), textcoords="offset points", xytext=(20,45), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')

        try:
            x_pos = x_values.index(2060)
            y_pos = y_values.index(10000)
            z_value = Z[y_pos, x_pos]
            ax.annotate(f"Overshoot {z_value:.2f}", (x_values[x_pos], y_values[y_pos]), textcoords="offset points", xytext=(-30, 10), ha='center', fontsize=8)
        except ValueError as e:
            pass

        if annotations_plot:
            pass

        if ax is None:
            return fig, ax
    
    def plot_growth_rate_trade_off(self, dependent_var, variables_considered, ax=None):
            if len(variables_considered) != 2:
                raise ValueError("variables_considered must contain exactly two elements")

            x_values_set = set()
            y_values_set = set()

            name_mapping = {"end_year": "Convergence Year",
                            "income_goal": "Income Goal $PPPpc", 
                            "carbon_budget": "Carbon Budget", 
                            "gdp_assumption": "GDP Assumption"}

            for key in dependent_var.keys():
                params_dict = {k: v for k, v in key}  
                x_values_set.add(params_dict[variables_considered[0]])
                y_values_set.add(params_dict[variables_considered[1]])

            x_values = sorted(x_values_set)
            y_values = sorted(y_values_set)

            X, Y = np.meshgrid(x_values, y_values)
            Z = np.zeros(X.shape)

            for key, value in dependent_var.items():
                params_dict = {k: v for k, v in key}  
                x_index = x_values.index(params_dict[variables_considered[0]])
                y_index = y_values.index(params_dict[variables_considered[1]])
                Z[y_index, x_index] = value

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.get_figure()
            
            contour = ax.contourf(X, Y, Z, levels=50, cmap='inferno')
            
            def to_percentage(x, pos):
                return '{:.0f}%'.format(x * 100)
            
            colorbar = fig.colorbar(contour, ax=ax, format=FuncFormatter(to_percentage))
            colorbar.set_label('Global growth rate hh income', rotation=270, labelpad=15)

            ax.set_xlabel(name_mapping[variables_considered[0]])
            ax.set_ylabel(name_mapping[variables_considered[1]])
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(x) for x in x_values], rotation=45)
            ax.set_yticks(y_values)
            ax.set_yticklabels([str(y) for y in y_values])
            ax.set_xlim(min(x_values), max(x_values))
            ax.set_ylim(min(y_values), max(y_values))

            contour_line_0 = ax.contour(X, Y, Z, levels=[0], colors='cyan', linestyles='dashed')
            ax.clabel(contour_line_0, fmt=f'0', inline=True, fontsize=8)

            contour_line_004 = ax.contour(X, Y, Z, levels=[0.04], colors='cyan', linestyles='dashed')
            ax.clabel(contour_line_004, fmt=f'4', inline=True, fontsize=8)

            try:
                x_pos_2100 = x_values.index(2100)
                y_pos_20000 = y_values.index(20000)
                x_coord_2100 = x_values[x_pos_2100]
                y_coord_20000 = y_values[y_pos_20000]
                ax.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)
                ax.annotate("Denmark 2100 scenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-80,-20), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
            except ValueError:
                pass

            try:
                x_pos_2050 = x_values.index(2060)
                y_pos_9100 = y_values.index(10000)
                x_coord_2050 = x_values[x_pos_2050]
                y_coord_9100 = y_values[y_pos_9100]
                ax.scatter(x_coord_2050, y_coord_9100, color='blue', s=100, zorder=5)  
                ax.annotate("Costa Rica 2060 scenario", (x_coord_2050, y_coord_9100), textcoords="offset points", xytext=(50,20), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
            except ValueError:
                pass

            if ax is None:
                return fig, ax
            
    def plot_overshoot_tradeoff(self, dependent_var, variables_considered, ax=None):
        if variables_considered != ["end_year", "income_goal"]:
            raise ValueError("variables_considered must be ['end_year', 'income_goal'] for this plot")

        data = {}
        years = set()
        goals = set()
        for key, val in dependent_var.items():
            params = dict(key)
            year = params['end_year']
            goal = params['income_goal']
            years.add(year)
            goals.add(goal)
            data.setdefault(year, {})[goal] = val
        years = sorted(years)
        goals = sorted(goals)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        cb_colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]

        for i, year in enumerate(years):
            color = cb_colors[i % len(cb_colors)]
            y_vals = [data[year].get(g, np.nan) for g in goals]
            ax.plot(goals, y_vals, marker='o', linestyle='-', label=str(year), color=color)

        ax.set_xlabel('Convergence Income Goal ($PPP per capita)', size=12)
        ax.set_ylabel('Overshoot (Emissions/Budget)', fontweight='bold')
        ax.set_title('Overshoot vs Income Goal')
        ax.set_xticks(goals)
        ax.set_xticklabels([str(g) for g in goals], rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

        return fig, ax
            

    def plot_gini_coefficient_change_trade_off(self, dependent_var, variables_considered, ax=None):
        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        x_values_set = set()
        y_values_set = set()
        for key in dependent_var.keys():
            params_dict = {k: v for k, v in key}

    def plot_final_emissions_trade_off(self, dependent_var, variables_considered, annotations_plot, colorscaleon, ax=None):
        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        x_values_set = set()
        y_values_set = set()
        name_mapping = {"end_year": "End Year",
                        "income_goal": "Income Goal $PPPpc", 
                        "carbon_budget": "Carbon Budget", 
                        "gdp_assumption": "GDP Assumption"}

        for key in dependent_var.keys():
            params_dict = {k: v for k, v in key}  
            x_values_set.add(params_dict[variables_considered[0]])
            y_values_set.add(params_dict[variables_considered[1]])

        x_values = sorted(x_values_set)
        y_values = sorted(y_values_set)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.zeros(X.shape)

        for key, value in dependent_var.items():
            params_dict = {k: v for k, v in key}  
            x_index = x_values.index(params_dict[variables_considered[0]])
            y_index = y_values.index(params_dict[variables_considered[1]])
            Z[y_index, x_index] = value
        
        simple_cmap = plt.cm.viridis  

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        contour = ax.contourf(X, Y, Z, levels=200, cmap=simple_cmap)
        if colorscaleon:
            colorbar = fig.colorbar(contour, ax=ax)
            colorbar.set_label(f'Pct of emissions left compared to 2022', rotation=270, labelpad=15, fontsize=8)
            def format_tick(x, _):
                return f'{x * 100:.0f}%'
            colorbar.formatter = FuncFormatter(format_tick)
            colorbar.update_ticks()

        ax.set_xlabel(name_mapping[variables_considered[0]])
        ax.set_ylabel(name_mapping[variables_considered[1]])
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values], rotation=45)
        ax.set_yticks(y_values)
        ax.set_yticklabels([str(y) for y in y_values])
        ax.set_xlim(min(x_values), max(x_values))
        ax.set_ylim(min(y_values), max(y_values))

        contour_line_0 = ax.contour(X, Y, Z, levels=[0.2], colors='orange', linestyles='dashed')
        ax.clabel(contour_line_0, inline=True, fmt = {0.2: "<20%"}, fontsize=10)

        contour_line_00 = ax.contour(X, Y, Z, levels=[0.01], colors='orange', linestyles='dashed')
        ax.clabel(contour_line_00, inline=True, fmt = {0.01: "<1%"}, fontsize=10)

        paths_0 = contour_line_0.get_paths() if not hasattr(contour_line_0, 'collections') else contour_line_0.collections[0].get_paths()

        def extract_coordinates(paths):
            coords_list = []
            for path in paths:
               vertices = path.vertices
               coords_list.append(vertices)  
            return coords_list

        coords_0 = extract_coordinates(paths_0)
        
        try:
            x_pos_2100 = x_values.index(2100)
            y_pos_20000 = y_values.index(20000)
            x_coord_2100 = x_values[x_pos_2100]
            y_coord_20000 = y_values[y_pos_20000]
            ax.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)
            ax.annotate("2100\n Denmark\n scenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-30, 40), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
        except ValueError:
            pass

        try:
            x_pos = x_values.index(2100)
            y_pos = y_values.index(20000)
            z_value = Z[y_pos, x_pos]
            ax.scatter(x_values[x_pos], y_values[y_pos], color='red', s=100, zorder=5)  
            ax.annotate(f"{z_value * 100:.0f}%", (x_values[x_pos], y_values[y_pos]), textcoords="offset points", xytext=(-30, 5), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")
        except ValueError as e:
            pass

        try:
            x_pos2 = x_values.index(2060)
            y_pos2 = y_values.index(10000)
            z_value2 = Z[y_pos2, x_pos2]
            ax.scatter(x_values[x_pos2], y_values[y_pos2], color='red', s=100, zorder=5)  
            ax.annotate(f"{z_value2 * 100:.0f}%", (x_values[x_pos2], y_values[y_pos2]), textcoords="offset points", xytext=(30, 0), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")
        except ValueError as e:
            pass

        try:
            x_pos3 = x_values.index(2100)
            y_pos3 = y_values.index(10000)
            z_value3 = Z[y_pos3, x_pos3]
            
            x_pos4 = x_values.index(2060)
            y_pos4 = y_values.index(20000)
            z_value4 = Z[y_pos4, x_pos4]

            ax.scatter(x_values[x_pos3], y_values[y_pos3], color='darkblue', s=100, zorder=5)  
            ax.annotate(f"{z_value3 * 100:.0f}%", (x_values[x_pos3], y_values[y_pos3]), textcoords="offset points", xytext=(-20, -10), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")
            
            ax.scatter(x_values[x_pos4], y_values[y_pos4], color='darkred', s=100, zorder=5)  
            ax.annotate(f"{z_value4 * 100:.0f}%", (x_values[x_pos4], y_values[y_pos4]), textcoords="offset points", xytext=(0, 20), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")

            ax.plot([2060, 2100], [10000, 10000], color='blue', linestyle='dotted')
            ax.plot([2060, 2100], [20000, 20000], color='red', linestyle='dotted')
        except ValueError as e:
            pass

        if annotations_plot:
            coords_growth1 = np.array([[2040., 7091.76725433],
                                [2060., 7104.1157445],
                                [2078.19032277, 7107.60378143],
                                [2081.80967737, 7108.11988921],
                                [2100., 7109.81960993]])
            ax.plot(coords_growth1[:, 0], coords_growth1[:, 1], color = "cyan", linestyle = '--', label='0%')  
            
            coords_growth2 = np.array([[2040., 13763.74664383],
                                [2044.7325622, 15000.],
                                [2053.52721919, 20000.],
                                [2057.06128359, 24277.62144207],
                                [2058.02526381, 25748.84170682],
                                [2060., 29776.31871013],
                                [2060.31708066, 30000.]])
            ax.plot(coords_growth2[:, 0], coords_growth2[:, 1], color = "cyan", linestyle = '--', label='4%')  

        if ax is None:
            return fig, ax

    def plot_growth_vs_decarbonization_rates(self):
        pass