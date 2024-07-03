# build code that varies the parameters end_year and income_goal and then plots the results and their effect on total emissions

import matplotlib.pyplot as plt
from scenario_class import Scenario
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import re

class ScenarioSweeper:
    
    """
    Description: 
        A class that sweeps through different scenarios and plots the total Trade-offS for different parameter values. 
        The class also plots the growth vs decarbonization rates for different scenarios.
    
    Parameters:
        (1) scenario: the scenario to be plotted
    """

    def __init__(self, end_year_values,
                       income_goal_values,
                       carbon_budget_values,
                       gdp_assumption_values,
                       pop_growth_assumption_values,
                       tech_evolution_assumption_values,
                       tech_hysteresis_assumption_values,
                       steady_state_high_income_assumption_values,
                       sigmoid_parameters_values,
                       final_improvement_rate,
                       population_hysteresis_assumption_values):

        """
        Description: 
            Initialize the Sweeper class with the parameter values to be swept through.
        
        Parameters:
            (1) scenario: the scenario to be plotted
        """ 
        # store the parameter values to be swept through
        self.end_year_values = end_year_values
        self.income_goal_values = income_goal_values
        self.carbon_budget_values = carbon_budget_values
        #self.hysteresis_tech_progress_values = hysteresis_tech_progress_values
        self.gdp_assumption_values = gdp_assumption_values
        self.pop_growth_assumption_values = pop_growth_assumption_values # Assume different population growth rates
        self.tech_evolution_assumption_values = tech_evolution_assumption_values # Assume  different technological evolution rates
        self.tech_hysteresis_assumption_values = tech_hysteresis_assumption_values # Assume different technological hysteresis rates    
        self.steady_state_high_income_assumption_values = steady_state_high_income_assumption_values # Assume  different steady state high income assumptions
        self.population_hysteresis_assumption_values = population_hysteresis_assumption_values # Assume different population hysteresis assumptions 

        self.sigmoid_parameters_values = sigmoid_parameters_values
        self.final_improvement_rate = final_improvement_rate
        
        # store emissions for each scenario in a dictionary where the key is the scenario specified via the params and the value is the total emissions
        self.total_emissions = {}
        # store the global average growth rate for each scenario in a dictionary where the key is the scenario specified via the params and the value is the global average growth rate
        self.growth_rate_global = {}
        # store the gini coefficient change rate for each scenario in a dictionary where the key is the scenario specified via the params and the value is the gini coefficient change rate
        self.gini_coefficient_change_rate_global = {}
        # store the final emissions for each scenario in a dictionary where the key is the scenario specified via the params and the value is the final emissions
        self.final_emissions = {}
        # national gini coefficients over time where the key is the scenario specified via the params and the value is the national gini coefficients over time
        self.gini_coefficient_national = {}
        # national gdp trajectories over time where the key is the scenario specified via the params and the value is the national gdp trajectories over time
        self.national_gdp_trajectories = {}
        
    def run_scenarios(self):
        
        """
        Description: 
            Iterate, "sweep", over the different parameter values and store the global aggregate variables for each scenario in a dictionary.
        
        Parameters:
            None
        """ 
        # Set the assumptions which a fixed single value for this iteration of sweeping 
        tech_hysteresis_assumption = self.tech_hysteresis_assumption_values[0] # this will be just one value in this iteration but for consistency and generality we loop over all the given values
        gdp_assumption = self.gdp_assumption_values[0] # this will be just one value in this iteration but for consistency and generality we loop over all the given values
        pop_growth_assumption = self.pop_growth_assumption_values[0] # this will be just one value in this iteration but for consistency and generality we loop over all the given values
        tech_evolution_assumption = self.tech_evolution_assumption_values[0]  # this will be just one value in this iteration but for consistency and generality we loop over all the given values
        steady_state_high_income_assumption = self.steady_state_high_income_assumption_values[0]  # this will be just one value in this iteration but for consistency and generality we loop over all the given values  
        population_hysteresis_assumption = self.population_hysteresis_assumption_values[0]



        sigmoid_parameters = self.sigmoid_parameters_values  # this will be just one value in this iteration but for consistency and generality we loop over all the given values
        final_improvement_rate = self.final_improvement_rate  # this will be just one value in this iteration but for consistency and generality we loop over all the given values
        
        
        # Iterate over all possible combinations of variable parameter values
        #for hysteresis_tech_progress in self.hysteresis_tech_progress_values: # this will be perhaps many values
        for carbon_budget in self.carbon_budget_values: # this will be perhaps many values
            for end_year in self.end_year_values: # this will be perhaps many values
                for income_goal in self.income_goal_values: # this will perhaps be many values
                    # Create a new scenario with the current parameter values
                    scenario_params = {
                        "end_year": end_year,
                        "income_goal": income_goal,
                        "carbon_budget": carbon_budget,
                        "gdp_assumption": gdp_assumption,
                        "pop_growth_assumption": pop_growth_assumption,
                        "tech_evolution_assumption": tech_evolution_assumption,
                        "tech_hysteresis_assumption": tech_hysteresis_assumption,
                        "steady_state_high_income_assumption": steady_state_high_income_assumption,
                        "k": sigmoid_parameters[0],
                        "t0": sigmoid_parameters[1],
                        "final_improvement_rate": final_improvement_rate,
                        "population_hysteresis_assumption": population_hysteresis_assumption
                    }
                    
                    scenario = self.create_scenario(scenario_params)
                    scenario.compute_country_scenario_params()

                    # Convert scenario_params dictionary to a tuple of tuples (key, value pairs)
                    scenario_key = tuple(sorted(scenario_params.items()))

                    # Calculate the global average necessary growth rate for the current scenario at the beginning so before the scenario runs
                    global_growth_rate = scenario.compute_average_global_growth_rate()
                    self.growth_rate_global[scenario_key] = global_growth_rate
                    
                    # Run the scenario
                    scenario.run()

                    # Calculate total emissions for the current scenario
                    total_emission = scenario.sum_cumulative_emissions()
                    total_emissions_gigatonnes = total_emission / 1e9  # convert to gigatonnes
                
                    # Store the total emissions in the list
                    self.total_emissions[scenario_key] = total_emissions_gigatonnes / carbon_budget # store the ratio of total emissions to the carbon budget for each scenario

                    # Calculate the gini coefficient change rate for the current scenario (VERY COMPUTATIONALLY EXPENSIVE)
                    #self.gini_coefficient_change_rate_global[scenario_key] = scenario.compute_gini_coefficient_change_rate()

                    self.final_emissions[scenario_key] = scenario.compute_ending_global_emissions()


                    self.gini_coefficient_national[scenario_key] = scenario.store_national_gini_coefficients()

                    self.national_gdp_trajectories[scenario_key] = scenario.store_national_gdp_trajectories()



        return self.total_emissions, self.growth_rate_global, self.gini_coefficient_change_rate_global, self.final_emissions, self.gini_coefficient_national, self.national_gdp_trajectories # self.national_gdp_ppp_pc, self.hh_consumption_pc, self.population
    
    def create_scenario(self, params):
        # Assuming Scenario is a class that takes a dictionary of parameters
        # and has methods compute_country_scenario_params() and run()
        return Scenario(params)
    

   

    def plot_total_emissions_trade_off(self, dependent_var, variables_considered, fixed_color_scale, annotations_plot, colorscaleon, ax=None):
        """
        Description: 
        Plot the corresponding trade-off between the total emissions and the parameters considered.

        Parameters:
            (1) output                 - a dictionary containing the total emissions for each scenario
            (2) variables_considered   - a list of the parameters to be considered in the trade-off plot
            (3) ax                     - optional, axes object to plot on
        """ 

        # Default index
        ax_index = None

        # Check if ax is passed and has a format like 'ax5'
        if ax is not None and isinstance(ax, str) and re.match(r'ax\d+', ax):
            ax_index = int(re.search(r'\d+', ax).group())

        #print("this is the ax_index", ax_index)

        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        # Initialize sets to hold unique values for the variables considered
        x_values_set = set()
        y_values_set = set()

        name_mapping = {"end_year": "End Year",
                        "income_goal": "Income Goal $PPPpc", 
                        "carbon_budget": "Carbon Budget", 
                        "gdp_assumption": "GDP Assumption"}

        # Iterate through the keys to extract variable values
        for key in dependent_var.keys():
            # Assuming key is a tuple of tuples representing key-value pairs
            params_dict = {k: v for k, v in key}  # Correctly convert tuple of tuples into a dictionary
            x_values_set.add(params_dict[variables_considered[0]])
            y_values_set.add(params_dict[variables_considered[1]])

        # Convert sets to sorted lists
        x_values = sorted(x_values_set)
        y_values = sorted(y_values_set)
        # Create meshgrid
        X, Y = np.meshgrid(x_values, y_values)
        # Initialize a 2D array for emissions data
        Z = np.zeros(X.shape)
        #print("this is the Z", Z)
        #print("this is the X", X)
        #print("this is the Y", Y)

        # Populate the Z array with total emissions data
        for key, value in dependent_var.items():
            params_dict = {k: v for k, v in key}  # Convert each key into a dictionary
            x_index = x_values.index(params_dict[variables_considered[0]])
            y_index = y_values.index(params_dict[variables_considered[1]])
            Z[y_index, x_index] = value
        
        #print("this is the Z", Z)

        # Define color ranges for values below and above the threshold
        colors_below = ['#4575b4', '#91bfdb']  # Blue shades for values below 1
        colors_above = ['#fdae61', '#d73027']  # Orange-red shades for values above 1

        # Function to combine both color maps with a threshold
        def combine_cmaps(cmap_below, cmap_above, threshold, data):
            # Create a new colormap that transitions at the threshold
            # Determine the proportion of the threshold within the data range
            min_val, max_val = np.min(data), np.max(data)

            if max_val <= threshold:
                return cmap_below

            threshold_norm = (threshold - min_val) / (max_val - min_val)
            #print("this is the threshold_norm", threshold_norm)

            # Generate colors for each part
            below_colors = cmap_below(np.linspace(0, 1, int(256 * threshold_norm)))
            above_colors = cmap_above(np.linspace(0, 1, 256 - int(256 * threshold_norm)))
            #print("this is the below_colors", below_colors)
            #print("this is the above_colors", above_colors)
            
            # Combine colors at the threshold
            all_colors = np.vstack((below_colors, above_colors))
            combined_cmap = mcolors.LinearSegmentedColormap.from_list('combined_cmap', all_colors)
            
            return combined_cmap
        
        # Create linear segmented colormaps for values below and above the threshold
        cmap_below = mcolors.LinearSegmentedColormap.from_list("below", colors_below)
        cmap_above = mcolors.LinearSegmentedColormap.from_list("above", colors_above)

        #print("this is the cmap_below", cmap_below)
        #print("this is the cmap_above", cmap_above)

        # Combine the color maps with a threshold at 1
        combined_cmap = combine_cmaps(cmap_below, cmap_above, 1, Z)
        #print("this is the combined_cmap", combined_cmap)
        # Initialize contour plot arguments with the custom colormap and normalization
        #print("this is combined_cmap", combined_cmap)
        contourf_kwargs = {
            "levels": 200,  # More levels for a smoother transition
            "cmap": combined_cmap
        }

        # Conditionally add vmin and vmax to the arguments
        #if fixed_color_scale:
        #   contourf_kwargs["vmin"] = 0  # Minimum value of Z for the color scale
        #   contourf_kwargs["vmax"] = 2.5  # Maximum value of Z for the color scale
        # Create figure and axes for plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
            current_axis = ax
            #print("this is the current_axis", current_axis)

        contour = ax.contourf(X, Y, Z, **contourf_kwargs)
        if colorscaleon:
            colorbar = fig.colorbar(contour, ax=ax)
            colorbar.set_label(f'Ratio cum. emissions to 2\u00B0C budget', rotation=270, labelpad=15, fontsize=10)
            # Set the colorbar's tick labels to predefined values
            #colorbar.set_ticks([0.5, 1, 1.5, 2])  # Predefined tick values ## needs to be activated for figure 3 and deactivated for figure 4
        ax.set_xlabel(name_mapping[variables_considered[0]])
        ax.set_ylabel(name_mapping[variables_considered[0]])
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values], rotation=45)
        ax.set_yticks(y_values)
        ax.set_yticklabels([str(y) for y in y_values])
        ax.set_xlim(min(x_values), max(x_values))
        ax.set_ylim(min(y_values), max(y_values))

        ############ ANNONTATIONS ############
        # Demarcate line where the ratio equals 1
        # Manually specify label positions
        
        contour_line_0 = ax.contour(X, Y, Z, levels=[1], colors='white', linestyles='dashed')
        def custom_fmt(x):
                    return '2°C 67%'
        
        if ax_index != 5:     
            pass #ax.clabel(contour_line_0, fmt=custom_fmt, inline=True, fontsize=8)
        else:
            pass

        # Demarcate line where the ratio equals 1.1858190709 2 degree budget with 50% 
        contour_line_1 = ax.contour(X, Y, Z, levels=[1.1858190709], colors='white', linestyles='dashed')
        def custom_fmt2(x):
            return '2°C 50%'
        ax.clabel(contour_line_1, fmt=custom_fmt2, inline=True, fontsize=8)

        # Demarcate line where the ratio equals ROUGHLY 2100/(1150*0.95 - 2*35) = 2.05378973105 for 50% chance to stay below 2.5 degree based on https://www.nature.com/articles/s41558-023-01848-5 fig.4 c
        #contour_line_2 = ax.contour(X, Y, Z, levels=[2.05378973105], colors='white', linestyles='dashed')
        #def custom_fmt2(x):
        #   return '2.5°C 50%'
        #ax.clabel(contour_line_2, fmt=custom_fmt2, inline=True, fontsize=8)

        # Extract paths
        paths_2_degree_budget = contour_line_0.collections[0].get_paths()
        paths_2_degree_budget_50pct = contour_line_1.collections[0].get_paths()

        def extract_coordinates(paths):
            coords_list = []
            for path in paths:
                vertices = path.vertices
                coords_list.append(vertices)  # Each item is an array of [X, Y] coordinates
            return coords_list

        coords_2_degree67 = extract_coordinates(paths_2_degree_budget)
        coords_2_degree50 = extract_coordinates(paths_2_degree_budget_50pct)

        #print("this is the coords_2_degree67", coords_2_degree67)


        # Annotate for the year 2100 and income goal 20000
        try:
            x_pos_2100 = x_values.index(2100)
            y_pos_20000 = y_values.index(20000)
            # Convert positions to actual coordinates on the plot
            x_coord_2100 = x_values[x_pos_2100]
            y_coord_20000 = y_values[y_pos_20000]
            # Annotate the point with a marker
            ax.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)
            # Annotate with text and a straight line pointing to the point
            ax.annotate("2100\n Denmark\n scenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-30, 40), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
        except ValueError:
            print("Specified year or income goal not found in the dataset for the 2100 scenario.")


        # Annotate for the z, level, value for the year 2100 and income goal 20000
        # After defining the Z array and before the other annotations...

        try:
            # Step 1: Find the indices for x = 2100 and y = 20000
            x_pos = x_values.index(2100)
            y_pos = y_values.index(20000)

            # Step 2: Use these indices to find the Z value
            z_value = Z[y_pos, x_pos]

            # Step 3: Annotate this Z value on the plot
            ax.scatter(x_values[x_pos], y_values[y_pos], color='red', s=100, zorder=5)  # Mark the point
            ax.annotate(f"Overshoot {z_value:.2f}", (x_values[x_pos], y_values[y_pos]), textcoords="offset points", xytext=(-40, 0), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'))
        except ValueError as e:
            print("Specified point (2100, 20000) not found in the dataset.")


        # annotate costa rica scenario dot and pure redistr. scenario dot with values
        #COSTA RICA SCENARIO
        ax.scatter(2060, 10000, s=100, c='blue', marker='o', zorder = 5, label='Costa Rica')
        ax.annotate("2060\nCosta Rica\nscenario", (2060, 10000), textcoords="offset points", xytext=(68,33), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')

        #PURE REDISTR. SCENARIO
        #ax.scatter(2060, 7000, s=100, c='black', marker='o', zorder = 5, label='Redistr. 2060')
        #ax.annotate("2060\npure\nredistr.", (2060, 7000), textcoords="offset points", xytext=(-60,15), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')

        #OVERSHOOT COSTA RICA SCENARIO
        try:
            # Step 1: Find the indices for x = 2100 and y = 20000
            x_pos = x_values.index(2060)
            y_pos = y_values.index(10000)

            # Step 2: Use these indices to find the Z value
            z_value = Z[y_pos, x_pos]

            # Step 3: Annotate this Z value on the plot
            #ax.scatter(x_values[x_pos], y_values[y_pos], color='red', s=100, zorder=5)  # Mark the point
            ax.annotate(f"Overshoot {z_value:.2f}", (x_values[x_pos], y_values[y_pos]), textcoords="offset points", xytext=(50, -5), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'))
        except ValueError as e:
            print("Specified point (2060, 10000) not found in the dataset.")

        #OVERSHOOT PURE REDISTR. SCENARIO
        #try:
            # Step 1: Find the indices for x = 2060 and y = 7000
        #    x_pos4 = x_values.index(2060)
         #   y_pos4 = y_values.index(7000)

            # Step 2: Use these indices to find the Z value
          #  z_value4 = Z[y_pos4, x_pos4]
           # print("this is Z", Z)

            # Step 3: Annotate this Z value on the plot
            #ax.scatter(x_values[x_pos4], y_values[y_pos4], color='red', s=100, zorder=5)  # Mark the point
            #ax.annotate(f"Overshoot {z_value4:.2f}", (x_values[x_pos4], y_values[y_pos4]), textcoords="offset points", xytext=(45, -5), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'))
        #except ValueError as e:
         #   print("Specified point (2060, 7000) not found in the dataset.")

        ######## additional annotations that should be only introduced for figure 3 but not figure 4  ########
        if annotations_plot:
            pass

        if ax is None:
            # Return figure and axes for external use
            return fig, ax
    
    def plot_growth_rate_trade_off(self, dependent_var, variables_considered, ax=None):
            
            """
            Description: 
               Plot the corresponding trade-off between the growth rate and the parameters considered.
            
            Parameters:
                (1) dependent_var          - a dictionary containing the growth rate for each scenario
                (2) variables_considered   - a list of the parameters to be considered in the trade-off plot
                (3) ax                     - optional, axes object to plot on
            """ 

            if len(variables_considered) != 2:
                raise ValueError("variables_considered must contain exactly two elements")

            # Initialize sets to hold unique values for the variables considered
            x_values_set = set()
            y_values_set = set()

            name_mapping = {"end_year": "End Year",
                            "income_goal": "Income Goal $PPPpc", 
                            "carbon_budget": "Carbon Budget", 
                            "gdp_assumption": "GDP Assumption"}

            # Iterate through the keys to extract variable values
            for key in dependent_var.keys():
                # Assuming key is a tuple of tuples representing key-value pairs
                params_dict = {k: v for k, v in key}  # Correctly convert tuple of tuples into a dictionary
                x_values_set.add(params_dict[variables_considered[0]])
                y_values_set.add(params_dict[variables_considered[1]])

            # Convert sets to sorted lists
            x_values = sorted(x_values_set)
            y_values = sorted(y_values_set)

            # Create meshgrid
            X, Y = np.meshgrid(x_values, y_values)

            # Initialize a 2D array for growth rate data
            Z = np.zeros(X.shape)

            # Populate the Z array with growth rate data
            for key, value in dependent_var.items():
                params_dict = {k: v for k, v in key}  # Convert each key into a dictionary
                x_index = x_values.index(params_dict[variables_considered[0]])
                y_index = y_values.index(params_dict[variables_considered[1]])
                Z[y_index, x_index] = value

            # Create figure and axes for plotting
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.get_figure()
            contour = ax.contourf(X, Y, Z, levels=50, cmap='inferno')
            
            # Format colorbar labels as percentages
            def to_percentage(x, pos):
                """Convert decimal to percentage string."""
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

            # Demarcate line where the values equal 0, and label it
            contour_line_0 = ax.contour(X, Y, Z, levels=[0], colors='cyan', linestyles='dashed')
            ax.clabel(contour_line_0, fmt=f'0', inline=True, fontsize=8)

            # Demarcate line where the values equal 0.04, and label it
            contour_line_004 = ax.contour(X, Y, Z, levels=[0.04], colors='cyan', linestyles='dashed')
            ax.clabel(contour_line_004, fmt=f'4', inline=True, fontsize=8)


            # Extract paths
            #paths_0 = contour_line_0.collections[0].get_paths()
            #paths_004 = contour_line_004.collections[0].get_paths()

            # Function to extract X, Y coordinates from contour paths
            #def extract_coordinates(paths):
             #   coords_list = []
              #  for path in paths:
               #     vertices = path.vertices
                #    coords_list.append(vertices)  # Each item is an array of [X, Y] coordinates
                #return coords_list

            #coords_0 = extract_coordinates(paths_0)
            #coords_004 = extract_coordinates(paths_004)


            #print("this is the coords_0", coords_0)
            #print("this is the coords_004", coords_004)

            # Annotate for the year 2100 and income goal 20000
            try:
                x_pos_2100 = x_values.index(2100)
                y_pos_20000 = y_values.index(20000)
                # Convert positions to actual coordinates on the plot
                x_coord_2100 = x_values[x_pos_2100]
                y_coord_20000 = y_values[y_pos_20000]

                # Annotate the point with a marker
                ax.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)

                # Annotate with text and a straight line pointing to the point
                ax.annotate("2100 Denmark scenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-80,-20), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
            except ValueError:
                print("Specified year or income goal not found in the dataset for the 2100 scenario.")

            # Annotate for the year 2050 and income goal 9100
            try:
                x_pos_2050 = x_values.index(2060)
                y_pos_9100 = y_values.index(10000)
                # Convert positions to actual coordinates on the plot
                x_coord_2050 = x_values[x_pos_2050]
                y_coord_9100 = y_values[y_pos_9100]

                # Annotate the point with a marker
                ax.scatter(x_coord_2050, y_coord_9100, color='blue', s=100, zorder=5)  # Use a different color for distinction

                # Annotate with text and a straight line pointing to the point
                ax.annotate("2060 Costa Rica scenario", (x_coord_2050, y_coord_9100), textcoords="offset points", xytext=(50,20), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
            except ValueError:
                print("Specified year or income goal not found in the dataset for the 2060 scenario.")

            if ax is None:
                # Return figure and axes for external use
                return fig, ax
            

    def plot_gini_coefficient_change_trade_off(self, dependent_var, variables_considered, ax=None):
        
        """
        Description: 
           Plot the corresponding trade-off between the gini coefficient change and the parameters considered.
        
        Parameters:
            (1) dependent_var          - a dictionary containing the gini coefficient change for each scenario
            (2) variables_considered   - a list of the parameters to be considered in the trade-off plot
            (3) ax                     - optional, axes object to plot on
        """ 

        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        # Initialize sets to hold unique values for the variables considered
        x_values_set = set()
        y_values_set = set()

        name_mapping = {"end_year": "End Year",
                        "income_goal": "Income Goal $PPPpc", 
                        "carbon_budget": "Carbon Budget", 
                        "gdp_assumption": "GDP Assumption"}

        # Iterate through the keys to extract variable values
        for key in dependent_var.keys():
            # Assuming key is a tuple of tuples representing key-value pairs
            params_dict = {k: v for k, v in key}

          

    def plot_final_emissions_trade_off(self, dependent_var, variables_considered, annotations_plot, colorscaleon, ax=None):
         
        """
        Description: 
           Plot the corresponding trade-off between the total emissions and the parameters considered.
        
        Parameters:
            (1) output                 - a dictionary containing the total emissions for each scenario
            (2) variables_considered   - a list of the parameters to be considered in the trade-off plot
            (3) ax                     - optional, axes object to plot on
        """ 

        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        # Initialize sets to hold unique values for the variables considered
        x_values_set = set()
        y_values_set = set()

        name_mapping = {"end_year": "End Year",
                        "income_goal": "Income Goal $PPPpc", 
                        "carbon_budget": "Carbon Budget", 
                        "gdp_assumption": "GDP Assumption"}

        # Iterate through the keys to extract variable values
        for key in dependent_var.keys():
            # Assuming key is a tuple of tuples representing key-value pairs
            params_dict = {k: v for k, v in key}  # Correctly convert tuple of tuples into a dictionary
            x_values_set.add(params_dict[variables_considered[0]])
            y_values_set.add(params_dict[variables_considered[1]])

        # Convert sets to sorted lists
        x_values = sorted(x_values_set)
        y_values = sorted(y_values_set)
        # Create meshgrid
        X, Y = np.meshgrid(x_values, y_values)
        # Initialize a 2D array for emissions data
        Z = np.zeros(X.shape)
        #print("this is the Z", Z)
        #print("this is the X", X)
        #print("this is the Y", Y)

        # Populate the Z array with total emissions data
        for key, value in dependent_var.items():
            params_dict = {k: v for k, v in key}  # Convert each key into a dictionary
            x_index = x_values.index(params_dict[variables_considered[0]])
            y_index = y_values.index(params_dict[variables_considered[1]])
            Z[y_index, x_index] = value
        
        #print("this is the Z", Z)
        # Define a simple continuous colormap
        simple_cmap = plt.cm.viridis  # Using a pre-existing colormap

        # Conditionally add vmin and vmax to the arguments
        #if fixed_color_scale:
         #   contourf_kwargs["vmin"] = 0  # Minimum value of Z for the color scale
         #   contourf_kwargs["vmax"] = 2.5  # Maximum value of Z for the color scale
        # Create figure and axes for plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        # Using the simple colormap directly in contourf
        contour = ax.contourf(X, Y, Z, levels=200, cmap=simple_cmap)
        if colorscaleon:
            colorbar = fig.colorbar(contour, ax=ax)
            colorbar.set_label(f'Pct of emissions left compared to 2022', rotation=270, labelpad=15, fontsize=8)
            # Set the colorbar's tick labels to predefined values
            #colorbar.set_ticks([0.5, 1, 1.5, 2])  # Predefined tick values ## needs to be activated for figure 3 and deactivated for figure 4
            # Define a function to format the ticks with a percentage sign
            def format_tick(x, _):
                return f'{x * 100:.0f}%'

            # Applying the formatter to the colorbar
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

        ############ ANNONTATIONS ############
        #    #Extract paths for contour lines that achieve 80% emissions reductions with respect to 2022 to 
        # get a feeling for "good scenarios overall"
        contour_line_0 = ax.contour(X, Y, Z, levels=[0.2], colors='orange', linestyles='dashed')
        ax.clabel(contour_line_0, inline=True, fmt = {0.2: "<20%"}, fontsize=10)

        contour_line_00 = ax.contour(X, Y, Z, levels=[0.01], colors='orange', linestyles='dashed')
        ax.clabel(contour_line_00, inline=True, fmt = {0.01: "<1%"}, fontsize=10)

        #Extract paths
        paths_0 = contour_line_0.collections[0].get_paths()
        #paths_004 = contour_line_004.collections[0].get_paths()

        # Function to extract X, Y coordinates from contour paths
        def extract_coordinates(paths):
            coords_list = []
            for path in paths:
               vertices = path.vertices
               coords_list.append(vertices)  # Each item is an array of [X, Y] coordinates
            return coords_list

        coords_0 = extract_coordinates(paths_0)
        


        print("this is the coords_0", coords_0)

        # Annotate for the year 2100 and income goal 20000
        try:
            x_pos_2100 = x_values.index(2100)
            y_pos_20000 = y_values.index(20000)
            # Convert positions to actual coordinates on the plot
            x_coord_2100 = x_values[x_pos_2100]
            y_coord_20000 = y_values[y_pos_20000]
            # Annotate the point with a marker
            ax.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)
            # Annotate with text and a straight line pointing to the point
            ax.annotate("2100\n Denmark\n scenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-30, 40), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"), color='white')
        except ValueError:
            print("Specified year or income goal not found in the dataset for the 2100 scenario.")


        # Annotate for the z, level, value for the year 2100 and income goal 20000 for the denmark scenario
        # After defining the Z array and before the other annotations...

        try:
            # Step 1: Find the indices for x = 2100 and y = 20000
            x_pos = x_values.index(2100)
            y_pos = y_values.index(20000)

            # Step 2: Use these indices to find the Z value
            z_value = Z[y_pos, x_pos]

            # Step 3: Annotate this Z value on the plot
            ax.scatter(x_values[x_pos], y_values[y_pos], color='red', s=100, zorder=5)  # Mark the point
            ax.annotate(f"{z_value * 100:.0f}%", (x_values[x_pos], y_values[y_pos]), textcoords="offset points", xytext=(-30, 5), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")
        except ValueError as e:
            print("Specified point (2100, 20000) not found in the dataset.")

        #  Annotate for the z, level, value for the year 2060 and income goal 10000 i.e. costa rica scenario
        # After defining the Z array and before the other annotations...

        try:
            # Step 1: Find the indices for x = 2100 and y = 20000
            x_pos2 = x_values.index(2060)
            y_pos2 = y_values.index(10000)

            # Step 2: Use these indices to find the Z value
            z_value2 = Z[y_pos2, x_pos2]

            # Step 3: Annotate this Z value on the plot
            ax.scatter(x_values[x_pos2], y_values[y_pos2], color='red', s=100, zorder=5)  # Mark the point
            ax.annotate(f"{z_value2 * 100:.0f}%", (x_values[x_pos2], y_values[y_pos2]), textcoords="offset points", xytext=(30, 0), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")
        except ValueError as e:
            print("Specified point (2060, 10000) not found in the dataset.")

        # add another scenario costa rica level income but until 2100 for comparison as well as the denmark scenario until 2060 for comparison
        try:
            #costa rica 2100
            # Step 1: Find the indices for x = 2100 and y = 20000
            x_pos3 = x_values.index(2100)
            y_pos3 = y_values.index(10000)

            # Step 2: Use these indices to find the Z value
            z_value3 = Z[y_pos3, x_pos3]
            #Denmark 2060
            # Step 1: Find the indices for x = 2100 and y = 20000
            x_pos4 = x_values.index(2060)
            y_pos4 = y_values.index(20000)
            # Step 2: Use these indices to find the Z value
            z_value4 = Z[y_pos4, x_pos4]

            # Step 3: Annotate these y values
            # costa rica 2100
            ax.scatter(x_values[x_pos3], y_values[y_pos3], color='darkblue', s=100, zorder=5)  # Mark the point
            ax.annotate(f"{z_value3 * 100:.0f}%", (x_values[x_pos3], y_values[y_pos3]), textcoords="offset points", xytext=(-20, -10), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")
            # denmark 2060
            ax.scatter(x_values[x_pos4], y_values[y_pos4], color='darkred', s=100, zorder=5)  # Mark the point
            ax.annotate(f"{z_value4 * 100:.0f}%", (x_values[x_pos4], y_values[y_pos4]), textcoords="offset points", xytext=(0, 20), ha='center', fontsize=8, arrowprops=dict(arrowstyle="-", color='black'), color ="white")


            # draw dotted line also between costa rica 2060 and 2100 as well as denmark 2060 and 2100
            ax.plot([2060, 2100], [10000, 10000], color='blue', linestyle='dotted')
            ax.plot([2060, 2100], [20000, 20000], color='red', linestyle='dotted')


        except ValueError as e:
            print("Specified point (2060, 10000) not found in the dataset.")
        

        ######## additional annotations that should be only introduced for figure 3 but not figure 4  ########
        if annotations_plot:

               ######## add extracted growth rates feasible regions lines
            # Plotting the lines for level 0
            coords_growth1 = np.array([[2040., 7091.76725433],
                                [2060., 7104.1157445],
                                [2078.19032277, 7107.60378143],
                                [2081.80967737, 7108.11988921],
                                [2100., 7109.81960993]])
            ax.plot(coords_growth1[:, 0], coords_growth1[:, 1], color = "cyan", linestyle = '--', label='0%')  # 'w--' for white dashed line
            
            # Plotting the lines for level 0.04
            coords_growth2 = np.array([[2040., 13763.74664383],
                                [2044.7325622, 15000.],
                                [2053.52721919, 20000.],
                                [2057.06128359, 24277.62144207],
                                [2058.02526381, 25748.84170682],
                                [2060., 29776.31871013],
                                [2060.31708066, 30000.]])
            ax.plot(coords_growth2[:, 0], coords_growth2[:, 1], color = "cyan", linestyle = '--', label='4%')  # 'w--' for white dashed line

            

           

        if ax is None:
            # Return figure and axes for external use
            return fig, ax



    def plot_growth_vs_decarbonization_rates(self):
        pass






