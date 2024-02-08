# build code that varies the parameters end_year and income_goal and then plots the results and their effect on total emissions

import matplotlib.pyplot as plt
from scenario_class import Scenario
import numpy as np

class ScenarioSweeper:
    
    """
    Description: 
        A class that sweeps through different scenarios and plots the total Trade-offS for different parameter values. 
        The class also plots the growth vs decarbonization rates for different scenarios.
    
    Parameters:
        (1) scenario: the scenario to be plotted
    """

    def __init__(self, end_year_values, income_goal_values, carbon_budget_values, gdp_assumption_values):


        """
        Description: 
            Initialize the Sweeper class with the parameter values to be swept through.
        
        Parameters:
            (1) scenario: the scenario to be plotted
        """ 


        self.end_year_values = end_year_values
        self.income_goal_values = income_goal_values
        self.carbon_budget_values = carbon_budget_values
        self.gdp_assumption_values = gdp_assumption_values
        # store emissions for each scenario in a dictionary where the key is the scenario specified via the params and the value is the total emissions
        self.total_emissions = {}
        
    def run_scenarios(self):
        
        """
        Description: 
            Iterate, "sweep", over the different parameter values and store the global aggregate variables for each scenario in a dictionary.
        
        Parameters:
            None
        """ 
        # Iterate over all possible combinations of parameter values
        for gdp_assumption in self.gdp_assumption_values:
            for carbon_budget in self.carbon_budget_values:
                for end_year in self.end_year_values:
                    for income_goal in self.income_goal_values:
                        # Create a new scenario with the current parameter values
                        scenario_params = {
                            "end_year": end_year,
                            "income_goal": income_goal,
                            "carbon_budget": carbon_budget,
                            "gdp_assumption": gdp_assumption
                        }
                        scenario = self.create_scenario(scenario_params)
                        scenario.compute_country_scenario_params()
                        scenario.run()

                        # Calculate total emissions for the current scenario
                        total_emission = scenario.sum_cumulative_emissions()
                        total_emissions_gigatonnes = total_emission / 1e9  # convert to gigatonnes
                        # Store the total emissions in the list
                        # Convert scenario_params dictionary to a tuple of tuples (key, value pairs)
                        scenario_key = tuple(sorted(scenario_params.items()))
                        self.total_emissions[scenario_key] = total_emissions_gigatonnes / carbon_budget # store the ratio of total emissions to the carbon budget for each scenario

        return self.total_emissions
    
    def create_scenario(self, params):
        # Assuming Scenario is a class that takes a dictionary of parameters
        # and has methods compute_country_scenario_params() and run()
        return Scenario(params)

    def plot_total_emissions_trade_off(self, output, variables_considered):
        
        """
        Description: 
           Plot the corresponding trade-off between the total emissions and the parameters considered.
        
        Parameters:
            (1) output                 - a dictionary containing the total emissions for each scenario
            (2) variables_considered   - a list of the parameters to be considered in the trade-off plot
        """ 

        if len(variables_considered) != 2:
            raise ValueError("variables_considered must contain exactly two elements")

        # Initialize sets to hold unique values for the variables considered
        x_values_set = set()
        y_values_set = set()

        name_mapping = {"end_year": "End Year",
                        "income_goal": "Income Goal", 
                        "carbon_budget": "Carbon Budget", 
                        "gdp_assumption": "GDP Assumption"}


        # Iterate through the keys to extract variable values
        for key in output.keys():
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

        # Populate the Z array with total emissions data
        for key, emission in output.items():
            params_dict = {k: v for k, v in key}  # Convert each key into a dictionary
            x_index = x_values.index(params_dict[variables_considered[0]])
            y_index = y_values.index(params_dict[variables_considered[1]])
            Z[y_index, x_index] = emission

        # Plotting the contour map
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(X, Y, Z, levels=15, cmap='inferno')
        colorbar = plt.colorbar(contour)
        # Label the colorbar
        colorbar.set_label(f'Ratio of global emissions to 2\u00B0C budget', rotation=270, labelpad=15)


        #plt.title('Total Emissions by Scenario')
        plt.xlabel(name_mapping[variables_considered[0]])
        plt.ylabel(name_mapping[variables_considered[1]])

        # Adjusting tick labels to show actual numeric values
        plt.xticks(ticks=x_values, labels=[str(x) for x in x_values], rotation=45)
        plt.yticks(ticks=y_values, labels=[str(y) for y in y_values])

        plt.xlim(min(x_values), max(x_values))
        plt.ylim(min(y_values), max(y_values))


        # Annotate for the year 2100 and income goal 20000
        try:
            x_pos_2100 = x_values.index(2100)
            y_pos_20000 = y_values.index(20000)
            # Convert positions to actual coordinates on the plot
            x_coord_2100 = x_values[x_pos_2100]
            y_coord_20000 = y_values[y_pos_20000]

            # Annotate the point with a marker
            plt.scatter(x_coord_2100, y_coord_20000, color='red', s=100, zorder=5)

            # Annotate with text and a straight line pointing to the point
            plt.annotate("Roser's Denmark scenario", (x_coord_2100, y_coord_20000), textcoords="offset points", xytext=(-80,-20), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"))
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
            plt.scatter(x_coord_2050, y_coord_9100, color='blue', s=100, zorder=5)  # Use a different color for distinction

            # Annotate with text and a straight line pointing to the point
            plt.annotate("2060 Costa Rica scenario", (x_coord_2050, y_coord_9100), textcoords="offset points", xytext=(50,-20), ha='center', arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"))
        except ValueError:
            print("Specified year or income goal not found in the dataset for the 2060 scenario.")

        plt.show()

    def plot_growth_vs_decarbonization_rates(self):
        pass






