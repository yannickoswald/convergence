from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

class Plots():

    """
    Description: 
         A class organizing all the plots of a specific scenario
    
    Parameters:
        (1) scenario: the scenario to be plotted
    """


    def __init__(self, scenario):

        """
        Description: 
            Initialize the Plots class with a specific scenario
        
        Parameters:
            (1) scenario: the scenario to be plotted
        """ 

        self.scenario = scenario


    
    def plot_country_economy(self, country):
        
        """
        Description: 
            A method that plots one given country's economic trajectory and for all deciles
        
        Parameters:
            (1) country: the country to be plotted
        """ 

        country = self.scenario.countries[country]


        # make plot with two panels
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # on the left panel plot income_hh_trajectory and gdp_pc_trajectory dictionaries where the keys are the years and the values are the income and gdp per capita respectively
        print(list(country.income_hh_trajectory.keys()))
        print(list(country.income_hh_trajectory.values()))

        ax[0].plot(list(country.income_hh_trajectory.keys()), list(country.income_hh_trajectory.values()), color = "tab:orange")
        ax[0].plot(list(country.gdppc_trajectory.keys()), list(country.gdppc_trajectory.values()), color = "tab:blue")
        # axes labels
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('$ (PPP)')
        # legend
        ax[0].legend(['HH disposable income per capita', 'GDP per capita'])
        # no margins
        ax[0].margins(0)
        # set ylims lower bound to 0 but no upper bound
        ax[0].set_ylim(bottom=0)
    
    
        # on the right panel plot the household income trajectories for each decile iterating over the dict self.decile_trajectories where each entry is another dict with the years as keys and the decile mean income as values
        for decile in country.decile_trajectories.items():
            ax[1].plot(list(decile[1].keys()), list(decile[1].values()))
        # axes labels
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('$ (PPP)')
        # legend as loop over deciles but outside the plot and also the order should be reversed
        ax[1].legend([f'Decile {decile_num}' for decile_num in range(1, 11)], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # no margins
        ax[1].margins(0)
         # set ylims lower bound to 0 but no upper bound
        ax[1].set_ylim(bottom=0)
        # tight layout
        plt.tight_layout()
        # show plot
        plt.show()



    def plot_country_emissions(self, country):
        """
        Description: 
            A method that plots one given country's emissions trajectory
        
        Parameters:
            (1) country: the country to be plotted
        """ 

        country = self.scenario.countries[country]

        # make plot with two panels
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        # plot carbon intensity trajectory on panel A
        ax[0].plot(list(country.carbon_intensity_trajectory.keys()), list(country.carbon_intensity_trajectory.values()))
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Carbon Intensity')
        ax[0].margins(0)
        ax[0].set_ylim(bottom=0)

        # plot emissions trajectory on panel B
        ax[1].plot(list(country.emissions_trajectory.keys()), list(country.emissions_trajectory.values()))
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('Emissions (metric tons)')
        ax[1].margins(0)
        ax[1].set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()




    def plot_global_economy(self):
        """
        Description: 
            A method that plots the global economy trajectory
        
        Parameters:
            None
        """ 
        # sum gdp over all countries in the given scenario and also household income and plot the two trajectories over the years
        # Initialize empty dictionaries to store the global trajectories
        global_gdp_trajectory = {}
        global_income_hh_trajectory = {}

        # Iterate over all countries in the scenario
        for country in self.scenario.countries.values():
            # Iterate over the years in the country's GDP per capita trajectory
            for year, gdp_pc in country.gdppc_trajectory.items():
                # Multiply the GDP per capita value with the country's population at the given year
                gdp_total = gdp_pc * country.population_trajectory[year]

                # If the year is already in the global trajectory dictionary, add the GDP per capita value to the existing value
                if year in global_gdp_trajectory:
                    global_gdp_trajectory[year] +=  gdp_total
                # Otherwise, create a new entry in the global trajectory dictionary with the GDP per capita value
                else:
                    global_gdp_trajectory[year] =  gdp_total

            # Iterate over the years in the country's household income trajectory
            for year, income_hh in country.income_hh_trajectory.items():
                # Multiply the household income value with the country's population at the given year
                income_hh_total = income_hh*country.population_trajectory[year]

                # If the year is already in the global trajectory dictionary, add the household income value to the existing value
                if year in global_income_hh_trajectory:
                    global_income_hh_trajectory[year] += income_hh_total
                # Otherwise, create a new entry in the global trajectory dictionary with the household income value
                else:
                    global_income_hh_trajectory[year] = income_hh_total

        # Sort the global trajectories by year
        sorted_years = sorted(global_gdp_trajectory.keys())
        sorted_gdp_trajectory = [global_gdp_trajectory[year] for year in sorted_years]
        sorted_income_hh_trajectory = [global_income_hh_trajectory[year] for year in sorted_years]

        # Plot the global trajectories
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_years, sorted_gdp_trajectory)
        ax.plot(sorted_years, sorted_income_hh_trajectory)
        ax.set_xlabel('Year')
        ax.set_ylabel('$ (PPP)')
        ax.legend(['Global GDP (Total)', 'Global HH disposable income (Total)'])
        ax.margins(0)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()


    def plot_global_emissions(self):
        """
        Description: 
            A method that plots the global emissions trajectory
            
        Parameters:
            None
        """ 
        # Initialize an empty dictionary to store the global emissions trajectory
        global_emissions_trajectory = {}

        # Iterate over all countries in the scenario
        for country in self.scenario.countries.values():
            # Iterate over the years in the country's GDP per capita trajectory
            for year, gdp_pc in country.gdppc_trajectory.items():
                # Multiply the GDP per capita value with the country's population at the given year
                gdp_total = gdp_pc * country.population_trajectory[year]

                # Multiply the GDP total with the country's carbon intensity at the given year
                emissions_total = gdp_total * country.carbon_intensity_trajectory[year]

                # If the year is already in the global emissions trajectory dictionary, add the emissions value to the existing value
                if year in global_emissions_trajectory:
                    global_emissions_trajectory[year] += emissions_total
                # Otherwise, create a new entry in the global emissions trajectory dictionary with the emissions value
                else:
                    global_emissions_trajectory[year] = emissions_total

        # Sort the global emissions trajectory by year
        sorted_years = sorted(global_emissions_trajectory.keys())
        sorted_emissions_trajectory = [global_emissions_trajectory[year]/1000 for year in sorted_years] # convert to metric tons from kg

        # Plot the global emissions trajectory
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_years, sorted_emissions_trajectory)
        ax.set_xlabel('Year')
        ax.set_ylabel('Emissions (metric tons)')
        ax.margins(0)
        ax.set_ylim(bottom=0)

        # plot linear carbon budget pathway
        years_lin, emissions_lin = self.scenario.compute_linear_carbon_budget_pathway()
        ax.plot(years_lin+2022, emissions_lin*1e9, color = "tab:orange") # convert from years to 2022 plus the years required and from gigatonnes to metric tonnes
        # plot exponential carbon budget pathway
        #years_exp, emissions_exp = self.scenario.compute_exponential_carbon_budget_pathway()
        #ax.plot(years_exp+2022, emissions_exp*1e9, color = "tab:red")
        #ax.legend(['Global Emissions'], ['Linear Carbon Budget Pathway'], ['Exponential Carbon Budget Pathway'])


        plt.tight_layout()
        plt.show()


    def plot_growth_rates_distribution(self, ax = None):
        """
        Description: 
            A method that plots the distribution of necessary average economic growth rates per country to achieve the income_goal
            
        Parameters:
            None

        """ 

        # for the given scenario plot the distribution of necessary average economic growth rates per country to achieve the income_goal
        # for all countries in the scenario loop over the countries and plot the distribution of growth rates via the country.cagr_average
        # Initialize an empty list to store the growth rates
        growth_rates = []

        # Iterate over all countries in the scenario
        for country in self.scenario.countries.values():
            # Append the average growth rate to the growth_rates list
            growth_rates.append(country.cagr_average)
        # Plot the distribution of growth rates using kernel density estimation
        sns.kdeplot(growth_rates, shade=True, ax=ax)
        ax.set_xlabel('Average Economic Growth Rate')
        ax.set_ylabel('Density')

        # Return the figure and axes object for further use
        #return ax


    def plot_growth_rates_vs_reality(self, ax = None):

        """
        Description: 
            A method that plots the distribution of necessary average economic growth rates per country to achieve the income_goal
            
        Parameters:
            None
        """

        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, 'data', 'pip_all_data', 'gdp_pc_empirical_trend.csv')
            data_empirical = pd.read_csv(file_path, sep=";", encoding='unicode_escape')
            print(data_empirical)
        except FileNotFoundError:
            print("File not found. Please ensure the file path is correct.")
            return  # Exit the function if file is not found
        except Exception as e:
            print(f"An error occurred: {e}")
            return  # Exit the function if file is not found


        # Define colors for different regions, this is just an example
        # Update it according to your actual regions

        region_colors = {
           "Sub-Saharan Africa": "tab:blue",
           "Europe & Central Asia": "tab:orange",
           "Other High Income Countries": "tab:green",
           "South Asia": "tab:red",
           "Latin America & Caribbean": "tab:purple",
           "East Asia & Pacific": "tab:brown",
           "Middle East & North Africa": "tab:pink"
        }
    
        # loop over the countries in the scenario and plot the necessary average economic growth rates vs the empirical growth rates
        plotted_regions = set() # Keep track of which regions have already been plotted to avoid duplicate legend entries
        for country in self.scenario.countries.values():
            country_data = data_empirical[data_empirical['country_code'] == country.code]
            if not country_data.empty: # Check if the country is in the empirical data
                growth_trend = country_data["growth_trend_2012_to_2022"].iloc[0]
                region_color = region_colors.get(country.region, 'gray')  # Use 'gray' if region not found
                
                # Check if the region has already been plotted
                if country.region not in plotted_regions:
                    ax.scatter(growth_trend, country.cagr_average, color=region_color, label=country.region, alpha=0.7)
                    plotted_regions.add(country.region)
                else:
                    ax.scatter(growth_trend, country.cagr_average, color=region_color, alpha=0.7)  # No label for duplicates

        # Plot x=y line and other plot elements
        min_val, max_val = ax.get_xlim()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.5, label = "x=y")  # k-- is for black dashed line
        # also plot a horizontal line at y=0
        ax.axhline(0, color='black', lw=1, alpha=0.5)
        # also plot a vertical line at x=0
        ax.set_xlabel('cagr empirical 2012-2022')
        ax.set_ylabel('required growth rate model')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        ax.margins(0)

        # annotate the USA and China and India in the plot
        for country in self.scenario.countries.values():
            if country.code == "USA" or country.code == "CHN" or country.code == "IND" or country.code == "NGA" or country.code == "BRA":
                country_data = data_empirical[data_empirical['country_code'] == country.code]
                if not country_data.empty: # Check if the country is in the empirical data
                    growth_trend = country_data["growth_trend_2012_to_2022"].iloc[0]
                    # annotate the country with an arrow without arrow head
                    if country.code == "IND":
                        textcoordinates = (growth_trend+0.01, country.cagr_average+0.02)
                    elif country.code == "BRA":
                        textcoordinates = (growth_trend-0.01, country.cagr_average)
                    elif country.code == "NGA":
                        textcoordinates = (growth_trend, country.cagr_average+0.01)
                    else:
                        textcoordinates = (growth_trend-0.01, country.cagr_average-0.01)
                    ax.annotate(country.code, (growth_trend, country.cagr_average), xytext=textcoordinates, arrowprops=dict(arrowstyle='-', lw=0.5, color='black', alpha=0.5), fontsize=8)
    












    
