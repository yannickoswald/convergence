from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.ticker import FuncFormatter


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

    def plot_only_deciles(self, country, ax=None):
        """
        Description:
            A method that plots one given country's economic trajectory for all deciles

        Parameters:
            (1) country: the country to be plotted
        """
        country = self.scenario.countries[country]

        ax = ax or plt.gca()

        # Initialize lists to store handles and labels for the legend
        handles = []
        labels = []

        # Plot the household income trajectories for each decile
        for decile_num, decile_data in country.decile_trajectories.items():
            # Convert decile_num to string and get the last character
            # if last two characters are 10, then subset last two characters
            if str(decile_num)[-2:] == '10':
                last_char = str(decile_num)[-2:]
            else:
                last_char = str(decile_num)[-1] 
            print(last_char)
            # Plot and collect the handle
            handle, = ax.plot(list(decile_data.keys()), list(decile_data.values()), label=f'Decile {last_char}')
            # Append handle and label to the lists
            handles.append(handle)
            labels.append(f'Decile {last_char}')

        # Axes labels
        ax.set_xlabel('Year')
        ax.set_ylabel('Cons. exp. $ (PPP) per capita per year')

        # Reverse the handles and labels for the legend to align it with the plot order
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

        # No margins
        ax.margins(0)
        # Set ylims lower bound to 0 but no upper bound
        ax.set_ylim(bottom=0)



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

    def plot_country_population(self, country):
        """
        Description: 
            A method that plots one given country's population trajectory
        
        Parameters:
            (1) country: the country to be plotted
        """ 

        country = self.scenario.countries[country]

        # plot population trajectory
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list(country.population_trajectory.keys()), list(country.population_trajectory.values()))
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
        ax.margins(0)
        ax.set_ylim(bottom=0)
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


    def plot_global_emissions(self, **kwargs):
        
        """
        Description: 
            A method that plots the global emissions trajectory
            
        Parameters:
            **kwargs: Arbitrary keyword arguments including:
                ax (matplotlib.axes.Axes): the axes object to plot on. If not provided, a new figure and axes are created.
                color (str): Color of the emissions trajectory line.
                label (str): Label for the emissions trajectory line.
        """ 


         # Default values for optional parameters
        ax = kwargs.get('ax', None)
        color = kwargs.get('color', "tab:blue")
        label = kwargs.get('label', "Global Emissions")

        # Create a new figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
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
        ax.plot(sorted_years, sorted_emissions_trajectory, color = color, label = label)
        ax.set_xlabel('Year')
        ax.set_ylabel('Emissions (metric tons)')
        ax.margins(0)
        ax.set_ylim(bottom=0)

        # plot linear carbon budget pathway
        years_lin, emissions_lin = self.scenario.compute_linear_carbon_budget_pathway()
        ax.plot(years_lin+2022, emissions_lin*1e9, color="tab:orange", label="Linear Budget") # convert from years to 2022 plus the years required and from gigatonnes to metric tonnes
        # plot exponential carbon budget pathway
        #years_exp, emissions_exp = self.scenario.compute_exponential_carbon_budget_pathway()
        #ax.plot(years_exp+2022, emissions_exp*1e9, color = "tab:red")
        #ax.legend(['Global Emissions'], ['Linear Carbon Budget Pathway'], ['Exponential Carbon Budget Pathway'])


        #plt.tight_layout()
        #plt.show()

    def plot_global_population(self):
        """
        Description: 
            A method that plots the global population trajectory
            
        Parameters:
            None
        """ 
        # Initialize an empty dictionary to store the global population trajectory
        global_population_trajectory = {}

        # Iterate over all countries in the scenario
        for country in self.scenario.countries.values():
            # Iterate over the years in the country's population trajectory
            for year, population in country.population_trajectory.items():
                # If the year is already in the global population trajectory dictionary, add the population value to the existing value
                if year in global_population_trajectory:
                    global_population_trajectory[year] += population
                # Otherwise, create a new entry in the global population trajectory dictionary with the population value
                else:
                    global_population_trajectory[year] = population

        # Sort the global population trajectory by year
        sorted_years = sorted(global_population_trajectory.keys())
        sorted_population_trajectory = [global_population_trajectory[year] for year in sorted_years]

        # Plot the global population trajectory
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_years, sorted_population_trajectory)
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
        ax.margins(0)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    def plot_global_gdp_per_capita(self):

        # plot global gdp per capita but first you must compute total gdp per country, sum this over all countries and then divide by the global population
        # Initialize an empty dictionary to store the global gdp per capita trajectory
        global_gdp_per_capita_trajectory = {}
        # Compute total GDP per country sum globally but per year 
        total_gdp = {}
        for country in self.scenario.countries.values():
            for year, gdp in country.gdppc_trajectory.items():
                if year not in total_gdp:
                    total_gdp[year] = 0
                total_gdp[year] += gdp*country.population_trajectory[year]
        
        # Compute total population per year globally
        total_population = {}
        for country in self.scenario.countries.values():
            for year, population in country.population_trajectory.items():
                if year not in total_population:
                    total_population[year] = 0
                total_population[year] += population

        # Compute global GDP per capita trajectory
        global_gdp_per_capita_trajectory = {year: total_gdp[year] / total_population[year] for year in total_population.keys()}

        # Sort the global GDP per capita trajectory by year
        sorted_years = sorted(global_gdp_per_capita_trajectory.keys())
        sorted_gdp_per_capita_trajectory = [global_gdp_per_capita_trajectory[year] for year in sorted_years]

        # Plot the global GDP per capita trajectory
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_years, sorted_gdp_per_capita_trajectory)
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP per Capita ($)')
        ax.margins(0)
        ax.set_ylim(bottom=0)
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
        ax.axvline(0, color='black', lw=1, alpha=0.5)
        # also plot a vertical line at x=0
        ax.set_xlabel('national cagr 2012-2022 GDP pc ($2017 PPP)')
        ax.set_ylabel('national cagr scen. hh income pc ($2017 PPP)')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        ax.margins(0)
        # shade the region where x > 0 and y > 0 yellow
        #ax.fill_between([0, max_val], 0, max_val, color='yellow', alpha=0.1)
        # shade the region where x < 0 and y > 0 gree
        #ax.fill_between([min_val, 0], 0, max_val, color='green', alpha=0.1)
        # shade the region where x > 0 and y < 0 red
        #ax.fill_between([0, max_val], min_val, 0, color='red', alpha=0.1)
        # shade the region where x < 0 and y < 0 blue
        #ax.fill_between([min_val, 0], min_val, 0, color='blue', alpha=0.1)
        # Dynamically get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate dynamic positions for annotations
        # Using a small percentage of the plot's range to offset the annotations from the edges
        x_offset = (xlim[1] - xlim[0]) * 0.02  # 2% of the x-axis range
        y_offset = (ylim[1] - ylim[0]) * 0.02  # 2% of the y-axis range

        # Annotate quadrants with multiline text
        ax.annotate("now growth,\nmodel growth", 
                    xy=(xlim[1] - x_offset, ylim[1] - y_offset), 
                    xytext=(xlim[1] - x_offset, ylim[1] - y_offset),
                    horizontalalignment='right', verticalalignment='top', fontsize=8)

        ax.annotate("now no growth,\nmodel growth", 
                    xy=(xlim[0] + x_offset, ylim[1] - y_offset), 
                    xytext=(xlim[0] + x_offset, ylim[1] - y_offset),
                    horizontalalignment='left', verticalalignment='top', fontsize=8)

        ax.annotate("now growth,\nno model growth", 
                    xy=(xlim[1] - x_offset, ylim[0] + y_offset), 
                    xytext=(xlim[1] - x_offset, ylim[0] + y_offset),
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        ax.annotate("now no growth,\nno model growth", 
                    xy=(xlim[0] + x_offset, ylim[0] + y_offset), 
                    xytext=(xlim[0] + x_offset, ylim[0] + y_offset),
                    horizontalalignment='left', verticalalignment='bottom', fontsize=8)


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
                        textcoordinates = (growth_trend-0.01, country.cagr_average+0.01)
                    else:
                        textcoordinates = (growth_trend-0.01, country.cagr_average-0.01)
                    ax.annotate(country.code, (growth_trend, country.cagr_average), xytext=textcoordinates, arrowprops=dict(arrowstyle='-', lw=0.5, color='black', alpha=0.5), fontsize=8)
    

        # Function to format tick labels as percentages
        def to_percentage(x, pos):
            # Multiply by 100 and format as an integer, followed by '%'
            return f'{x * 100:.0f}%'

        # Create a formatter object
        percentage_formatter = FuncFormatter(to_percentage)

        # Apply the formatter to the x-axis and y-axis
        ax.xaxis.set_major_formatter(percentage_formatter)
        ax.yaxis.set_major_formatter(percentage_formatter)

    
    def plot_global_carbon_intensity(self, ax = None, color = None, label = None):
        
        """
        Description:
            A method that plots the global carbon intensity trajectory
        
        Parameters:
            None    
        
        """

        # Compute total GDP per year globally use the country.gdppc_trajectory
        total_gdp = {}
        for country in self.scenario.countries.values():
            for year in country.gdppc_trajectory.keys():
            # Compute total GDP per country sum globally but per year
                if year not in total_gdp:
                    total_gdp[year] = 0
                total_gdp[year] += country.gdppc_trajectory[year]*country.population_trajectory[year]

        # Compute total emissions per year globally use the country.emissions_trajectory
        total_emissions = {}
        for country in self.scenario.countries.values():
            for year in country.emissions_trajectory.keys():
                if year not in total_emissions:
                    total_emissions[year] = 0
                total_emissions[year] += country.emissions_trajectory[year]
        

        # Compute global carbon intensity trajectory
        global_carbon_intensity_trajectory = {year: total_emissions[year]*1000 / total_gdp[year] for year in total_gdp.keys()} # convert to kg from metric tons hence *1000

        # Sort the global carbon intensity trajectory by year
        sorted_years = sorted(global_carbon_intensity_trajectory.keys())
        sorted_carbon_intensity_trajectory = [global_carbon_intensity_trajectory[year] for year in sorted_years]

        # Plot the global carbon intensity trajectory
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        if color is not None and label is not None:
            ax.plot(sorted_years, sorted_carbon_intensity_trajectory, color = color, label = label)
        else:
            ax.plot(sorted_years, sorted_carbon_intensity_trajectory)
        ax.set_xlabel('Year')
        ax.set_ylabel('Carbon Intensity in kg per 2017 PPP $ (GDP)')
        ax.margins(0)
        ax.set_ylim(bottom=0, top=max(sorted_carbon_intensity_trajectory) + 0.05)
        if ax is None:
            plt.tight_layout()
            plt.show()








    
