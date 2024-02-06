from matplotlib import pyplot as plt


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
                A class organizing all the plots of a specific scenario
        
        Parameters:
            (1) scenario: the scenario to be plotted
        """ 

        self.scenario = scenario


    
    def plot_specific_country_econ(self, country):
        
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
        sorted_emissions_trajectory = [global_emissions_trajectory[year] for year in sorted_years]

        # Plot the global emissions trajectory
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_years, sorted_emissions_trajectory)
        ax.set_xlabel('Year')
        ax.set_ylabel('Emissions (metric tons)')
        ax.legend(['Global Emissions'])
        ax.margins(0)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()







    
