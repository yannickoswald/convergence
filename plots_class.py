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


        country = self.scenario.countries[country]


        # make plot with two panels
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # on the left panel plot income_hh_trajectory and gdp_pc_trajectory dictionaries where the keys are the years and the values are the income and gdp per capita respectively
        print(list(country.income_hh_trajectory.keys()))
        print(list(country.income_hh_trajectory.values()))

        ax[0].plot(list(country.income_hh_trajectory.keys()), list(country.income_hh_trajectory.values()))
        ax[0].plot(list(country.gdppc_trajectory.keys()), list(country.gdppc_trajectory.values()))
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




    
