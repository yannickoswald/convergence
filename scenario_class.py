import os
import pandas as pd
from country_class import Country

class Scenario():
    """
    Description: 
            A class representing a global North-South
            convergence scenario
    
    Parameters:
        (1) Scenario-parameters
            time_frame       - time it takes to perfect convergence 
            income_goal      - income that countries convergence towards to
            
        (2) Country-data
            data             - real-world data of all countries
            specifies many country-level parameters
    """

    def __init__(self, scenario_params):
        """
        Description: 
                Initialize scenario instance based on scenario parameters.
        Parameters:
                Scenario parameters
        """
        self.end_year = scenario_params["end_year"]
        self.income_goal = scenario_params["income_goal"]
        self.raw_data = self.load_country_data()
        self.countries = self.initialize_countries()  # Use self since this method now belongs to the class

    @classmethod
    def load_country_data(cls):
        """
        Description: 
                Load the data.
        Parameters:
                None
        """
        with open(os.path.join('data', 'data_nowcasted_extended.csv')) as f:
            data = pd.read_csv(f, encoding='unicode_escape')
        return data

    def initialize_countries(self):   
        """
        Description: 
                Initializes countries from raw data. 
                It loops of the list of all given countries and sets the attributes
                dynamically. This is more flexible if all attributes are preset. So 
                there is a **kwargs argument in the Country class. 

        Parameters:
            None
        """
        countries = list()
        for i in range(len(self.raw_data)):
            attributes = self.raw_data.iloc[i].to_dict()
            countries.append(Country(self, **attributes)) ## **necessary to unpack attributes
        return countries