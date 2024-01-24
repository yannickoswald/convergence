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
        with open(os.path.join('data\pip_all_data', 'data_nowcasted_extended.csv')) as f:
            data = pd.read_csv(f, encoding='unicode_escape')
        return data

    def initialize_countries(self):
            
            """
            Description: 
                    Initializes countries from raw data. 
                    It loops over the list of countries identified by country_code or country_name 
                    and sets the attributes dynamically. This is more flexible if the attributes 
                    are not preset. Each column in the dataframe, except for country_code and 
                    country_name, becomes an attribute of the Country instance prefixed with 
                    'country_'. 

            Parameters:
                None
            """
            countries = {}
            # Assuming 'country_code' or 'country_name' column is present in the raw data to identify countries
            for country_identifier in self.raw_data['country_code'].unique():
                country_data = self.raw_data[self.raw_data['country_code'] == country_identifier].iloc[0]
                attributes = country_data.to_dict()
                country_name_or_code = attributes.pop('country_name', attributes.get('country_code'))
                countries[country_name_or_code] = Country(self, **attributes)

            return countries
    
    def compute_country_params(self):
        """
        Description: 
                Compute country parameters based on scenario parameters.
        Parameters:
                None
        """
       