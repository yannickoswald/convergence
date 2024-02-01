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

        self.start_year = 2022  # Assuming the scenario starts in 2023 (2022 is the last year of the data)
        self.current_year = self.start_year
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
    

    def compute_group_growth_rates(self):
        
        """
        Description: 
                Compute the CAGR for the country
        Parameters:
                None
        """

        start_year = self.start_year  # Assuming the scenario starts in 2023
        years_to_end = self.end_year - start_year

        for country in self.countries.values():
            # Compute the CAGR for each decile
            for decile_num in range(1, 11):
                decile_income = getattr(country, f'decile{decile_num}_abs')

                # Compute CAGR
                if decile_income > 0 and years_to_end > 0:
                    cagr = (self.income_goal / decile_income) ** (1 / years_to_end) - 1
                else:
                    cagr = 0  # Assigning 0 if the decile income is 0 or years to end is not positive

                # Store the CAGR value for the decile
                country.cagr_by_decile[f'decile{decile_num}'] = cagr


    def compute_country_scenario_params(self):

        """
        Description: 
                Compute country parameters based on scenario parameters.

                that is 
                 - growth rates for each decile
        Parameters:
                None
        """

        self.compute_group_growth_rates()


    def step(self):
            
            """
            Description: 
                    Compute one scenario step
            Parameters:
                    None
            """

            for country in self.countries.values():
                country.growth()

    def run(self):

        """
        Description: 
                Run the scenario
        Parameters:
                None
        """

        for year in range(self.start_year, self.end_year + 1):
            print("the year is ", year)
            self.current_year = year
            self.step()

    
       