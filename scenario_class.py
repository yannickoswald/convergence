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
            carbon_budget    - the amount of carbon that can be emitted
            gdp_assumption   - fundamental model assumption about the ratio of GDP to household income
                               Two possible values: constant_ratio, model_ratios
                               If this is constant_ratio it just applies the empirically observed ratio of countries.
                               If this is model_ratio it applies piecewise linear model to the ratio of GDP to household income
            
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
        self.end_year = scenario_params["end_year"]
        self.income_goal = scenario_params["income_goal"]
        self.raw_data = self.load_country_data()
        self.countries = self.initialize_countries()  # Use self since this method now belongs to the class
        self.population_growth_rates = self.load_population_growth_rates()

        # Check if 'gdp_assumption' in scenario_params is a string and one of the two specified values
        gdp_assumption_value = scenario_params.get("gdp_assumption", None)
        valid_values = ["constant_ratio", "model_ratio"]

        if isinstance(gdp_assumption_value, str) and gdp_assumption_value in valid_values:
            self.gdp_assumption = gdp_assumption_value
        else:
            # Handle the case where 'gdp_assumption' is not a string or not one of the specified values
            raise ValueError(f"gdp_assumption must be a string and one of {valid_values}")
        

    @staticmethod
    def load_country_data():
        """
        Description: 
                Load the country level data with most variables being values for 2022.
        Parameters:
                None
        """
        try:
            file_path = os.path.join('data', 'pip_all_data', 'data_nowcasted_extended.csv')
            data = pd.read_csv(file_path, encoding='unicode_escape')
            return data
        except FileNotFoundError:
            print("File not found. Please ensure the file path is correct.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    @staticmethod
    def load_population_growth_rates():
        """
        Description: 
                Load the population growth rates.
        Parameters:
                None
        """

        try:
            file_path = os.path.join('data', 'pip_all_data', 'population_growth_rates.csv')
            data = pd.read_csv(file_path, sep=";", encoding='unicode_escape')
            data.set_index('code', inplace=True)
            return data
        except FileNotFoundError:
            print("File not found. Please ensure the file path is correct.")
        except KeyError:
            print("Error setting index: 'code' column not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    

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
                Compute the economic CAGR for the country
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
                country.economic_growth()
                country.population_growth()
                country.year += 1 # increase the year by one

    def run(self):

        """
        Description: 
                Run the scenario
        Parameters:
                None
        """

        for year in range(self.start_year, self.end_year): # the scenario must run the change from 2022 to 2023 and as last step the change from endyear - 1 to endyear, it cannot run through endyear again
            self.step()

    
       