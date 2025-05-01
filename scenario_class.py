import os
import pandas as pd
import numpy as np
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


            run_until_2100             - flag indicating if simulation should continue until 2100 regardless of convergence

    """

    def __init__(self, scenario_params):
        """
        Description: 
                Initialize scenario instance based on scenario parameters.
        Parameters:
                Scenario parameters
        """
        # Set the key scenario parameters
        self.start_year = 2022  # Assuming the scenario starts in 2023 (2022 is the last year of the data)
        self.end_year = scenario_params["end_year"]
        self.income_goal = scenario_params["income_goal"]
        self.carbon_budget = scenario_params["carbon_budget"]
        #self.hysteresis_tech_progress = scenario_params["hysteresis_tech_progress"]
        self.k = scenario_params["k"]
        self.t0 = scenario_params["t0"]
        self.final_improvement_rate = scenario_params["final_improvement_rate"]

    
        # Load the country data
        self.raw_data = self.load_country_data()
        self.countries = self.initialize_countries()  # Use self since this method now belongs to the class
        self.population_growth_rates = self.load_population_growth_rates()
        self.linear_yearly_carbon_budget = self.compute_linear_carbon_budget_pathway() # this outputs a tuple (years, emissions)
        self.total_population = self.compute_current_global_population() # this is the total population in the current year

        # Check on key model assumptions
        self.gdp_assumption = self.validate_assumption(scenario_params, "gdp_assumption", ["constant_ratio", "model_ratio"])
        self.pop_growth_assumption = self.validate_assumption(scenario_params, "pop_growth_assumption", ["UN_medium", "semi_log_model", "semi_log_model_elasticity"])
        self.tech_evolution_assumption = self.validate_assumption(scenario_params, "tech_evolution_assumption", ["plausible", "necessary"])
        self.tech_hysteresis_assumption = self.validate_assumption(scenario_params, "tech_hysteresis_assumption", ["on", "off"])
        self.steady_state_high_income_assumption = self.validate_assumption(scenario_params, "steady_state_high_income_assumption", ["on", "off", "on_with_growth"])
        self.population_hysteresis_assumption = self.validate_assumption(scenario_params, "population_hysteresis_assumption", ["on", "off"])
        self.cdr_assumption = self.validate_assumption(scenario_params, "cdr_assumption", ["on", "off"])

        # Initialize global outcomes storage
        self.gini_data = {"years": [], "population": [], "income": []}

        # initialize the national gini coefficients holding place
        self.national_gini_coefficients = None

        # New parameters for CDR and extended simulation
        self.run_until_2100 = scenario_params.get("run_until_2100", False) # on or off
        self.cdr_assumption = scenario_params.get("cdr_assumption", False) # on or off
        #print("this is the cdr assumption in scenario class", self.cdr_assumption)
        self.cdr_level_2100 = scenario_params.get("cdr_level_2100", 0) # in gigatons
        self.cdr_global_level_trajectory = self.compute_global_cdr_trajectory() if self.cdr_assumption == "on" else None # this is the global cdr trajectory


    @staticmethod 
    def validate_assumption(params, key, valid_values):
        """
        Validates if the value for a given key assumption in params is one of the valid_values.
        Raises ValueError if the value is not a string or not in valid_values.
        """
        value = params.get(key)
        if isinstance(value, str) and value in valid_values:
            return value
        else:
            raise ValueError(f"{key} must be a string and one of {valid_values}")   
        
        
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
            data = pd.read_csv(file_path, sep=",", encoding='unicode_escape')
            data.set_index('code', inplace=True)
            # replace 0 values with very small value to avoid division by zero
            data = data.replace(0, 1e-9)          
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
                Compute the economic CAGR for the country per decile or another specified income group division.
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

    def compute_average_growth_rates(self):
                
        """
        Description: 
                Compute the average growth rate for each country.
        Parameters:
                None
        """

        start_year = self.start_year  # Assuming the scenario starts in 2023
        years_to_end = self.end_year - start_year

        if self.steady_state_high_income_assumption == "on_with_growth" or self.steady_state_high_income_assumption == "on":
               
               for country in self.countries.values():
                        #set positive dummy value just so that hysteresis is not triggered in the scenarios where it matters
                        country.cagr_average = 0.01
                
        elif self.steady_state_high_income_assumption == "off":
              
                for country in self.countries.values():
                        # Compute the CAGR for the average income
                        average_income = country.hh_mean *365 # convert to annual household disposable income

                        # Compute CAGR
                        if average_income > 0 and years_to_end > 0:
                                cagr = (self.income_goal / average_income) ** (1 / years_to_end) - 1
                                #print("this is cagr", cagr)
                        else:
                                cagr = 0  # Assigning 0 if the average income is 0 or years to end is not positive
                
                        # Store the CAGR value for the country
                        country.cagr_average = cagr

    def compute_average_global_hh_income(self):
            
            """
            Description: 
                    Compute the average global household income.
            Parameters:
                    None
            """

            global_hh_income_total = 0
            global_population = 0
            for country in self.countries.values():
                global_hh_income_total += country.hh_mean * country.population * 365 # convert to annual household disposable income
                global_population += country.population
      
            return global_hh_income_total / global_population

    def compute_average_global_growth_rate(self):
                  
           """
           Description: 
                 Compute the average global growth rate.
           Parameters:
                 None
           """
        
           start_year = self.start_year  # Assuming the scenario starts in 2023
           years_to_end = self.end_year - start_year
           average_global_hh_income = self.compute_average_global_hh_income()
           # Compute CAGR
           if average_global_hh_income > 0 and years_to_end > 0:
                cagr = (self.income_goal / average_global_hh_income) ** (1 / years_to_end) - 1           
           else:
                # throw error message
                raise ValueError("Average global household income is 0 or years to end is not positive")        
           return cagr

    def compute_starting_global_emissions(self):
            
            """
            Description: 
                    Compute the starting global emissions which is helpful for computing carbon budget pathways.
            Parameters:
                    None
            """
    
            # Compute the starting global emissions
            global_emissions = 0
            for country in self.countries.values():
                global_emissions += country.emissions_trajectory[2022] # only interested in the emissions at the start 
            return global_emissions/1e9 # converted to gigatons from tons
    

    def compute_ending_global_emissions(self):
                
                """
                Description: 
                        Compute the ending global emissions which is helpful to see where emissions are but should be at that point.
                Parameters:
                        None
                """

                # Compute the ending global emissions
                global_emissions = 0
                # compute emissions at the end of the scenario over all countries
                for country in self.countries.values():
                        global_emissions += country.emissions_trajectory[self.end_year] # only interested in the emissions at the end   
                return global_emissions/1e9 # converted to gigatons from tons

    def compute_linear_carbon_budget_pathway(self):
        
        """
        Description: 
                Compute the linear carbon budget pathway.
        Parameters:
                None
        """

        start_emissions = self.compute_starting_global_emissions()

        # Compute the time to deplete the carbon budget at a constant linear rate
        time_to_deplete = (2 * self.carbon_budget) / start_emissions # using triangle formula to calculate the time to deplete the carbon budget
        # print("this is time to deplete", time_to_deplete)
        # Compute the slope (rate of change) of the line
        slope = -start_emissions / time_to_deplete

        # Generate the years and emissions for plotting
        years = np.linspace(0, time_to_deplete, num=int(time_to_deplete)+1)
        emissions = slope * years + start_emissions

        return years, emissions
    
    def compute_exponential_carbon_budget_pathway(self):
        
        """
        Description: 
                Compute the exponential carbon budget pathway that respects the given budget constraint.
                Aiming for emissions to approach 1% of the initial level, within the carbon budget.
                
        Parameters:
                None
        """

        start_emissions = self.compute_starting_global_emissions()
        # Initial guess for the number of years to reach 1% of the initial emissions
        years_to_1_percent_initial_guess = 100  # Arbitrary initial guess, but should be rather high
        target_emissions = 0.01 * start_emissions

        # Function to calculate total emissions given a reduction rate
        def calculate_total_emissions(reduction_rate, years):
            emissions = start_emissions * np.exp(-reduction_rate * years)
            total_emissions = np.trapz(emissions, years)
            return total_emissions

        # Find the correct reduction rate that ensures total emissions <= carbon_budget
        # Initial guess for the reduction rate
        reduction_rate = np.log(start_emissions / target_emissions) / years_to_1_percent_initial_guess
        
        # Define a high-resolution time span for accurate integration
        years = np.linspace(0, years_to_1_percent_initial_guess, 1000)
        total_emissions = calculate_total_emissions(reduction_rate, years)

        # Adjust reduction rate to fit within the carbon budget
        while total_emissions > self.carbon_budget:
            # Decrease the reduction rate to reduce total emissions
            reduction_rate *= 0.95  # Adjust the factor as needed for finer control
            total_emissions = calculate_total_emissions(reduction_rate, years)
        
        # Recalculate emissions with the final reduction rate
        emissions = start_emissions * np.exp(-reduction_rate * years)

        return years, emissions


    def compute_country_scenario_params(self):

        """
        Description: 
                Compute country parameters based on scenario parameters.

                that is 
                 - growth rates for each decile
        Parameters:
                None
        """

        self.compute_group_growth_rates() # compute the growth rates for each decile
        self.compute_average_growth_rates() # compute the average growth rates for each country


    def sum_cumulative_emissions(self):
        
        """
        Description: 
                Sum the cumulative emissions for all countries over time in a given scenario
        Parameters:
                None
        """

        cumulative_emissions = 0
        for country in self.countries.values():
                # sum the emissions trajectory for each country and add to cumulative emissions
                cumulative_emissions += sum(country.emissions_trajectory.values())

        # 
        print("this is the cumulative emissions  before cdr adjustment", cumulative_emissions)
        

        # if cdr is on, subtract the cdr trajectory from the cumulative emissions
        if self.cdr_assumption == "on":
                for year, cdr_value in self.cdr_global_level_trajectory.items():
                        cumulative_emissions -= cdr_value * 1e9 # convert to kgs from gigatons

        print("this is the cumulative emissions  after cdr adjustment", cumulative_emissions)
        return cumulative_emissions
    
    
    def compute_current_global_population(self):
        
        """
        Description: 
                Compute the current global population in the current year. So this needs to be recalculated every year.
        Parameters:
                None
        """
        # sum the population in a loop over all countries
        global_population = 0
        for country in self.countries.values():
                global_population += country.population
        return global_population
    
    def create_global_population_and_income_vectors(self):
                
                """
                Description: 
                        Create a vector of population and income for all countries across *all deciles*.
                        Necessary for computing global gini coefficient.
                Parameters:
                        None
                """
                
                # create empty lists to store the population and income
                population = []
                income = []
                for country in self.countries.values():
                        for country_decile in range(1, 11):
                                population.append(country.population / 10) # divide by 10 to get the population of each decile
                                income.append(getattr(country, f'decile{country_decile}_abs')) # get the income of each decile

                # sort the income and population vectors according to the income vector
                income, population = zip(*sorted(zip(income, population)))

                # store the population and income vectors in the self.gini_data dictionary
                # store the year by extracting it from a country instance

                self.gini_data["years"].append(self.countries["Germany"].year)
                self.gini_data["population"].append(population)
                self.gini_data["income"].append(income)

    def compute_gini_coefficient_change_rate(self):

            
        """
        Description: 
                Compute global gini coefficient change rate over time.
        Parameters:
                None
        """

        gini_data = pd.DataFrame(self.gini_data)

         # read out the columns population and income from gini data and plot each values as lorenz curve
        #subset for every year in gini_data years column the value in population and income column
        # loop ovre the years

        list_of_gini_coefficients = []

        for year in gini_data["years"]:
                # Get the population data for the year
                population_data = list(gini_data.loc[gini_data["years"] == year, "population"].iloc[0])
                # Get the income data for the year
                income_data = list(gini_data.loc[gini_data["years"] == year, "income"].iloc[0])

                # compute the total income_data as well by multiplying the income_data with the population_data
                total_income = [population_data[i]*income_data[i] for i in range(len(population_data))]

                # compute the cumulative total income share and the cumulative population share
                cumulative_income_share = [sum(total_income[:i])/sum(total_income) for i in range(len(total_income)+1)]

                cumulative_population_share = [sum(population_data[:i])/sum(population_data) for i in range(len(population_data)+1)]

                # Compute the Gini coefficient
                # The area under the Lorenz curve can be computed as the sum of the areas of the trapezoids under it
                area_under_lorenz_curve = sum((cumulative_population_share[i+1] - cumulative_population_share[i]) * 
                                                (cumulative_income_share[i+1] + cumulative_income_share[i]) / 2 
                                                for i in range(len(cumulative_population_share)-1))
                gini_coefficient = 1 - 2 * area_under_lorenz_curve

                list_of_gini_coefficients.append(gini_coefficient)
        

        ## use a placeholder approximately 0, so 0.1 here, to avoid computation problems of average necessary cagr
        gini_coefficient_change_rate = (0.01 / list_of_gini_coefficients[0]) ** (1 / (len(list_of_gini_coefficients) - 1)) - 1

        return gini_coefficient_change_rate
    

    def store_national_gini_coefficients(self):
         
        """
        Description: 
                Compute the national gini coefficients for all countries and store in a dataframe with a scenario key etc.
        Parameters:
                None
        """
        scenario_gini_data = pd.DataFrame(columns=['scenario', 'country', 'year', 'gini'])

        ### NECESSARY SUB PROCEDURE TO ADD A LIST TO A DATAFRAME
        def add_list_to_dataframe(df, elements):
                # Validate the input
                #print(len(elements))
                #print(len(df.columns))
                if len(elements) != len(df.columns):
                        raise ValueError("The number of elements must be exactly equal to the number of columns.")               
                # Create a DataFrame from the list and append it to the existing DataFrame
                # If the dataframe is empty, use DataFrame constructor
                if df.empty:
                        df = pd.DataFrame([elements], columns=df.columns)
                else:
                # Create a DataFrame from the list and append it to the existing DataFrame
                        new_df = pd.DataFrame([elements], columns=df.columns)
                        df = pd.concat([df, new_df], ignore_index=True)      
                return df
                        

        # loop over the countries and add each country with the concat() method to the dataframe
        i = 0
        scenario_id = str(self.income_goal) + '_' + str(self.end_year)
        for country in self.countries.values():
                for years, value in country.gini_coefficient_trajectory.items():
                        countryvalues2 = [] # create empty list for current country values
                        countryvalues2.append(scenario_id) # add scenario_id to the list
                        countryvalues2.append(country.code) # add country code to the list
                        countryvalues2.append(years)
                        countryvalues2.append(value)   
                        scenario_gini_data = add_list_to_dataframe(scenario_gini_data, countryvalues2)
       
        return scenario_gini_data

    def step(self):
            
        """
        Description: 
                Compute one scenario step
        Parameters:
                None
        """
        


        # loop over all countries for country specific steps
        for country in self.countries.values():
                country.technological_change()
                # Differentiate between scenarios that run regardless of end year to 2100 and those that stop at the end year
                # if self.run_until_2100 == "on":
                        # check if the year is less than 2100
                       #  if country.year < self.end_year:
                                # run the scenario until 2100
                               #  country.economic_growth()
                #  else:
                        # country.economic_growth()

                country.economic_growth()
                country.population_growth()
                country.update_emissions()
                country.calculate_current_carbon_budget()
                country.calculate_national_gini_coefficient()
                country.year += 1 # increase the year by one
                country.save_current_state() # save the current state of the country

        # global level steps such as gini coefficient calculation
        
        self.create_global_population_and_income_vectors()

    def store_national_gdp_trajectories(self):
                
                """
                Description: 
                        Store the national GDP trajectories for all countries in a dataframe.
                Parameters:
                        None
                """
                scenario_gdp_data = pd.DataFrame(columns=['scenario', 'country', 'year', 'gdp', 'gdppc'])
        
                ### NECESSARY SUB PROCEDURE TO ADD A LIST TO A DATAFRAME
                # which takes the four elements of the dataframe and adds them to the dataframe
                def add_list_to_dataframe(df, elements):
                        # Validate the input
                        if len(elements) != len(df.columns):
                                raise ValueError("The number of elements must be exactly equal to the number of columns.")               
                        # Create a DataFrame from the list and append it to the existing DataFrame
                        # If the dataframe is empty, use DataFrame constructor
                        if df.empty:
                                df = pd.DataFrame([elements], columns=df.columns)
                        else:
                        # Create a DataFrame from the list and append it to the existing DataFrame
                                new_df = pd.DataFrame([elements], columns=df.columns)
                                df = pd.concat([df, new_df], ignore_index=True)      
                        return df
                
                # loop over the countries and add each country with the concat() method to the dataframe
                scenario_id = str(self.income_goal) + '_' + str(self.end_year)
                for country in self.countries.values():
                        for years, value in country.gdp_trajectory.items():
                                countryvalues2 = []
                                value2 = country.gdppc_trajectory[years]
                                countryvalues2.append(scenario_id)
                                countryvalues2.append(country.code)
                                countryvalues2.append(years)
                                countryvalues2.append(value)
                                countryvalues2.append(value2)
                                scenario_gdp_data = add_list_to_dataframe(scenario_gdp_data, countryvalues2)

                return scenario_gdp_data
    
    #def compute_global_cdr_trajectory(self):
        #"""
        #Compute the global carbon dioxide removal (CDR) trajectory using a convex exponential model.
        
        #The trajectory is given by:
         #   CDR(t) = cdr_level_2100 * [exp(k * (t - start_year)) - 1] / [exp(k * (2100 - start_year)) - 1]
        
        #where:
         #   - t is any year between start_year and 2100.
         #   - k is a growth rate constant (set here arbitrarily to 0.03 for example purposes).
        
        #Returns:
         #   dict: A dictionary with years as keys (int) and the corresponding CDR values as floats.
        #"""
        # Growth rate constant; adjust as needed for a sharper or milder exponential look.
        #k = 0.03
        
        #final_year = 2100
        
        # Create an array of years from start_year to 2100.
        #years = np.arange(self.start_year, final_year + 1)
        
        # Compute the convex exponential CDR trajectory.
        #cdr = self.cdr_level_2100 * (np.exp(k * (years - self.start_year)) - 1) / (np.exp(k * (final_year - self.start_year)) - 1)
        
        # Build a dictionary where the key is the year (as an integer) and the value is the CDR (as a float)
        #trajectory = {int(year): float(cdr_value) for year, cdr_value in zip(years, cdr)}

        #print("this is the trajectory", trajectory)
        
        #return trajectory
    


    def compute_global_cdr_trajectory(self, k=0.2, t0=None):
                """
                Compute the global carbon dioxide removal (CDR) trajectory using a logistic S-curve
                that reaches full capacity by 2060 and then remains constant.

                CDR(t) = L * [σ(t) - σ(start)] / [σ(2060) - σ(start)],  for start_year ≤ t ≤ 2060
                        = L,                                        for t > 2060

                where σ(t) = 1 / (1 + exp(-k*(t - t0))). 
                By default, t0 is midpoint of start_year and 2060.

                Returns:
                        dict[int,float]: years → CDR value
                """
                import numpy as np

                start = self.start_year
                peak  = 2060
                final = 2100
                L     = self.cdr_level_2100

                # default inflection point to midpoint between start and peak
                if t0 is None:
                        t0 = start + 0.5 * (peak - start)

                years = np.arange(start, final + 1)
                trajectory = {}

                # compute raw logistic
                raw = 1 / (1 + np.exp(-k * (years - t0)))

                # values at start and at peak for normalization
                sigma_start = 1 / (1 + np.exp(-k * (start - t0)))
                sigma_peak  = 1 / (1 + np.exp(-k * (peak  - t0)))

                for t, s in zip(years, raw):
                        if t <= start:
                                cdr_t = 0.0
                        elif t <= peak:
                                # normalize s to [0,1] over [start,peak] then scale to [0,L]
                                cdr_t = L * (s - sigma_start) / (sigma_peak - sigma_start)
                        else:
                                cdr_t = L

                        trajectory[int(t)] = float(cdr_t)

                return trajectory
    


    
    def get_decile_income_distribution_data(self):
        """
        Gather every country’s decile incomes (year 2100) and their population weights.

        Returns:
            incomes (np.ndarray): all decile incomes across all countries.
            weights (np.ndarray): corresponding population in each decile (country.population/10).
        """
        incomes = []
        weights = []

        if self.run_until_2100 != "on":
            print("Warning: run_until_2100 != 'on'; results may not be for year 2100.")

        for country in self.countries.values():
            # each decile is 1/10 of the national population
            decile_pop = country.population / 10.0
            for decile_num in range(1, 11):
                incomes.append(getattr(country, f"decile{decile_num}_abs"))
                weights.append(decile_pop)

        return np.array(incomes), np.array(weights)
                
          

    def run(self):

        """
        Description: 
                Run the scenario
        Parameters:
                None
        """

        # set up necessary parameters for the scenario
        self.compute_country_scenario_params()
        
        if self.run_until_2100 == "on":
                # run the scenario over time
                for year in range(self.start_year, 2100): 
                        self.step()

        elif self.run_until_2100 == "off":
                # run the scenario over time
                for year in range(self.start_year, self.end_year): # the scenario must run the change from 2022 to 2023 and as last step the change from endyear - 1 to endyear, it cannot run through endyear again
                        self.step()


  