import numpy as np

class Country():

        """
        Description: 
                A class representing one country and its
                defining parameters.
        
        Parameters:
                id                          - unique identifier for the country (mapped from 'index')
                region                      - name of the region where the country is located (mapped from 'region_name')
                region_code                 - code representing the region of the country (mapped from 'region_code')
                code                        - country code (mapped from 'country_code')
                hh_mean                     - average household cons. exp./income mean (mapped from 'mean')
                gdp_pc                      - GDP per capita, PPP (constant 2017 international $) (mapped from 'gdp_pc_ppp_2017')
                gini_hh                     - GINI coefficient of household income (mapped from 'gini')
                carbon_intensity            - carbon intensity per $ of income in 2022 (mapped from 'carbon_intensity')
                carbon_intensity_trend      - trend in carbon intensity from 2010 to 2020 (mapped from 'carbon_intensity_trend')
                decile1_abs                 - absolute value for the first decile (mapped from 'decile1_abs')
                decile2_abs                 - absolute value for the second decile (mapped from 'decile2_abs')
                decile3_abs                 - absolute value for the third decile (mapped from 'decile3_abs')
                decile4_abs                 - absolute value for the fourth decile (mapped from 'decile4_abs')
                decile5_abs                 - absolute value for the fifth decile (mapped from 'decile5_abs')
                decile6_abs                 - absolute value for the sixth decile (mapped from 'decile6_abs')
                decile7_abs                 - absolute value for the seventh decile (mapped from 'decile7_abs')
                decile8_abs                 - absolute value for the eighth decile (mapped from 'decile8_abs')
                decile9_abs                 - absolute value for the ninth decile (mapped from 'decile9_abs')
                decile10_abs                - absolute value for the tenth decile (mapped from 'decile10_abs')
                gdp_hh_income_ratio         - ratio of GDP to mean household income (mapped from 'gdp_to_mean_hh_income_ratio')
                total_emissions             - emissions of the country (mapped from 'emissions') per year total

        """

        def __init__(self, scenario, **kwargs):

                """
                Parameters:
                        The parameters here are given as attributes and are specified in the class doc.
                """
                self.scenario = scenario
                self.year = 2022  # All countries are initialized with 2022 data
                self.cagr_by_decile = {}  # Necessary for convergence growth rates in scenario method
                self.cagr_average = None # Necessary for AVERAGE country convergence growth rate in scenario method (so for the country as a whole it is not literally the average of decile growth rates)
                self.pop_growth_rate = None # Necessary for population growth rate in case the assumption needs to dynamically change this for instane in the semi_log_model case of population growth
                self.carbon_budget_per_current_year = None # Necessary for the carbon budget of the country in the current year

                # Initialize dictionaries for the country's trajectories which is necessary data to be collected for plotting.
                self.income_hh_trajectory = {}  # Necessary for plotting the trajectory of the countrys
                self.gdppc_trajectory = {}  # Necessary for plotting the trajectory of the countrys
                self.decile_trajectories = {}  # Necessary for plotting the trajectory of the countrys deciles here each dictionary entry is another dictionary with the years as keys and the decile incomes as values
                self.population_trajectory = {}  # Necessary for plotting the trajectory of the countrys population
                self.carbon_intensity_trajectory = {}  # This is the (future) historical trajectory of the carbon intensity of the country
                self.emissions_trajectory = {}  # This is the (future) historical trajectory of the emissions of the country
                self.carbon_emissions_pc_trajectory = {}  # This is the (future) historical trajectory of the carbon emissions per capita of the country        
                self.gini_coefficient_trajectory = {}  # This is the (future) historical trajectory of the gini coefficient of the country

                # Dictionary mapping kwargs names to class attribute names
                attribute_mapping = {
                        'index': 'id',
                        'region_name': 'region',
                        'region_code': 'region_code',
                        'country_code': 'code',
                        'mean': 'hh_mean', # this per year and per capita
                        'gdp_pc_ppp_2017': 'gdp_pc', # this per year and per capita
                        'gini': 'gini_hh',
                        'carbon_intensity': 'carbon_intensity', # this is the intensity of carbon per $ of income in 2022 (2021 and 2022 already modelled on trend)
                        'carbon_intensity_trend': 'carbon_intensity_trend', # This is the trend in carbon intensity 2010 - 2020
                        'decile1_abs': 'decile1_abs',
                        'decile2_abs': 'decile2_abs',
                        'decile3_abs': 'decile3_abs',
                        'decile4_abs': 'decile4_abs',
                        'decile5_abs': 'decile5_abs',
                        'decile6_abs': 'decile6_abs',
                        'decile7_abs': 'decile7_abs',
                        'decile8_abs': 'decile8_abs',
                        'decile9_abs': 'decile9_abs',
                        'decile10_abs': 'decile10_abs',
                        'gdp_to_mean_hh_income_ratio': 'gdp_hh_income_ratio',
                        'population': 'population',
                        'total_emissions': 'total_emissions',
                        'growth_trend_2012_to_2022': 'gdp_pc_historical_growth'
                        }

                # Set attributes based on attribute mapping above
                for kwarg_attr, class_attr in attribute_mapping.items():
                        if kwarg_attr in kwargs:
                                setattr(self, class_attr, kwargs[kwarg_attr])
                        else:
                                print(f"Warning: '{kwarg_attr}' not found in kwargs. Attribute '{class_attr}' not set.")

                # Set other attributes dynamically with 'country_' prefix
                for key, value in kwargs.items():
                        if key not in attribute_mapping:
                                setattr(self, f'country_{key}', value)


                # set 2022 values of trajectory dictionaries with the initial values
                self.income_hh_trajectory[self.year] = self.hh_mean*365 # this is the mean household cons.income
                self.gdppc_trajectory[self.year] = self.gdp_pc # this is the mean gross domestic product per capita
                self.population_trajectory[self.year] = self.population # this is the population
                self.carbon_intensity_trajectory[self.year] = self.carbon_intensity # this is the carbon intensity of the country
                self.emissions_trajectory[self.year] = self.carbon_intensity * self.gdp_pc * self.population / 1000 # this is the emissions of the country, divide by 1000 to get to metric tons from kg
                self.carbon_emissions_pc_trajectory[self.year] = self.carbon_intensity/1000 * self.gdp_pc # this is the carbon emissions per capita of the country in tonnes


                # set more variables necessary to compute the country carbon budget consistent behaviour
                self.diff_budget_and_emissions = None
                self.diff_budget_and_emissions_ratio = None

        def save_current_state(self):

                """
                Description: 
                        A method saving the current state of the country across all kinds of variables. This is necessary for plotting the trajectory of the country's income and gdp per capita.
                
                Parameters:
                        None

                """
                #### ECONOMIC VARIABLES ####
                # add current year and current income to the income trajectory
                # if the year is 2022 then the income is the mean household income times 365 to get the annual income otherwise it is annual already
                if self.year == 2022:
                        self.income_hh_trajectory[self.year] = self.hh_mean*365 # this is the mean household cons.income
                else: # otherwise it is annual already
                        self.income_hh_trajectory[self.year] = self.hh_mean
                
                self.gdppc_trajectory[self.year] = self.gdp_pc # this is the mean gross domestic product per capita
                self.population_trajectory[self.year] = self.population # this is the population
                self.carbon_intensity_trajectory[self.year] = self.carbon_intensity # this is the carbon intensity of the country
                self.emissions_trajectory[self.year] = self.carbon_intensity * self.gdp_pc * self.population / 1000 # this is the emissions of the country, divide by 1000 to get to metric tons from kg
                self.carbon_emissions_pc_trajectory[self.year] = self.carbon_intensity/1000 * self.gdp_pc # this is the carbon emissions per capita of the country in tonnes
                self.gini_coefficient_trajectory[self.year] = self.gini_hh # this is the gini coefficient of the country
                # add and save current decile incomes to the decile trajectories where every decile in the dictionary is another dictionary with the years as keys and the decile incomes as values
                for decile_num in range(1, 11):
                        decile_income = getattr(self, f'decile{decile_num}_abs')
                        if f'decile{decile_num}' not in self.decile_trajectories:
                                self.decile_trajectories[f'decile{decile_num}'] = {}
                        self.decile_trajectories[f'decile{decile_num}'][self.year] = decile_income

        def technological_change(self):

                """
                Description: 
                        A method computing the technological change of the country expressed as a change in carbon intensity.

                Parameters:
                        None

                """
                # Define subprocedures for the sigmoidal function and the weighted average to compute a model of technological change

                def sigmoid(t, k=0.1, t0=50):
                        """
                        Sigmoid function for calculating the weight w(t).
                        
                        Parameters:
                        - t: The time variable.
                        - k: Steepness of the curve.
                        - t0: Midpoint of the sigmoid, where w(t) = 0.5.
                        """
                        return 1 / (1 + np.exp(-k * (t - t0)))

                def weighted_average(t, y, z, k=0.1, t0=50):
                        """
                        Calculates the weighted average of y and z over time using a sigmoidal function for weights.
                        
                        Parameters:
                        - t: Time variable, can be a scalar or a numpy array.
                        - y: The y variable.
                        - z: The z variable.
                        - k, t0: Parameters for the sigmoid function.
                        """
                        w = sigmoid(t, k, t0)
                        return (1 - w) * y + w * z

                # DIFFERENTIATE TECHNOLOGICAL CHANGE ASSUMPTIONS
                #################################################
                #if self.scenario.tech_evolution_assumption == "plausible":
                # for the first ten years assume the ongoing trend in carbon intensity from 2010 to 2020
                if self.year < 2025:
                        self.carbon_intensity = self.carbon_intensity * (1 + self.carbon_intensity_trend)
                # after that assume a constant the logarithmic model empirically determined via cross country data gdppc 2022 vs trend 2010 2020
                # which is this equation y = -0.015ln(x) + 0.1309 where x is the gdp per capita in 2022 and y is the trend in carbon intensity from 2010 to 2020
                else:   
                        # distinguish between the two cases of country that grows or degrows its average gdp per capita and hence introduce hysteresis in the technological change
                        if self.scenario.tech_hysteresis_assumption == "on":

                                if self.cagr_average > 0:

                                        modelled_trend = -0.015 * np.log(self.gdp_pc) + 0.1309
                                        z = self.scenario.final_improvement_rate ## this is a constant  uniform value for the carbon intensity decline that the world adopts slowly and then rapidly.
                                        k = self.scenario.k # this is the steepness of the sigmoidal function
                                        t0 = self.scenario.t0 # this is the midpoint of the sigmoidal function
                                        weighted_model = weighted_average(self.year, modelled_trend, z,  k=k, t0=t0)
                                        self.carbon_intensity = self.carbon_intensity * (1 + weighted_model)
                                else:
                                        # if the country is in a planned degrowth scenario then assume the fixed progress in technology i.e. -1% per year already from the start
                                        #assume progress in technology under planned degrowth i.e. -1% per year
                                        self.carbon_intensity = self.carbon_intensity * (1 + self.scenario.final_improvement_rate) # CAREFUL RATE GIVEN IS NEGATIVE so + operator leads to multiplier < 1

                        elif self.scenario.tech_hysteresis_assumption == "off":

                                modelled_trend = -0.015 * np.log(self.gdp_pc) + 0.1309
                                z = self.scenario.final_improvement_rate ## this is a constant  uniform value for the carbon intensity decline that the world adopts slowly and then rapidly.
                                k = self.scenario.k # this is the steepness of the sigmoidal function
                                t0 = self.scenario.t0 # this is the midpoint of the sigmoidal function
                                weighted_model = weighted_average(self.year, modelled_trend, z,  k=k, t0=t0)
                                self.carbon_intensity = self.carbon_intensity * (1 + weighted_model)
                        
                #elif self.scenario.tech_evolution_assumption == "necessary":
                        # generally here we will calculate the necessary carbon intensity reduction rate to stay within the country specific allocated carbon budget
                        # store the variable self.diff_budget_and_emissions_percentage in a local variable
                        #self.calculate_diff_budget_and_emissions() ## execute the method to calculate the difference between the carbon budget and the emissions of the country in the current year
                        ###################################################################################
                        ######## USE A simple IPAT framework to calculate the new carbon intensity ########
                        ###################################################################################
                        # the IPAT framework is I = P * A * T where I is the impact, P is the population, A is the affluence and T is the technology
                        # we want to calculate the new technology T to stay within the carbon budget
                        # we assume the population stays constant and the affluence stays constant so we only need to calculate the new technology
                        # we just must rearrange the formula to T = I_new / (P * A) and use the new I which is the carbon budget adjusted for the self.diff_budget_and_emissions_percentage

                        # check though first whether the country is already within its carbon budget, a ratio of 1 or less means it is within the budget
                        #if self.diff_budget_and_emissions_ratio < 1:
                         #        modelled_trend = -0.015 * np.log(self.gdp_pc) + 0.1309
                          #       self.carbon_intensity = self.carbon_intensity * (1 + modelled_trend)
                       #else:
                                # if the country is not within its carbon budget then we calculate the new carbon intensity
                                # one must take the inverse of the ratio to get the necessary reduction in emissions to stay within the budget i.e. 1/self.diff_budget_and_emissions_ratio
                        #        if self.code == "USA":
                          #              print("this is the inverse ratio", 1/self.diff_budget_and_emissions_ratio)
                         #               print("this is the carbon intensity calculated from the IPAT framework of",self.code," ", (self.total_emissions * (1/self.diff_budget_and_emissions_ratio)) / (self.gdp_pc * self.population)*1000)
                           #     self.carbon_intensity = (self.total_emissions * (1/self.diff_budget_and_emissions_ratio)) / (self.gdp_pc * self.population) * 1000 # this is the emissions of the country, multiplied by 1000 to get to kg co2 per $ from metric tons co2 per $
                        

                
        def update_emissions(self):

                """
                Description: 
                        A method computing the emissions of the country. 

                Parameters:
                        None

                """
                self.total_emissions = self.carbon_intensity * self.gdp_pc * self.population / 1000 # this is the emissions of the country, divide by 1000 to get to metric tons from kg        


        def economic_growth(self):

                """
                Description: 
                        A method computing the economic growth of the country.        
                
                Parameters:
                        None

                """
                # compute new state
                # loop over all deciles and apply the growth rate
                for decile_num in range(1, 11):
                        decile_income = getattr(self, f'decile{decile_num}_abs')
                        # use self.cagr_by_decile to get the growth rate for the decile
                        cagr = self.cagr_by_decile[f'decile{decile_num}']
                        # calculate the new income
                        # distinguish between steady state and non steady state assumption
                        if self.scenario.steady_state_high_income_assumption == "on":

                                if decile_income >= self.scenario.income_goal:
                                        new_income = decile_income
                                else:
                                        new_income = decile_income * (1 + cagr)

                        # here the deciles that are already above the income goal are assumed to grow at the historical growth rate of the country        
                        elif self.scenario.steady_state_high_income_assumption == "on_with_growth":

                                if decile_income >= self.scenario.income_goal:
                                        # if the income is already above the goal then apply the historical growth rate
                                        new_income = decile_income * (1 + self.gdp_pc_historical_growth)
                                else:
                                        new_income = decile_income * (1 + cagr)

                        elif self.scenario.steady_state_high_income_assumption == "off":
                                new_income = decile_income * (1 + cagr)

                     
                        # set the new income
                        setattr(self, f'decile{decile_num}_abs', new_income)


                # COMPUTE NEW AGGREGATE QUANTITIES        
                # compute NEW mean country household cons. exp.income as average of decile incomes
                self.hh_mean = sum([getattr(self, f'decile{decile_num}_abs') for decile_num in range(1, 11)]) / 10
                
                #### compute NEW gdp per capita ####
                #### DIFFERENTIATE GDP SCENARIOS/ASSUMPTIONS ####
                #################################################
                if self.scenario.gdp_assumption == "constant_ratio":
                        ## just apply the empirically found ratio of gdp to mean household income
                        self.gdp_pc =  self.hh_mean / self.gdp_hh_income_ratio                   
                  
                elif self. scenario.gdp_assumption == "model_ratio":
                        # the gdp ratio is conditional on the mean household income see script first data explorations
                        if self.hh_mean < 5000: # use piecewise linear fits from first data explorations for this, so yearly cons. exp/disposable income vs. gdp to mean household income ratio
                                self.gdp_hh_income_ratio = -0.0000571 * (self.hh_mean) + 0.67
                                #print("this is gdp_hh_income_ratio", self.gdp_hh_income_ratio)
                                self.gdp_pc =  self.hh_mean / self.gdp_hh_income_ratio 
                        else:
                                self.gdp_hh_income_ratio = 0.000002 * (self.hh_mean) + 0.39
                                self.gdp_pc = self.hh_mean  / self.gdp_hh_income_ratio # this is the ratio of gdp to mean household income for countries with mean household income > 10000 which seems to be a reasonable assumption according to the cross sectional country data 


        def population_growth(self):
                
                """
                Description: 
                        A method computing the population growth of the country. 
                        This method applies a rather complicated control flow for different assumptions about population growth
                
                Parameters:
                        None

                """
                # DIFFERENTIATE POPULATION GROWTH MAIN ASSUMPTIONS
                if self.scenario.pop_growth_assumption == "UN_medium":
                        # based on the assigned scenario instance which carries the scenario.population_growth_rates dataframe with row keys as country codes make the population grow
                        # get the growth rate for the country for the correct year which is the current year
                        # Filter the DataFrame for the row matching both the country code and the correct year.
                        # Assuming 'year' is also a column in the DataFrame, and it stores years as integers or strings that match self.year + 1.
                        self.pop_growth_rate = self.scenario.population_growth_rates.loc[str(self.code)][str(self.year)]
                        if self.pop_growth_rate is not None:
                                new_population = self.population * (1 + self.pop_growth_rate)
                                self.population = new_population
                        else:
                                print("No growth rate found for", self.code, "in year", self.year)


                elif self.scenario.pop_growth_assumption == "semi_log_model":
                        # in this case we start from the empirical population growth rate in 2022 and then apply the semi log model equation for population growth change rate
                        # for the future years apply the semi log model equation for population growth change rate y = 0.09 - 0.01*log(x) 
                        # where x is gdp per capita and y is the population growth rate

                        # however also here we make an hysteresis assumption, meaning we only apply this rule if the gdp per capita increases, if it decreases we do not change population growth rate
                        # because it is not clear that planned degrowth economies revert socio-cultural norms to higher fertility rates

                        if self.scenario.population_hysteresis_assumption == "on":

                                if self.cagr_average > 0:
                                        self.pop_growth_rate = 0.0874 - 0.0190*np.log10(self.gdp_pc)
                                        new_population = self.population * (1 + self.pop_growth_rate)
                                        self.population = new_population
                                else:
                                        # if degrowing from 2023 onwards keep population growth as it has been in 2022, preservation of socio-cultural norms
                                        self.pop_growth_rate = 0.0874 - 0.0190*np.log10(self.gdppc_trajectory[2022])
                                        new_population = self.population * (1 + self.pop_growth_rate)
                                        self.population = new_population

                        elif self.scenario.population_hysteresis_assumption == "off":

                                self.pop_growth_rate = 0.0874 - 0.0190*np.log10(self.gdp_pc)
                                new_population = self.population * (1 + self.pop_growth_rate)
                                self.population = new_population

                        
                elif self.scenario.pop_growth_assumption == "semi_log_model_elasticity":
                        # in this case we start from the empirical population growth rate in 2022 and then apply the semi log model equation for population growth change rate
                        if self.year == 2022:
                                self.pop_growth_rate = self.scenario.population_growth_rates.loc[str(self.code)][str(self.year)]
                                if self.pop_growth_rate is not None:
                                        new_population = self.population * (1 + self.pop_growth_rate)
                                        self.population = new_population
                        else:
                                # for the future years apply the semi log model equation for population growth change rate y = 0.09 - 0.01*log(x) 
                                # where x is gdp per capita and y is the population growth rate, so we must first calculate the derivative of y with respect to x
                                # which is dy/dx = -0.01/x
                                # then we calculate the elasticity of population growth with respect to gdp per capita which is the derivative of y with respect to x times x/y
                                # which is (dy/y)/(dx/x) = dy/dx *x/y = -0.01/x * x/y = -0.01/y

                                # then we also assume hysteresis in this assumption, meaning we only apply this elasticity if the gdp per capita increases, if it decreases we do not change population growth rate
                                # this means in our convergence scenario, for countries who apply deliberate degrowth, they do not actually get poorer in terms of living standards but only in gdp.
                                # so there is no reason to assume their population growth rate would change upward in this case.
                                if self.cagr_average > 0:
                                        # then we apply the elasticity as percentage change in population growth rate
                                        elasticity = -0.01/self.pop_growth_rate
                                        self.pop_growth_rate =  self.pop_growth_rate * (1 + elasticity) # that is the higher the gdp per capita the lower the population growth rate but
                                else:
                                        # if the gdp per capita decreases we still apply the UN medium projections
                                        self.pop_growth_rate = self.scenario.population_growth_rates.loc[str(self.code)][str(self.year)]

                                new_population = self.population * (1 + self.pop_growth_rate)
                                self.population = new_population
                                
        def calculate_current_carbon_budget(self):

                """
                Description: 
                        A method computing the carbon budget of the country in the current year. 

                Parameters:
                        None

                """
                # this is the carbon budget of the country in the current year
                years, emissions = self.scenario.linear_yearly_carbon_budget
                # round down the years to their nearest integer and add 2022 to every one of them
                years = [int(np.floor(year)) + 2022 for year in years]
                # transform to dictionary for better handling
                z = dict(zip(years, emissions))
                # print z
                #print("this is z ", z)
                # check first whether current year is in the dictionary
                if self.year in z:
                        global_current_budget = z[self.year]
                        fair_share_budget = global_current_budget * (self.population / self.scenario.total_population)
                else:
                        fair_share_budget = 0
                self.carbon_budget_per_current_year = fair_share_budget
                #print("this is the carbon budget per current year of ", self.code, " ", self.carbon_budget_per_current_year)


        def calculate_diff_budget_and_emissions(self):
                        
                        """
                        Description: 
                                A method computing the difference between the carbon budget and the emissions of the country in the current year. And gives
                                out the necessary percentage reduction in emissions to reach the carbon budget for the given year. 
        
                        Parameters:
                                None
        
                        """
                        if self.code == "USA": 
                                print("this is the total emissions of ", self.code, " ", self.total_emissions)
                        self.calculate_current_carbon_budget()
                        if self.code == "USA": 
                                print("this is the current budget of ", self.code, " ", self.carbon_budget_per_current_year*1e9)
                        self.diff_budget_and_emissions = self.carbon_budget_per_current_year * 1e9 - self.total_emissions # make units the same in tonnes so carbon budgets need to be in tonnes from gigatonnes 
                        self.diff_budget_and_emissions_ratio = (self.total_emissions / (self.carbon_budget_per_current_year*1e9))  # make units the same in tonnes so carbon budgets need to be in tonnes from gigatonnes 
                        if self.code == "USA":
                                print("this is the ratio of emissions to budget of ", self.code, " ", self.diff_budget_and_emissions_ratio)
        
        def calculate_national_gini_coefficient(self):
                        
                        """
                        Description: 
                                A method computing the gini coefficient of the country as the relative absolute mean difference.
        
                        Parameters:
                                None
        
                        """
                        # Compute mean of income deciles 
                        mean_income = sum([getattr(self, f'decile{decile_num}_abs') for decile_num in range(1, 11)]) / 10
                        #denominator of the final formula
                        denominator = 2 * 10**2 * mean_income
                        # Loop over all deciles and compute the absolute difference to the other deciles
                        numerator = 0
                        for decile_num in range(1, 11):
                                decile_income = getattr(self, f'decile{decile_num}_abs')
                                for other_decile_num in range(1, 11):
                                        other_decile_income = getattr(self, f'decile{other_decile_num}_abs')
                                        numerator += abs(decile_income - other_decile_income)
                        # Compute the gini coefficient
                        self.gini_hh = numerator / denominator


        def __repr__(self): # This is the string representation of the object
                # Retrieve the dynamic attributes by removing the 'country_' prefix and format them.
                attributes = [f"{key.split('country_')[1]}: {getattr(self, key)}" for key in self.__dict__ if key.startswith('country_')]
                return f"{self.__class__.__name__}({', '.join(attributes)})"