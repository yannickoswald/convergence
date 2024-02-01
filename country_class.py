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

        """

        def __init__(self, scenario, **kwargs):

                """
                Parameters:
                        The parameters here are given as attributes and are specified in the class doc.
                """
                self.scenario = scenario
                self.year = 2022  # All countries are initialized with 2022 data
                self.cagr_by_decile = {}  # Necessary for convergence growth rates in scenario method
                self.income_hh_trajectory = {}  # Necessary for plotting the trajectory of the countrys
                self.gdppc_trajectory = {}  # Necessary for plotting the trajectory of the countrys
                self.decile_trajectories = {}  # Necessary for plotting the trajectory of the countrys deciles here each dictionary entry is another dictionary with the years as keys and the decile incomes as values

                # Dictionary mapping kwargs names to class attribute names
                attribute_mapping = {
                        'index': 'id',
                        'region_name': 'region',
                        'region_code': 'region_code',
                        'country_code': 'code',
                        'mean': 'hh_mean',
                        'gdp_pc_ppp_2017': 'gdp_pc',
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
                        }

                # Set attributes based on mapping
                for kwarg_attr, class_attr in attribute_mapping.items():
                        if kwarg_attr in kwargs:
                                setattr(self, class_attr, kwargs[kwarg_attr])
                        else:
                                print(f"Warning: '{kwarg_attr}' not found in kwargs. Attribute '{class_attr}' not set.")

                # Set other attributes dynamically with 'country_' prefix
                for key, value in kwargs.items():
                        if key not in attribute_mapping:
                                setattr(self, f'country_{key}', value)


        def save_current_state(self):
                """
                Description: 
                        A class representing one country and its
                        defining parameters.
                
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

                # add and save current decile incomes to the decile trajectories where every decile in the dictionary is another dictionary with the years as keys and the decile incomes as values
                for decile_num in range(1, 11):
                        decile_income = getattr(self, f'decile{decile_num}_abs')
                        if f'decile{decile_num}' not in self.decile_trajectories:
                                self.decile_trajectories[f'decile{decile_num}'] = {}
                        self.decile_trajectories[f'decile{decile_num}'][self.year] = decile_income
                
        def growth(self):

                # save current state
                self.save_current_state()

                # compute new state
                # loop over all deciles and apply the growth rate
                for decile_num in range(1, 11):
                        decile_income = getattr(self, f'decile{decile_num}_abs')
                        # use self.cagr_by_decile to get the growth rate for the decile
                        cagr = self.cagr_by_decile[f'decile{decile_num}']
                        # calculate the new income
                        new_income = decile_income * (1 + cagr)
                        # set the new income
                        setattr(self, f'decile{decile_num}_abs', new_income)


                # COMPUTE NEW AGGREGATE QUANTITIES        
                # compute NEW mean country household cons. exp.income as average of decile incomes
                self.hh_mean = sum([getattr(self, f'decile{decile_num}_abs') for decile_num in range(1, 11)]) / 10
                # compute NEW gdp per capita
                # the gdp ratio is conditional on the mean household income see script first data explorations
                # if it is lower than 10000 then the ratio is fixed at their empirically observed ratio
                if self.hh_mean < 10000:
                         self.gdp_pc =  self.hh_mean / self.gdp_hh_income_ratio 
                else:
                         self.gdp_pc = self.hh_mean / 0.46 # this is the ratio of gdp to mean household income for countries with mean household income > 10000 which seems to be a reasonable assumption according to the cross sectional country data 

                self.year += 1 # increase the year by one

        def __repr__(self): # This is the string representation of the object
                # Retrieve the dynamic attributes by removing the 'country_' prefix and format them.
                attributes = [f"{key.split('country_')[1]}: {getattr(self, key)}" for key in self.__dict__ if key.startswith('country_')]
                return f"{self.__class__.__name__}({', '.join(attributes)})"