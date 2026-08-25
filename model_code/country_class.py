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
                hh_income_to_gdp_ratio      - ratio of mean household income to GDP (mapped from 'hh_income_to_gdp_ratio')
                total_emissions             - emissions of the country (mapped from 'emissions') per year total

        """

        def __init__(self, scenario, **kwargs):
                """
                Parameters:
                        The parameters here are given as attributes and are specified in the class doc.
                """
                # =========================================================
                # 1. CORE ATTRIBUTES & STATE VARIABLES
                # =========================================================
                self.scenario = scenario
                self.year = 2022  # All countries are initialized with 2022 data
                
                self.cagr_by_decile = {}      # Necessary for convergence growth rates
                self.cagr_average = None      # Necessary for average country convergence growth rate
                self.pop_growth_rate = None   # Dynamically changes based on population assumptions
                
                self.carbon_budget_per_current_year = None
                self.diff_budget_and_emissions = None
                self.diff_budget_and_emissions_ratio = None

                # =========================================================
                # 2. TRAJECTORY DICTIONARIES (DATA LOGGING)
                # =========================================================
                # Macro-economic & demographic trajectories
                self.income_hh_trajectory = {}
                self.gdppc_trajectory = {}
                self.gdp_trajectory = {}
                self.population_trajectory = {}
                self.gini_coefficient_trajectory = {}
                
                # Emissions & carbon intensity trajectories
                self.carbon_intensity_trajectory = {}
                self.emissions_trajectory = {}
                self.carbon_emissions_pc_trajectory = {}
                
                # Decile-specific trajectories
                self.decile_trajectories = {}
                self.decile_emissions_trajectories = {f'decile{d}': {} for d in range(1, 11)}
                self.decile_emissions_trajectories_pc = {f'decile{d}': {} for d in range(1, 11)}

                # =========================================================
                # 3. KWARGS MAPPING & DYNAMIC ATTRIBUTE ASSIGNMENT
                # =========================================================
                attribute_mapping = {
                        'index': 'id',
                        'region_name': 'region',
                        'region_code': 'region_code',
                        'country_code': 'code',
                        'mean': 'hh_mean',
                        'gdp_pc_ppp_2017': 'gdp_pc',
                        'gini': 'gini_hh',
                        'carbon_intensity': 'carbon_intensity',
                        'carbon_intensity_trend': 'carbon_intensity_trend',
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
                        'gdp_to_mean_hh_income_ratio': 'hh_income_to_gdp_ratio',
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

                # =========================================================
                # 4. PRE-CALCULATION SETUP & ELASTICITY INITIALIZATION
                # =========================================================
                # Set initial 2022 values before calculations
                self.income_hh_trajectory[self.year] = self.hh_mean * 365 
                self.gdppc_trajectory[self.year] = self.gdp_pc 
                self.population_trajectory[self.year] = self.population 
                self.carbon_intensity_trajectory[self.year] = self.carbon_intensity 
                
                # Elasticity attributes initialization
                self.base_carbon_intensity = self.carbon_intensity
                self.decile_carbon_intensities = {}
                self.base_A = None
                
                # =========================================================
                # 5. BASELINE PIPELINE (CALCULATE & LOG 2022 STATE)
                # =========================================================
                # Execute emission calculation for 2022 to populate baseline values
                self.update_emissions()
                
                # Set initial macro trajectories based on the dynamically calculated emissions
                self.emissions_trajectory[self.year] = self.total_emissions
                self.carbon_emissions_pc_trajectory[self.year] = self.total_emissions / self.population if self.population > 0 else 0

                # Immediately log the baseline 2022 year into all remaining trajectory dictionaries
                self.save_current_state()


        def save_current_state(self):
                """
                Description: 
                        A method saving the current state of the country across all kinds of variables. 
                        This is necessary for plotting the trajectories of the country's income, gdp, and emissions.
                """
                #### ECONOMIC & DEMOGRAPHIC VARIABLES ####
                self.income_hh_trajectory[self.year] = self.hh_mean
                self.gdppc_trajectory[self.year] = self.gdp_pc
                self.gdp_trajectory[self.year] = self.gdp_pc * self.population
                self.population_trajectory[self.year] = self.population
                self.gini_coefficient_trajectory[self.year] = self.gini_hh 
                
                #### MACRO EMISSION VARIABLES ####
                self.carbon_intensity_trajectory[self.year] = self.carbon_intensity 
                self.emissions_trajectory[self.year] = self.total_emissions
                self.carbon_emissions_pc_trajectory[self.year] = self.total_emissions / self.population if self.population > 0 else 0
                
                #### DECILE TRAJECTORIES ####
                for decile_num in range(1, 11):
                        # Ensure nested dictionaries exist
                        if f'decile{decile_num}' not in self.decile_trajectories:
                                self.decile_trajectories[f'decile{decile_num}'] = {}
                        if f'decile{decile_num}' not in self.decile_emissions_trajectories:
                                self.decile_emissions_trajectories[f'decile{decile_num}'] = {}
                        if f'decile{decile_num}' not in self.decile_emissions_trajectories_pc:
                                self.decile_emissions_trajectories_pc[f'decile{decile_num}'] = {}

                        # 1. Save decile incomes
                        self.decile_trajectories[f'decile{decile_num}'][self.year] = getattr(self, f'decile{decile_num}_abs')
                        
                        # 2. Save decile absolute emissions
                        self.decile_emissions_trajectories[f'decile{decile_num}'][self.year] = getattr(self, f'decile{decile_num}_emissions') 
                        
                        # 3. Save decile per capita emissions (This fixes your plotting issue!)
                        self.decile_emissions_trajectories_pc[f'decile{decile_num}'][self.year] = getattr(self, f'decile{decile_num}_emissions_pc')


        def technological_change(self):

                """
                Description: 
                        A method computing the technological change of the country expressed as a change in carbon intensity.

                Parameters:
                        None

                """
                # Define subprocedures for the sigmoidal function and the weighted average to compute a model of technological change

                def sigmoid(t, kappa=0.1, t0=50):
                        """
                        Sigmoid function for calculating the weight w(t).
                        
                        Parameters:
                        - t: The time variable.
                        - kappa: Steepness of the curve.
                        - t0: Midpoint of the sigmoid, where w(t) = 0.5.
                        """
                        return 1 / (1 + np.exp(-kappa * (t - t0)))

                def weighted_average(t, y, z, kappa=0.1, t0=50):
                        """
                        Calculates the weighted average of y and z over time using a sigmoidal function for weights.
                        
                        Parameters:
                        - t: Time variable, can be a scalar or a numpy array.
                        - y: The y variable.
                        - z: The z variable.
                        - k, t0: Parameters for the sigmoid function.
                        """
                        w = sigmoid(t, kappa, t0)
                        return (1 - w) * y + w * z

                # DIFFERENTIATE TECHNOLOGICAL CHANGE ASSUMPTIONS
                #################################################
                # 1. Determine the uniform improvement rate for this year
                improvement_rate = 0.0
                
                if self.year < 2021:
                        improvement_rate = self.carbon_intensity_trend
                else:   
                        if self.scenario.tech_hysteresis_assumption == "optimistic_degrowth":
                                if self.cagr_average > 0:
                                        modelled_trend = -0.015 * np.log(self.gdp_pc) + 0.1309
                                        improvement_rate = weighted_average(self.year, modelled_trend, self.scenario.final_improvement_rate, kappa=self.scenario.kappa, t0=self.scenario.t0)
                                else:
                                        improvement_rate = self.scenario.final_improvement_rate 

                        ### OLD         
                        #elif self.scenario.tech_hysteresis_assumption == "pessimistic_degrowth":
                         #       if self.cagr_average > 0:
                          #              modelled_trend = -0.015 * np.log(self.gdp_pc) + 0.1309
                           #             improvement_rate = weighted_average(self.year, modelled_trend, self.scenario.final_improvement_rate, k=self.scenario.k, t0=self.scenario.t0)
                            #    else:
                             #           improvement_rate = 0 

                        elif self.scenario.tech_hysteresis_assumption == "pessimistic_degrowth":
                                ### just countries evolve according to the semi log model of technological change, so path dependency is not considered, but the semi log model is applied to all countries, so the technological change rate is always the same for all countries and only depends on the gdp per capita of the country
                                modelled_trend = -0.015 * np.log(self.gdp_pc) + 0.1309
                                improvement_rate = weighted_average(self.year, modelled_trend, self.scenario.final_improvement_rate, kappa=self.scenario.kappa, t0=self.scenario.t0)
                
                # 2. Apply the rate to the country's macro carbon intensity
                self.carbon_intensity = self.carbon_intensity * (1 + improvement_rate)
                
                # 3. CLEAR UNIFORM EVOLUTION: Apply the exact same rate to every decile's carbon intensity
                assumption = getattr(self.scenario, 'elasticity_assumption', 'off')
                if assumption in ["constant", "income_dependent"]:
                        # Ensure the dictionary isn't empty before trying to evolve it
                        if self.decile_carbon_intensities:
                                for d in range(1, 11):
                                        self.decile_carbon_intensities[f'decile{d}'] *= (1 + improvement_rate)


                # PATH B: The New Absolute Elasticity Method
                elif assumption == "absolute_income_elasticity":
                        if getattr(self, 'current_A', None) is not None:
                                self.current_A *= (1 + improvement_rate)


        def update_emissions(self):

                """
                Description: 
                        A method computing the emissions of the country by summing the 
                        heterogeneous emissions of each decile based on absolute elasticity.
                """

                assumption = getattr(self.scenario, 'elasticity_assumption', 'off')

                valid_assumptions = ["off", "constant", "absolute_income_elasticity"]
                assert assumption in valid_assumptions, f"Path Error: Unrecognized elasticity assumption '{assumption}'."
                assert hasattr(self.scenario, 'income_goal'), "Path Error: 'income_goal' parameter did not pass from the interface to the scenario."
                if assumption != "off":
                        assert hasattr(self.scenario, 'elasticity_value'), "Path Error: 'elasticity_value' missing but assumption is turned on."
                
                self.decile_emissions_pc_dist = []
                pop_d = self.population / 10.0 # population per decile
                macro_gdp = self.gdp_pc * self.population
                decile_incomes = [getattr(self, f'decile{d}_abs') for d in range(1, 11)]
                mean_income = sum(decile_incomes) / 10.0 if sum(decile_incomes) > 0 else 1.0
                
                if assumption == "off":
                        self.total_emissions = self.carbon_intensity * macro_gdp / 1000 
                        
                        for d in range(1, 11):
                                yd = getattr(self, f'decile{d}_abs')
                                gdp_d = macro_gdp * (yd / (10 * mean_income))
                                emissions_d = (self.carbon_intensity * gdp_d) / 1000
                                emissions_d_pc = emissions_d / pop_d if pop_d > 0 else 0
                                
                                self.decile_emissions_pc_dist.append(emissions_d_pc)
                                setattr(self, f'decile{d}_emissions', emissions_d)
                                setattr(self, f'decile{d}_emissions_pc', emissions_d_pc)
                                
                elif assumption in ["constant", "income_dependent"]:
                        
                        # --- CLEAR BASE YEAR CALIBRATION ---
                        # We do this math ONLY ONCE during __init__ to lock in the starting carbon intensities.
                        if not self.decile_carbon_intensities:
                                epsilon = self.get_current_elasticity()
                                total_e_2022 = (self.carbon_intensity * macro_gdp) / 1000
                                
                                # Absolute elasticity formula: Emissions = Pop_d * A * Y_d^epsilon
                                sum_y_eps = sum(yd ** epsilon for yd in decile_incomes)
                                
                                # Find the structural scalar 'A' for the base year
                                base_A = total_e_2022 / (pop_d * sum_y_eps) if sum_y_eps > 0 else 0
                                
                                # Calculate and store the EXACT starting carbon intensity (kg/$PPP) for each decile
                                for i, yd in enumerate(decile_incomes):
                                        d = i + 1
                                        e_d = pop_d * base_A * (yd ** epsilon) # Absolute emissions in tonnes
                                        gdp_d = macro_gdp * (yd / (10 * mean_income))
                                        
                                        # Convert back to kg per dollar to store as the decile's Carbon Intensity
                                        ci_d = (e_d * 1000) / gdp_d if gdp_d > 0 else 0 
                                        self.decile_carbon_intensities[f'decile{d}'] = ci_d

                        # --- CALCULATE EMISSIONS FOR CURRENT YEAR ---
                        total_scaled_emissions = 0
                        
                        for i, yd in enumerate(decile_incomes):
                                d = i + 1
                                gdp_d = macro_gdp * (yd / (10 * mean_income))
                                
                                # Simply multiply the decile's GDP by its specific, evolved Carbon Intensity
                                ci_d = self.decile_carbon_intensities[f'decile{d}']
                                emissions_d = (ci_d * gdp_d) / 1000
                                emissions_d_pc = emissions_d / pop_d if pop_d > 0 else 0
                                
                                self.decile_emissions_pc_dist.append(emissions_d_pc)
                                setattr(self, f'decile{d}_emissions', emissions_d)
                                setattr(self, f'decile{d}_emissions_pc', emissions_d_pc)
                                
                                total_scaled_emissions += emissions_d
                                
                        self.total_emissions = total_scaled_emissions

                # ---------------------------------------------------------
                # PATH 3: The New Absolute Income Elasticity
                # ---------------------------------------------------------
                elif assumption == "absolute_income_elasticity":
                        epsilon = self.get_current_elasticity()
                        
                        # Calibrate the base year once to find the structural scalar
                        if getattr(self, 'current_A', None) is None:
                                total_e_2022 = (self.carbon_intensity * macro_gdp) / 1000
                                sum_y_eps = sum(yd ** epsilon for yd in decile_incomes)
                                self.current_A = total_e_2022 / (pop_d * sum_y_eps) if sum_y_eps > 0 else 0

                        # ---> THIS IS THE FIX: Initialize the counter to 0 <---
                        total_scaled_emissions = 0

                        # Calculate current year emissions
                        for i, yd in enumerate(decile_incomes):
                                d = i + 1
                                emissions_d = pop_d * self.current_A * (yd ** epsilon)
                                emissions_d_pc = emissions_d / pop_d if pop_d > 0 else 0
                                
                                # Back-calculate the apparent Carbon Intensity for plotting compatibility
                                gdp_d = macro_gdp * (yd / (10 * mean_income))
                                self.decile_carbon_intensities[f'decile{d}'] = (emissions_d * 1000) / gdp_d if gdp_d > 0 else 0

                                
                                self.decile_emissions_pc_dist.append(emissions_d_pc)
                                setattr(self, f'decile{d}_emissions', emissions_d)
                                setattr(self, f'decile{d}_emissions_pc', emissions_d_pc)
                                total_scaled_emissions += emissions_d
                                
                        self.total_emissions = total_scaled_emissions

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

                        # get the decile income and decile specific growth rate
                        decile_income = getattr(self, f'decile{decile_num}_abs')
                        # use self.cagr_by_decile to get the growth rate for the decile
                        cagr = self.cagr_by_decile[f'decile{decile_num}']

                        # Then distinguish various growth cases for the different scenarios and assumptions

                        ##############################################################################
                        ######## AFTER REVISION 2.0 DISTINGUISH BETWEEN UP TO 2100 or not ########  
                        ############################################################################## 
     
                        if self.scenario.run_until_2100 == "on":
                        # check if the year is less than 2100 because above the convergence year 
                        # the counterfactual case of ongoing economic growth for the richest needs to be considered
                                if self.year < self.scenario.end_year:

                                        # calculate the new income
                                        # distinguish between steady state and non steady state assumption
                                        if self.scenario.steady_state_high_income_assumption == "on":
                                
                                                if decile_income >= self.scenario.income_goal:
                                                        new_income = decile_income
                                                else:
                                                        new_income = decile_income * (1 + cagr)

                                        # here the deciles that are already above the income goal are assumed to grow at the historical growth rate of the country        
                                        elif self.scenario.steady_state_high_income_assumption == "on_with_growth":

                                                if self.gdp_pc_historical_growth > 0:
                                                         new_income = decile_income * (1 + 0.02) # Growth rate is set so that the GDPpc growth rate matches SSP2 IIASA version
                                                else:
                                                         new_income = decile_income * (1 + 0.02) # Growth rate is set so that the GDPpc growth rate matches SSP2 IIASA version
                                
                                                  #if decile_income >= self.scenario.income_goal + 1e-3: # add a small number to avoid floating point errors
                                                        # if the income is already above the goal then apply the historical growth rate
                                                        # but only if the growth rate is positive, otherwise do not apply the growth rate
                                                     #     if self.gdp_pc_historical_growth > 0:
                                                           #       new_income = decile_income * (1 + self.gdp_pc_historical_growth)
                                                        #  else: # keep the income as it is
                                                              #    new_income = decile_income * (1 + 0.01) # assume low positive growth rate of 1% for the richest decile
                                                  #else:
                                                          #new_income = decile_income * (1 + cagr)

                                        elif self.scenario.steady_state_high_income_assumption == "off":    

                                                new_income = decile_income * (1 + cagr)

                                if self.year >= self.scenario.end_year:

                                        if self.scenario.steady_state_high_income_assumption == "on_with_growth":

                                                if self.gdp_pc_historical_growth > 0:
                                                         new_income = decile_income * (1 + 0.02) # Growth rate is set so that the GDPpc growth rate matches SSP2 IIASA version
                                                else:
                                                         new_income = decile_income * (1 + 0.02) # Growth rate is set so that the GDPpc growth rate matches SSP2 IIASA version

                                                #if decile_income > self.scenario.income_goal + 1e-3:
                                                        # if the income is already above the goal then apply the historical growth rate
                                                        # but only if the growth rate is positive, otherwise do not apply the growth rate
                                                 #       if self.gdp_pc_historical_growth > 0:
                                                  #              new_income = decile_income * (1 + self.gdp_pc_historical_growth)
                                                   #     else:
                                                    #            new_income = decile_income * (1 + 0.01) # assume low positive growth rate of 1% for the richest decile
                                        #else:
                                                 #       new_income = decile_income # do nothing

                                        else:
                                                new_income = decile_income # do nothing also because all should be in steady state


                        ##############################################################################
                        ######## BEFORE REVISION 2.0 THIS WAS THE CASE FOR THE SCENARIO "OFF" ########  
                        ##############################################################################     
                                                         
                        elif self.scenario.run_until_2100 == "off":

                                # calculate the new income
                                # distinguish between steady state and non steady state assumption
                                if self.scenario.steady_state_high_income_assumption == "on":
                        
                                        if decile_income >= self.scenario.income_goal:
                                                new_income = decile_income
                                        else:
                                                new_income = decile_income * (1 + cagr)

                                # here the deciles that are already above the income goal are assumed to grow at the historical growth rate of the country        
                                elif self.scenario.steady_state_high_income_assumption == "on_with_growth":


                                        if self.gdp_pc_historical_growth > 0:
                                                         new_income = decile_income * (1 + self.gdp_pc_historical_growth)
                                        else:
                                                         new_income = decile_income * (1 + 0.01) # assume low positive growth rate of 1% for all deciles
                        
                                        #if decile_income >= self.scenario.income_goal:
                                                # if the income is already above the goal then apply the historical growth rate
                                                # but only if the growth rate is positive, otherwise do not apply the growth rate
                                         #       if self.gdp_pc_historical_growth > 0:
                                         #               new_income = decile_income * (1 + self.gdp_pc_historical_growth)
                                         #       else: # keep the income as it is
                                         #               new_income = decile_income
                                        #else:
                                         #       new_income = decile_income * (1 + cagr)

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
                        self.gdp_pc =  self.hh_mean / self.hh_income_to_gdp_ratio                   
                  
                elif self.scenario.gdp_assumption == "model_ratio":
                        # We use the exact intersection point of Fit 1 (-0.0000571x + 0.67) and y=0.40 to ensure continuity
                        if self.hh_mean < 4728.55: 
                                self.hh_income_to_gdp_ratio = -0.0000571 * (self.hh_mean) + 0.67
                        else:
                                self.hh_income_to_gdp_ratio = 0.40 # Constant structural scalar for high-income nations
                        
                        # Apply the ratio to calculate the GDPpc linearly without the asymptotic cap
                        self.gdp_pc = self.hh_mean  / self.hh_income_to_gdp_ratio


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
                                        self.pop_growth_rate = 0.088 - 0.020*np.log10(self.gdp_pc) ## new population weighted regression
                                        new_population = self.population * (1 + self.pop_growth_rate)
                                        self.population = new_population
                                else:
                                        # if degrowing from 2023 onwards assume UN_medium as well for "decoupling" countries, because it is not clear that planned degrowth economies revert socio-cultural norms to higher fertility rates
                                        self.pop_growth_rate = self.scenario.population_growth_rates.loc[str(self.code)][str(self.year)]
                                        if self.pop_growth_rate is not None:
                                                new_population = self.population * (1 + self.pop_growth_rate)
                                                self.population = new_population
                                        else:
                                                print("No growth rate found for", self.code, "in year", self.year)

                        elif self.scenario.population_hysteresis_assumption == "off":

                                self.pop_growth_rate = 0.088 - 0.020*np.log10(self.gdp_pc)
                                new_population = self.population * (1 + self.pop_growth_rate)
                                self.population = new_population

                        
                # elif self.scenario.pop_growth_assumption == "semi_log_model_elasticity":
                        # in this case we start from the empirical population growth rate in 2022 and then apply the semi log model equation for population growth change rate
                       #  if self.year == 2022:
                               #  self.pop_growth_rate = self.scenario.population_growth_rates.loc[str(self.code)][str(self.year)]
                              #   if self.pop_growth_rate is not None:
                                     #    new_population = self.population * (1 + self.pop_growth_rate)
                                     #    self.population = new_population
                       #  else:
                                # for the future years apply the semi log model equation for population growth change rate y = 0.09 - 0.01*log(x) 
                                # where x is gdp per capita and y is the population growth rate, so we must first calculate the derivative of y with respect to x
                                # which is dy/dx = -0.01/x
                                # then we calculate the elasticity of population growth with respect to gdp per capita which is the derivative of y with respect to x times x/y
                                # which is (dy/y)/(dx/x) = dy/dx *x/y = -0.01/x * x/y = -0.01/y

                                # then we also assume hysteresis in this assumption, meaning we only apply this elasticity if the gdp per capita increases, if it decreases we do not change population growth rate
                                # this means in our convergence scenario, for countries who apply deliberate degrowth, they do not actually get poorer in terms of living standards but only in gdp.
                                # so there is no reason to assume their population growth rate would change upward in this case.
                               #  if self.cagr_average > 0:
                                        # then we apply the elasticity as percentage change in population growth rate
                                #         elasticity = -0.01/self.pop_growth_rate
                                     #    self.pop_growth_rate =  self.pop_growth_rate * (1 + elasticity) # that is the higher the gdp per capita the lower the population growth rate but
                               #  else:
                                        # if the gdp per capita decreases we still apply the UN medium projections
                                 #        self.pop_growth_rate = self.scenario.population_growth_rates.loc[str(self.code)][str(self.year)]

                              #   new_population = self.population * (1 + self.pop_growth_rate)
                              #   self.population = new_population
                                
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

        def get_current_elasticity(self):

                """
                Description: 
                        A method computing the current elasticity of emissions with respect to income based on the scenario assumption.
                Parameters:
                        None

                """

                assumption = getattr(self.scenario, 'elasticity_assumption', 'off')
                
                # --- FIX: Added 'absolute_income_elasticity' here so it fetches the real value ---
                if assumption in ["constant", "absolute_income_elasticity"]:
                        return getattr(self.scenario, 'elasticity_value', 1.0)
                else:
                        return 1.0
             

        def __repr__(self): # This is the string representation of the object
                # Retrieve the dynamic attributes by removing the 'country_' prefix and format them.
                attributes = [f"{key.split('country_')[1]}: {getattr(self, key)}" for key in self.__dict__ if key.startswith('country_')]
                return f"{self.__class__.__name__}({', '.join(attributes)})"