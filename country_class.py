class Country():
        """
        Description: 
                A class representing one country and its
                defining parameters.
        
        Parameters:
                income_avg       - average income in a country
                gini             - gini coefficient of income
                growth_rate      - yearly economic growth rate
                year             - current year
                carbon_intensity - average carbon intensity per $ of income

                self.id = id
                self.name = name
                self.year = year
                self.income = income ### average gdp per capita
                self.g_rate = g_rate
                self.carbon_intensity = carbon_intensity
                self.gini_income = gini_income
                self.carbon_income_elasticity = carbon_income_elasticity
        """
        def __init__(self, scenario, **kwargs):
                """
                Parameters:
                        The parameters here are given as attributes and are specified in the class doc.
                """

                ## set the above attributes, and all other attributes, dynamically
                for key, value in kwargs.items():
                        setattr(self, f'country_{key}', value)



        def calculate_national_cagr(self, beginning_value, ending_value):
                """
                Calculate the Compound Annual Growth Rate (CAGR) for a country in a 
                specific scenario which is fixed over time for simplicity. This is also only for scenarios
                in which we only consider between country inequality and convergence.

                Parameters:
                        beginning_value (float): The initial value of the country income.
                        ending_value (float): The final value of the  country income.

                Returns:
                float: The CAGR as a decimal.
                """
                if years <= 0:
                        raise ValueError("Number of years should be greater than 0.")
                if beginning_value <= 0:
                        raise ValueError("Beginning value should be greater than 0.")
                
                years = self

                cagr = (ending_value / beginning_value) ** (1 / years) - 1
                return cagr
        
                
        def growth(self):
                self.income = self.income * (1+self.growth_rate)  

        def set_decile(self, decile_name, value):
                # Set the value for the given decile name
                self.deciles[decile_name] = value

        def get_decile(self, decile_name):
                # Retrieve the value for the given decile name
                return self.deciles.get(decile_name, None)

        def __str__(self):
                return str(self.deciles)


        def __repr__(self): # This is the string representation of the object
                # Retrieve the dynamic attributes by removing the 'country_' prefix and format them.
                attributes = [f"{key.split('country_')[1]}: {getattr(self, key)}" for key in self.__dict__ if key.startswith('country_')]
                return f"{self.__class__.__name__}({', '.join(attributes)})"