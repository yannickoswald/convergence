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
                ## ensure important representation attributes exist throughout
                self.id = None
                self.name = None
                self.year = None
                self.income = None ### average gdp per capita
                self.scenario = scenario ### assign the current scenario to each country object

                ## set the above attributes, and all other attributes, dynamically
                for key, value in kwargs.items():
                        #print(key, value)
                        setattr(self, key, value)


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
                
                years = self.


                cagr = (ending_value / beginning_value) ** (1 / years) - 1
                return cagr

                
        def growth(self):
                self.income = self.income * (1+self.growth_rate)  

        def __repr__(self):
                return f"{self.__class__.__name__}('ID: {self.id}')('name: {self.name}')('income: {self.income}')"

