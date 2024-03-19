import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

class User():
    """
    A class that represents a user profile with a dataframe with user 
    inputed step and date information that can perform various methods.
    
    Attributes
    ----------
    name: str
        name of user
    goal: int
        step goal of user
    model: LinearRegression
        trained linear regression model
    """
    
    def __init__(self, name, goal, model=None):
        self.name = name
        self.goal = goal
        self.model = model
        self.user_entries = []
        self.user_df = pd.DataFrame()
        self.predicted_calories_df = pd.DataFrame()

    def about(self):
        return f'{self.name}: {self.goal} daily steps goal'
            
    def add_new_entry(self):
        
        """
        Prompts user for two input and assigns entered values input_steps and 
        input_date accordingly. Adds a new dictionary entry into the user_entries list which is then sorted
        to update user_df dataframe. 

        Returns
        -------
        user_df: pd.DataFrame
            updated user_df dataframe with new entry in corresponding 
            "Date" and "Daily Steps" columns         
        """
      
        print('Daily steps')
        input_steps = input()
        print('YYYY-MM-DD')
        input_date = input()
        return self.add_new_entry_normal(input_steps, input_date)

    
    def add_new_entry_normal(self, steps, date):
        user_input = {'Date': str(date), 'Daily Steps': int(steps)}
        self.user_entries.append(user_input)
        # print(self.user_entries)
        self.user_entries.sort(key=lambda x: x['Date']) # sort using date comparison
        self.user_df = pd.DataFrame(self.user_entries)
        # print(self.user_entries)
        return self.user_df
    
    
    def remove_last_entry(self):
        self.user_entries.pop(-1)
        self.user_df = pd.DataFrame(self.user_entries)
        return self.user_df 
    
    
    def get_all_entries(self):
        return self.user_df
    
    
    def predict_calories_burnt(self):
        
        """
        Predicts calories burnt based on linear regression model of entries in 
        "Daily Steps" column in user_df dataframe.
        
        Code uses snippets from ChatGPT but has been modified and adapted for my purposes.

        Returns
        -------
        result_table: pd.DataFrame
            concated dataframe with a new column "Predicted Calories Burnt" 
            populated with array of floats
        None:
            returned if trained model is not passed through when class 
            is initiatlized, if user_df is empty, or "Daily Steps" is not
            a column in user_df
                        
        """
                
        if self.model is None:
            print("Model not trained. Train model or load saved model.")
            return None
        
        if self.user_df.empty or 'Daily Steps' not in self.user_df.columns:
            print("No user entries or missing 'Daily Steps' column in user entries.")
            return None
    
        user_steps = self.user_df['Daily Steps'].values.reshape(-1, 1)
        predicted_calories = self.model.predict(user_steps)
        rounded_predicted_calories = np.round(predicted_calories)
        
        self.user_df['Predicted Calories Burnt'] = rounded_predicted_calories
            
        return self.user_df
   
        
    def average_steps(self):
        
        """
        Compares the average steps of all the entries in the dataframe with class initialized step goal.

        Returns
        -------
        str: 
            Formatted string congratulating user if their averaged steps in user_entries 
            is equal or more than initialized step goal
        str:
            Formatted string encouraging user if their averaged steps in user_entries 
            is less than initialized step goal
        """

        daily_steps = self.user_df['Daily Steps']
        average_steps = round(daily_steps.mean())
        if average_steps >= self.goal:
            return f'Congrats! You reached your goal of {self.goal} steps'\
                    f' by having an average step count of {average_steps}!'
        else:
            return f"You're doing great! You have an average step count of {average_steps}." \
       f" Just a little more to go until reaching {self.goal} steps."
        
        
    def plot_steps(self):
        
        """
        Plots every entry in the data frame based on its index.
        
        Cannot have more than one 'Date' entries.

        Returns
        -------
        plt: 
            Plot of all user entries over each "Date" index in user_df. Max y-value of the plot
            is 1000 more than user's initialized step goal
        """

        plt.plot(np.array(self.user_df['Date']), np.array(self.user_df['Daily Steps']))
        
        plt.xlabel('Date')
        plt.ylabel('Daily Steps')
        plt.title('Daily Steps over Time')

        plt.show()

        
    def step_goal_calorie_summary(self):
        """
        To get information such as the percent of all step entries that have 
        reached initialized step goal, total steps of the last seven entries,
        and total calories burnt of the last seven entries.

        Returns
        -------
        str: 
            formatted string with percent of step entries, sum steps and sum calories 
            burnt based on the last seven entries
                        
        """
        step_goal_reached = 0
        count = 0
        for entry in self.user_df['Daily Steps']:
            if entry >= self.goal:
                step_goal_reached += 1
            count += 1
        sum_steps = sum(self.user_df['Daily Steps'][-7:])
        sum_calories = sum(self.user_df['Predicted Calories Burnt'][-7:])
        return f'Yay! Keep working hard, {int((step_goal_reached/count)*100)}% of all'\
        ' your entries have reached your target step goal. In the past seven days (entries),'\
        f' you have walked a total of {sum_steps} steps and burned a total of {round(sum_calories)} calories!'
