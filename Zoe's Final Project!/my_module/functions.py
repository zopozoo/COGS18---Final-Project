import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

from classes import User

def train_model():
    # dataframe
    original_df = pd.read_csv('data/Sleep and Activity.csv')
    df = original_df.rename(columns={'Average_steps': 'Daily Steps', \
                                'Average_calories': 'Calories Burnt',\
                                'Average_sleep_hours': 'Hours of Sleep'})\
                                .drop(columns=['Average_time_in_bed_hours', 'TotalDistance'])
    
    X = df['Daily Steps']
    y = df['Calories Burnt']
    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 23)
    # reshaping data for model training
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    # the model
    model = LinearRegression()
    # Creating and training the Linear Regression model parameters 
    model.fit(X_train, y_train)
    # Save the model to a file using pickle
    print('Model is trained')
    return model


def create_user_profile(trained_model):
    print('What is your name?')
    input_name = input()
    print('What is your step goal?')
    input_goal = input()
    user = User(str(input_name), int(input_goal), trained_model)
    print(f'User account created: {user.about()}')
    return user