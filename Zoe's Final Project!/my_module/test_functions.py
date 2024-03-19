import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pytest

from classes import User
from functions import create_user_profile, train_model

# test functions for class methods

# fixed user variable

@pytest.fixture
def user():
    trained_model = train_model()
    user = User('Zoe', 10000, trained_model)
    user.add_new_entry_normal(2000, '2024-03-16')
    user.add_new_entry_normal(4000, '2024-03-17')
    user.predict_calories_burnt()
    yield user

def test_User_class():
    Zoe = User('Zoe', 10000)
    assert isinstance(Zoe, User)
    assert isinstance(Zoe.about(), str)
    assert Zoe.about() == 'Zoe: 10000 daily steps goal'

def test_create_user_profile(monkeypatch):
    mock_input_values = ['Zoe', '10000'] # input returns strings
    trained_model = train_model()
    assert isinstance(trained_model, LinearRegression)
    # mock behavior of input function, removes and returns first value from the mock_input_values list
    monkeypatch.setattr('builtins.input', lambda: mock_input_values.pop(0)) 
    user = create_user_profile(trained_model)
    assert user.name == 'Zoe'
    assert int(user.goal) == 10000
    assert isinstance(user, User)
    assert user.model == trained_model  
    
def test_add_new_entry(monkeypatch, user):
    mock_input_values = ['1000', '2024-03-16']
    monkeypatch.setattr('builtins.input', lambda: mock_input_values.pop(0)) 
    result = user.add_new_entry()
    assert type(result) == pd.DataFrame
    assert list(result.iloc[0]) == ['2024-03-16', 2000]
    assert len(user.user_entries) == 3
    
def test_remove_last_entry(monkeypatch, user):
    mock_input_values = ['1000', '2024-03-16']
    monkeypatch.setattr('builtins.input', lambda: mock_input_values.pop(0)) 
    result1 = user.add_new_entry()
    result1_len = len(user.user_entries)
    result2 = user.remove_last_entry()
    assert type(result2) == pd.DataFrame
    assert len(result2) < len(result1)
    assert len(user.user_entries) == result1_len - 1
        
def test_predict_calories_burnt(user):
    result_table = user.predict_calories_burnt()
    assert isinstance(result_table, pd.DataFrame)
    assert 'Predicted Calories Burnt' in result_table.columns 
    assert isinstance(result_table.iloc[0, 2], float)
    
def test_average_steps(user):
    average_steps = user.average_steps()
    assert isinstance(average_steps, str)
    assert '3000' in average_steps
    
def test_step_goal_calorie_summary(user):
    summary = user.step_goal_calorie_summary()
    assert isinstance(summary, str)
    assert '6000' in summary
    assert '0%' in summary