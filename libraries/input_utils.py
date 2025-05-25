"""
input_utils.py
This module provides utility functions for validating user input.
"""


from datetime import datetime


def get_valid_date(prompt: str) -> str:
    while True:
        date_str = input(prompt).strip()
        try:
            date_obj = datetime.strptime(date_str, "%m-%d")
            if datetime(2021, 1, 1) <= datetime(2021, date_obj.month, date_obj.day) <= datetime(2021, 12, 30):
                return f"2021-{date_str}"
            else:
                print("⚠️  Date must be between 01-01 and 12-30")
        except ValueError:
            print("⚠️  Invalid date format. Use MM-DD")


def get_valid_hour(prompt: str) -> str:
    while True:
        hour_str = input(prompt).strip()
        if hour_str.isdigit():
            hour = int(hour_str)
            if 0 <= hour <= 23:
                return f"{hour:02d}:00"
            else:
                print("⚠️  Hour must be between 0 and 23")
        else:
            print("⚠️  Please enter a valid hour as an integer (e.g., 9 or 17)")


def get_valid_int(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"⚠️  Value must be between {min_val} and {max_val}")
        except ValueError:
            print("⚠️  Please enter an integer")


def get_valid_scenario(prompt: str) -> str:
    scenarios = ["normal"]
    while True:
        scenario = input(prompt).strip().lower()
        if scenario.isalnum():
            if scenario in scenarios:
                return scenario
            else:
                print(f"⚠️  Invalid scenario name. Available scenarios: {', '.join(scenarios)}")
        else:
            print("⚠️  Scenario name must be alphanumeric")