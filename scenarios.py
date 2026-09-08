# scenarios.py
import pandas as pd


def scenario_high_ets(raw_data):
    """Scenario: High ETS Price Trajectory"""
    if "TimeSeries" in raw_data:
        df_ts = raw_data["TimeSeries"]
        # Point the active scenario column to the high column
        df_ts["ets_SCENARIO"] = df_ts["ets_high"]
    return raw_data


def scenario_low_ets(raw_data):
    """Scenario: Low ETS Price Trajectory"""
    if "TimeSeries" in raw_data:
        df_ts = raw_data["TimeSeries"]
        # Point the active scenario column to the low column
        df_ts["ets_SCENARIO"] = df_ts["ets_low"]
    return raw_data
