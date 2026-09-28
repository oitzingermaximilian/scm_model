import pandas as pd


def scenario_base_with_club(raw_data):
    if "timeseries" in raw_data:
        df_ts = raw_data["timeseries"]
        df_ts["ets_SCENARIO"] = df_ts["c_ets_mid"]
        if "g" in df_ts.columns:
            df_ts["g_SCENARIO"] = df_ts["g"]
    return raw_data


def scenario_without_club(raw_data):
    if "timeseries" in raw_data:
        df_ts = raw_data["timeseries"]
        df_ts["ets_SCENARIO"] = df_ts["c_ets_mid"]
        df_ts["g_SCENARIO"] = 0.0
    return raw_data


def make_scenario_ets_sensitivity(theta):
    def scenario_func(raw_data):
        if "timeseries" in raw_data:
            df_ts = raw_data["timeseries"]

            # Force the column to be numeric
            c_ets_numeric = pd.to_numeric(df_ts["c_ets_mid"], errors="coerce")
            df_ts["ets_SCENARIO"] = c_ets_numeric * theta

            if "g" in df_ts.columns:
                df_ts["g_SCENARIO"] = pd.to_numeric(df_ts["g"], errors="coerce")

        return raw_data

    theta_str = str(theta).replace('.', '_')
    scenario_func.__name__ = f"scenario_with_club_ets_theta_{theta_str}"
    return scenario_func


def make_scenario_ets_sensitivity_without_club(theta):
    def scenario_func(raw_data):
        if "timeseries" in raw_data:
            df_ts = raw_data["timeseries"]

            # Force the column to be numeric
            c_ets_numeric = pd.to_numeric(df_ts["c_ets_mid"], errors="coerce")
            df_ts["ets_SCENARIO"] = c_ets_numeric * theta

            # ZERO OUT THE CLUB OBLIGATION
            if "g" in df_ts.columns:
                df_ts["g_SCENARIO"] = 0.0

        return raw_data

    theta_str = str(theta).replace('.', '_')
    scenario_func.__name__ = f"scenario_without_club_ets_theta_{theta_str}"
    return scenario_func


# ==========================================
# DICTIONARY REGISTRY
# ==========================================
thetas = [0.25, 0.5, 0.75, 1.5, 2.0,3.0,4.0,5.0]

SCENARIO_DICT = {
    "Base_With_Club": scenario_base_with_club,
    "Base_Without_Club": scenario_without_club,
}

# 1. Register "With Club" Sensitivities
for t in thetas:
    SCENARIO_DICT[f"With_Club_ETS_x{t}"] = make_scenario_ets_sensitivity(t)

# 2. Register "Without Club" Sensitivities
for t in thetas:
    SCENARIO_DICT[f"Without_Club_ETS_x{t}"] = make_scenario_ets_sensitivity_without_club(t)