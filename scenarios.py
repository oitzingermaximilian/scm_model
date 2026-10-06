import pandas as pd
import numpy as np


def make_shifted_ets_scenario(incline_start_year, with_club=True):
    """
    Verschiebt den Startpunkt für den ETS-Preisanstieg und aktiviert/deaktiviert den Club.
    - Steigung 1 (4.37 €/yr): Von 2026 bis zum 'incline_start_year'
    - Steigung 2 (20.01 €/yr): Dauert exakt 10 Jahre
    - Steigung 3 (20.52 €/yr): Restliche Jahre bis 2050
    """

    def scenario_func(raw_data):
        if "timeseries" in raw_data:
            df_ts = raw_data["timeseries"]

            # Jahre auslesen - ignore strings wie "Source" durch errors="coerce"
            if "year" in df_ts.columns:
                years = pd.to_numeric(df_ts["year"], errors="coerce").values
            else:
                years = 2026 + np.arange(len(df_ts))

            # Die drei Steigungen
            slope_1 = 4.37
            slope_2 = 20.01
            slope_3 = 20.52

            new_prices = []
            current_price = 80.00

            for y in years:
                # Falls y keine gültige Zahl ist (z.B. der Text "Source"), überspringen
                if pd.isna(y):
                    new_prices.append(np.nan)
                    continue

                if y == 2026:
                    current_price = 80.00
                elif y <= incline_start_year:
                    # Phase 1: Flacher Anstieg
                    current_price += slope_1
                elif y <= incline_start_year + 10:
                    # Phase 2: Mittlerer Anstieg (Dauer: 10 Jahre)
                    current_price += slope_2
                else:
                    # Phase 3: Steiler Anstieg (bis Ende)
                    current_price += slope_3

                new_prices.append(round(current_price, 2))

            # 1. ETS Preis überschreiben
            df_ts["ets_SCENARIO"] = new_prices

            # 2. Club Obligation (g) an- oder ausschalten
            if "g" in df_ts.columns:
                if with_club:
                    df_ts["g_SCENARIO"] = df_ts["g"]
                else:
                    df_ts["g_SCENARIO"] = 0.0

        return raw_data

    club_status = "With_Club" if with_club else "Without_Club"
    scenario_func.__name__ = f"{club_status}_ets_incline_start_year_{incline_start_year}"
    return scenario_func


# ==========================================
# DICTIONARY REGISTRY
# ==========================================
# Die Startjahre für die Steigung (2030 ist dein Base Case, 2050 ist der flache Low Case)
incline_years = [2030, 2035, 2040, 2045, 2050]

SCENARIO_DICT = {}

for y in incline_years:
    # 1. Register "With Club" Szenarien
    SCENARIO_DICT[f"With_Club_ets_incline_start_year_{y}"] = make_shifted_ets_scenario(incline_start_year=y,
                                                                                       with_club=True)

    # 2. Register "Without Club" Szenarien
    SCENARIO_DICT[f"Without_Club_ets_incline_start_year_{y}"] = make_shifted_ets_scenario(incline_start_year=y,
                                                                                          with_club=False)