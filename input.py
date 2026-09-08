import pandas as pd
import os


def load_data():
    """
    Reads the Excel file and returns a dictionary of raw DataFrames.
    """
    file_path = (
        r"C:\Users\maxoi\OneDrive\Desktop\Repositories\scm_model\scm_input_data.xlsx"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Returns a dictionary of DataFrames mapped by sheet name
    return pd.read_excel(file_path, sheet_name=None)


def prepare_model_data(raw_data):
    """
    Converts raw DataFrames into Pyomo-ready lists (for sets)
    and dictionaries (for parameters).
    """
    model_data = {}

    # ==========================================
    # 1. Process "TimeSeries" Sheet
    # ==========================================
    df_ts = raw_data["TimeSeries"].set_index("year")

    # ---------------------------------------------------------
    # STRICT INDEXING: Filter out any row that isn't a valid year
    # This converts strings like "Source" or URLs into NaN, then drops them.
    # ---------------------------------------------------------
    df_ts.index = pd.to_numeric(df_ts.index, errors="coerce")
    df_ts = df_ts[df_ts.index.notna()]
    df_ts.index = df_ts.index.astype(int)

    # --- Time Sets & Params ---
    model_data["y"] = df_ts.index.tolist()

    # Now we ONLY pull values specifically for the years in our clean set
    model_data["g"] = {y: float(df_ts.at[y, "scm_quota"]) for y in model_data["y"]}
    model_data["psi"] = {y: float(df_ts.at[y, "ci_factor"]) for y in model_data["y"]}

    # --- ETS Prices ---
    if "ets_SCENARIO" not in df_ts.columns:
        df_ts["ets_SCENARIO"] = df_ts["ets_middle"]
    model_data["c_ETS"] = {
        y: float(df_ts.at[y, "ets_SCENARIO"]) for y in model_data["y"]
    }

    # --- Sector Set & Rho Limits ---
    rho_mapping = {
        "rho_refining": "Industry",
        "rho_power": "Power",
        "rho_buildings": "Buildings",
    }
    model_data["sector"] = list(rho_mapping.values())

    # --- Region Set ---
    model_data["region"] = ["NO", "UK", "EU27"]

    rho_dict = {}
    for excel_col, sector_name in rho_mapping.items():
        if excel_col in df_ts.columns:
            # STRICT INDEXING: Only loop through valid model years
            for year in model_data["y"]:
                val = df_ts.at[year, excel_col]
                rho_dict[(sector_name, year)] = float(val)

    model_data["rho"] = rho_dict

    # ==========================================
    # 2. Process "economic_psc" Sheet
    # ==========================================
    df_psc = raw_data["economic_psc"]
    allowed_techs = [
        "PSC_cement",
        "PSC_iron_steel",
        "PSC_chemicals",
        "PSC_industry",
        "PSC_refining",
        "PSC_power",
        "PSC_bioenergy",
        "Direct_Air_Capture",
    ]

    # Force the column to text and strip any accidental spaces (e.g. "PSC_power ")
    df_psc["technology_k"] = df_psc["technology_k"].astype(str).str.strip()

    # FILTER: Only keep rows where the technology is in our exact list
    df_psc = df_psc[df_psc["technology_k"].isin(allowed_techs)]

    # --- Technology Sets ---
    # Now we only extract the valid ones that actually survived the filter
    tech_list = df_psc["technology_k"].tolist()

    model_data["tech"] = tech_list
    model_data["tech_point"] = [t for t in tech_list if t.startswith("PSC")]
    model_data["tech_rem"] = [t for t in tech_list if t == "Direct_Air_Capture"]

    # --- Cost and Efficiency Parameters ---
    dict_capex = {}
    dict_opex = {}
    dict_mu = {}

    # Identify the known data years from the columns (e.g., [2025, 2030, 2040, 2050])
    known_years = sorted(
        [int(c.split("_")[1]) for c in df_psc.columns if c.startswith("capex_")]
    )

    for _, row in df_psc.iterrows():
        k = row["technology_k"]

        # Base efficiency from the Excel sheet
        max_eff_base = float(row["eff_high"]) / 100.0

        for s in model_data["sector"]:
            # -------------------------------------------------
            # STRICT SECTOR-TECH MAPPING
            # -------------------------------------------------
            is_valid = False
            if k == "Direct_Air_Capture":
                is_valid = True  # DAC is valid for all sectors
            elif s == "Power" and k in ["PSC_power", "PSC_bioenergy"]:
                is_valid = True
            elif s == "Industry" and k in ["PSC_industry", "PSC_bioenergy"]:
                is_valid = True

            # Apply the efficiency ONLY if the combination is valid. Otherwise, 0%.
            dict_mu[(s, k)] = max_eff_base if is_valid else 0.0

            for t in model_data["y"]:
                # STEP-FUNCTION LOGIC
                past_years = [y for y in known_years if y <= t]
                if past_years:
                    applicable_year = max(past_years)
                else:
                    applicable_year = min(known_years)

                # ROUND TO 3 DECIMAL PLACES
                val_capex = float(row[f"capex_{applicable_year}"])
                val_opex = float(row[f"opex_{applicable_year}"])

                # Region 'i' dropped: Costs are indexed strictly by (sector, tech, year)
                dict_capex[(s, k, t)] = round(val_capex, 3)
                dict_opex[(s, k, t)] = round(val_opex, 3)

    model_data["c_cap_capex"] = dict_capex
    model_data["c_cap_opex"] = dict_opex
    model_data["mu"] = dict_mu

    # ==========================================
    # 3. Process "tech_expansion_init_cap" Sheet
    # ==========================================
    if "tech_expansion_init_cap" in raw_data:
        df_init = raw_data["tech_expansion_init_cap"]

        # Drop empty rows where essential data is missing
        df_init = df_init.dropna(
            subset=["region_i", "sector_s", "capture technlogy_k", "init_cap"]
        )

        dict_q_init = {}

        for _, row in df_init.iterrows():
            # Clean and map region ('EU' -> 'EU27')
            reg = str(row["region_i"]).strip()
            if reg == "EU":
                reg = "EU27"

            sec = str(row["sector_s"]).strip()

            # Normalize technology name: "PSC, refining" -> "PSC_refining"
            tech_raw = str(row["capture technlogy_k"]).strip()
            tech_clean = tech_raw.replace(", ", "_").replace(" ", "_").replace(",", "_")

            val = float(row["init_cap"])

            # Accumulate capacity if multiple facilities share the same region, sector, and tech
            key = (reg, sec, tech_clean)
            dict_q_init[key] = dict_q_init.get(key, 0.0) + val

        # Round final capacities to 3 decimal places
        model_data["Q_init"] = {k: round(v, 3) for k, v in dict_q_init.items()}
    else:
        model_data["Q_init"] = {}

        # Dummy values for testing capacity expansion limits in input.py
        # Add these into your model_data dictionary:

    dict_q_delta_abs = {}
    dict_q_ir_cap = {}

    for i in model_data["region"]:
        for s in model_data["sector"]:
            for k in model_data["tech"]:
                # Give a large absolute annual expansion headroom (e.g., 10 million tCO2/yr)
                dict_q_delta_abs[(i, s, k)] = 100000.0

                # Give a reasonable percentage growth rate (e.g., 20% growth per year)
                dict_q_ir_cap[(i, s, k)] = 0.20

    model_data["Q_delta_abs"] = dict_q_delta_abs
    model_data["Q_IR_cap"] = dict_q_ir_cap

    # ==========================================
    # Process "economic_storage" Sheet
    # ==========================================
    if "economic_storage" in raw_data:
        df_store = raw_data["economic_storage"]

        allowed_storage = [
            "Onshore_depleted_fields_reuseinfrastructure",
            "Onshore_depleted_fields_noinfrastructure",
            "Onshore_saline_aquifers",
            "Offshore_depleted_fields_reuseinfrastructure",
            "Offshore_depleted_fields_noinfrastructure",
            "Offshore_saline_aquifers",
        ]

        df_store["technology_k"] = df_store["technology_k"].astype(str).str.strip()
        df_store = df_store[df_store["technology_k"].isin(allowed_storage)]

        model_data["storage_tech"] = df_store["technology_k"].tolist()

        dict_store_capex_low = {}
        dict_store_capex_high = {}
        dict_store_opex = {}

        # Identify known years dynamically from columns containing 'capex_' and 'low'
        known_store_years = sorted(
            list(
                set(
                    [
                        int(c.split("_")[1].split(",")[0].strip())
                        for c in df_store.columns
                        if "capex_" in c and "low" in c
                    ]
                )
            )
        )

        for _, row in df_store.iterrows():
            k = row["technology_k"]

            for t in model_data["y"]:
                # Step-function logic
                past_years = [y for y in known_store_years if y <= t]
                app_year = max(past_years) if past_years else min(known_store_years)

                # Construct matching column names dynamically
                col_low = [
                    c
                    for c in df_store.columns
                    if f"capex_{app_year}" in c and "low" in c
                ][0]
                col_high = [
                    c
                    for c in df_store.columns
                    if f"capex_{app_year}" in c and "high" in c
                ][0]
                col_opex = [c for c in df_store.columns if f"opex_{app_year}" in c][
                    0
                ]

                # Region 'i' dropped: keys are indexed strictly by (storage_tech, year)
                dict_store_capex_low[(k, t)] = round(float(row[col_low]), 3)
                dict_store_capex_high[(k, t)] = round(float(row[col_high]), 3)
                dict_store_opex[(k, t)] = round(float(row[col_opex]), 3)

        model_data["storage_capex_low"] = dict_store_capex_low
        model_data["storage_capex_high"] = dict_store_capex_high
        model_data["storage_opex"] = dict_store_opex

    return model_data


model_data = prepare_model_data(load_data())

print(model_data)
