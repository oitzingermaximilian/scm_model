import os
import pandas as pd


def load_data():
    """Reads the Excel file and returns a dictionary of raw DataFrames."""
    file_path = (
        r"C:\Users\maxoi\OneDrive\Desktop\Repositories\scm_model\scm_input_cleaned.xlsx"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_excel(file_path, sheet_name=None)


def prepare_model_data(raw_data):
    """Converts raw DataFrames into Pyomo-ready lists and dictionaries."""
    model_data = {}

    # ==========================================
    # 1. Process "timeseries" Sheet
    # ==========================================
    df_ts = raw_data["timeseries"].set_index("year")

    df_ts.index = pd.to_numeric(df_ts.index, errors="coerce")
    df_ts = df_ts[df_ts.index.notna()]
    df_ts.index = df_ts.index.astype(int)

    model_data["y"] = df_ts.index.tolist()

    model_data["g"] = {y: round(float(df_ts.at[y, "g"]), 3) for y in model_data["y"]}
    model_data["psi"] = {
        y: round(float(df_ts.at[y, "psi"]), 3) for y in model_data["y"]
    }

    if "ets_SCENARIO" not in df_ts.columns:
        df_ts["ets_SCENARIO"] = df_ts["c_ets_mid"]
    model_data["c_ETS"] = {
        y: round(float(df_ts.at[y, "ets_SCENARIO"]), 3) for y in model_data["y"]
    }

    # --- Expanded Sector Set ---
    model_data["sector"] = [
        "Industry_cement",
        "Industry_iron_steel",
        "Industry_chemicals",
        "Industry_refining",
        "Power",
        "Building",
        "Transport",
    ]

    rho_mapping = {
        "rho_refining": "Industry_refining",
        "rho_chemicals": "Industry_chemicals",
        "rho_transport": "Transport",
        "rho_cement": "Industry_cement",
        "rho_iron_steel": "Industry_iron_steel",
        "rho_power": "Power",
        "rho_buildings": "Building",
    }

    model_data["region"] = ["NO", "UK", "EU27"]

    rho_dict = {}
    for excel_col, sector_name in rho_mapping.items():
        if excel_col in df_ts.columns:
            for year in model_data["y"]:
                val = df_ts.at[year, excel_col]
                rho_dict[(sector_name, year)] = round(float(val), 3)

    model_data["rho"] = rho_dict

    # ==========================================
    # 2. Process "economic_cc_tech" Sheet
    # ==========================================
    df_psc = raw_data["economic_cc_tech"]

    allowed_techs = [
        "psc_cement",
        "psc_iron_steel",
        "psc_chemicals",
        "psc_refining",
        "psc_power",
    ]

    df_psc["technology_k"] = df_psc["technology_k"].astype(str).str.strip()
    df_psc["sector_i"] = df_psc["sector_i"].astype(str).str.strip()
    df_psc = df_psc[df_psc["technology_k"].isin(allowed_techs)]

    tech_list = df_psc["technology_k"].tolist()
    model_data["tech"] = tech_list
    model_data["tech_point"] = [t for t in tech_list if t.startswith("psc")]
    model_data["tech_rem"] = []

    dict_capex = {}
    dict_opex = {}
    dict_mu = {}

    known_years = sorted(
        [int(c.split("_")[1]) for c in df_psc.columns if c.startswith("capex_")]
    )

    for _, row in df_psc.iterrows():
        k = row["technology_k"]
        s = row["sector_i"]

        eff_avg_base = float(row["eff_avg"]) / 100.0
        dict_mu[(s, k)] = eff_avg_base

        for t in model_data["y"]:
            past_years = [y for y in known_years if y <= t]
            app_year = max(past_years) if past_years else min(known_years)

            dict_capex[(s, k, t)] = round(float(row[f"capex_{app_year}"]), 3)
            dict_opex[(s, k, t)] = round(float(row[f"opex_{app_year}"]), 3)

    model_data["c_cap_capex"] = dict_capex
    model_data["c_cap_opex"] = dict_opex
    model_data["mu"] = dict_mu

    # ==========================================
    # 3. Process "sink_blocks" Sheet
    # ==========================================
    df_sinks = raw_data["sink_blocks"]

    df_sinks["block_id"] = df_sinks["block_id"].astype(str).str.strip()
    df_sinks["region_i"] = df_sinks["region_i"].astype(str).str.strip()
    df_sinks["storage_type"] = df_sinks["storage_type"].astype(str).str.strip()

    for col in ["max_injection_yr", "max_capacity"]:
        df_sinks[col] = df_sinks[col].astype(str).str.replace(" ", "").astype(float)

    model_data["sink_blocks"] = df_sinks["block_id"].unique().tolist()
    model_data["storage_types"] = df_sinks["storage_type"].unique().tolist()

    dict_sink_timedelay = {}
    dict_sink_inj_rate = {}
    dict_sink_cap = {}
    dict_sink_capex = {}
    dict_sink_opex = {}
    dict_sink_type_map = {}

    known_sink_years = sorted(
        [int(c.split("_")[1]) for c in df_sinks.columns if c.startswith("capex_")]
    )

    for _, row in df_sinks.iterrows():
        b = row["block_id"]
        i = row["region_i"]

        dict_sink_type_map[(i, b)] = row["storage_type"]
        dict_sink_timedelay[(i, b)] = int(row["timedelay_td"])
        dict_sink_inj_rate[(i, b)] = row["max_injection_yr"]
        dict_sink_cap[(i, b)] = row["max_capacity"]

        for t in model_data["y"]:
            past_years = [y for y in known_sink_years if y <= t]
            app_year = max(past_years) if past_years else min(known_sink_years)

            unit_capex = float(row[f"capex_{app_year}"])
            unit_opex = float(row[f"opex_{app_year}"])

            dict_sink_capex[(i, b, t)] = round(unit_capex * row["max_capacity"], 3)
            dict_sink_opex[(i, b, t)] = round(unit_opex, 3)

    model_data["sink_timedelay"] = dict_sink_timedelay
    model_data["sink_injection_rate"] = dict_sink_inj_rate
    model_data["sink_block_cap"] = dict_sink_cap
    model_data["c_sink_capex"] = dict_sink_capex
    model_data["c_sink_opex"] = dict_sink_opex
    model_data["sink_type_map"] = dict_sink_type_map

    # ==========================================
    # 4. Process "ng_flows" Sheet (with Disaggregation)
    # ==========================================
    df_ng = raw_data["ng_flows"]

    df_ng["region_i"] = df_ng["region_i"].astype(str).str.strip()
    df_ng["region_j"] = df_ng["region_j"].astype(str).str.strip()
    df_ng["sector_i"] = df_ng["sector_i"].astype(str).str.strip()
    df_ng["year"] = df_ng["year"].astype(int)
    df_ng["ng_volume"] = pd.to_numeric(df_ng["ng_volume"], errors="coerce").fillna(0.0)

    # Industrial sub-sector split weights
    industry_splits = { #TODO THIS NEED TO BE CHECKED AND CHANGED TO A TIMESERIES FOR EVERY NODE
        "Industry_cement": 0.15,
        "Industry_iron_steel": 0.25,
        "Industry_chemicals": 0.30,
        "Industry_refining": 0.30,
    }

    raw_ng_lookup = {}
    for _, row in df_ng.iterrows():
        i = row["region_i"]
        j = row["region_j"]
        sec = row["sector_i"]
        yr = int(row["year"])
        vol = float(row["ng_volume"])

        if sec.lower() == "industry":
            for sub_sec, frac in industry_splits.items():
                raw_ng_lookup[(i, j, sub_sec, yr)] = (
                    raw_ng_lookup.get((i, j, sub_sec, yr), 0.0) + vol * frac
                )
        else:
            raw_ng_lookup[(i, j, sec, yr)] = (
                raw_ng_lookup.get((i, j, sec, yr), 0.0) + vol
            )

    known_ng_years = sorted(df_ng["year"].unique())
    dict_f_ng_intern = {}
    dict_f_ng_import = {}

    for i in model_data["region"]:
        for j in model_data["region"]:
            for s in model_data["sector"]:
                for t in model_data["y"]:
                    past_years = [y for y in known_ng_years if y <= t]
                    app_year = max(past_years) if past_years else min(known_ng_years)

                    vol = raw_ng_lookup.get((i, j, s, app_year), 0.0)

                    if i != j:
                        dict_f_ng_intern[(i, j, s, t)] = round(vol, 3)
                    else:
                        dict_f_ng_import[(i, s, t)] = round(vol, 3)

    model_data["f_NG_intern"] = dict_f_ng_intern
    model_data["f_NG_import"] = dict_f_ng_import

    # ==========================================
    # 5. Process "economic_transport" & Distances
    # ==========================================
    df_trans = raw_data["economic_transport"]
    df_trans["type"] = df_trans["type"].astype(str).str.strip()

    trans_data = {}
    for _, row in df_trans.iterrows():
        trans_data[row["type"]] = float(row["c_trans_avg"])

    onshore_cost = trans_data.get("onshore_pipeline", 0.07)
    offshore_cost = trans_data.get("offshore_pipeline", 0.0345)

    df_dist = raw_data["distances"]
    df_dist["region_i"] = df_dist["region_i"].astype(str).str.strip()
    df_dist["region_j"] = df_dist["region_j"].astype(str).str.strip()
    df_dist["value"] = pd.to_numeric(df_dist["value"], errors="coerce").fillna(0.0)

    dist_lookup = {}
    for _, row in df_dist.iterrows():
        dist_lookup[(row["region_i"], row["region_j"])] = row["value"]

    dict_c_trans = {}
    for i in model_data["region"]:
        for j in model_data["region"]:
            if i != j:
                d_km = dist_lookup.get((i, j), 0.0)
                dict_c_trans[(i, j)] = round(onshore_cost * d_km, 3)

    model_data["c_trans"] = dict_c_trans

    for _, row in df_sinks.iterrows():
        b = row["block_id"]
        i = row["region_i"]
        storage_type = row["storage_type"].lower()

        avg_sink_dist = dist_lookup.get((i, i), 0.0)
        if "offshore" in storage_type:
            transport_tariff = offshore_cost * avg_sink_dist
        else:
            transport_tariff = onshore_cost * avg_sink_dist

        for t in model_data["y"]:
            past_years = [y for y in known_sink_years if y <= t]
            app_year = max(past_years) if past_years else min(known_sink_years)

            base_opex = float(row[f"opex_{app_year}"])
            dict_sink_opex[(i, b, t)] = round(base_opex + transport_tariff, 3)

    model_data["c_sink_opex"] = dict_sink_opex

    # ==========================================
    # 6. Process "cc_tech" Sheet for Expansion Limits
    # ==========================================
    df_tech = raw_data["cc_tech"]

    for col in ["region_i", "sector_s", "technology_k"]:
        df_tech[col] = df_tech[col].astype(str).str.strip()

    for col in ["cap_init", "delta_cap", "ir_cap"]:
        if df_tech[col].dtype == object:
            df_tech[col] = df_tech[col].astype(str).str.replace(",", "", regex=False)
        df_tech[col] = pd.to_numeric(df_tech[col], errors="coerce").fillna(0.0)

    dict_q_init = {}
    dict_q_delta_abs = {}
    dict_q_ir_cap = {}

    for _, row in df_tech.iterrows():
        key = (row["region_i"], row["sector_s"], row["technology_k"])
        dict_q_init[key] = float(row["cap_init"])
        dict_q_delta_abs[key] = float(row["delta_cap"])
        dict_q_ir_cap[key] = float(row["ir_cap"])

    model_data["Q_init"] = dict_q_init
    model_data["Q_delta_abs"] = dict_q_delta_abs
    model_data["Q_IR_cap"] = dict_q_ir_cap

    return model_data


def export_model_data_to_excel(model_data, output_path="model_verification_output.xlsx"):
    """Exports processed model_data dictionaries into a readable Excel workbook."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # 1. Timeseries / Global parameters
        ts_list = [{"year": y, "g": model_data["g"][y], "psi": model_data["psi"][y], "c_ETS": model_data["c_ETS"][y]}
                   for y in model_data["y"]]
        pd.DataFrame(ts_list).to_excel(writer, sheet_name="timeseries", index=False)

        # 2. Rho parameters
        rho_list = [{"sector": s, "year": y, "rho": v} for (s, y), v in model_data["rho"].items()]
        pd.DataFrame(rho_list).to_excel(writer, sheet_name="rho", index=False)

        # 3. Technology Costs & Efficiency
        tech_list = []
        for (s, k, t), val in model_data["c_cap_capex"].items():
            tech_list.append({
                "sector": s,
                "technology": k,
                "year": t,
                "capex": val,
                "opex": model_data["c_cap_opex"].get((s, k, t)),
                "mu_efficiency": model_data["mu"].get((s, k))
            })
        pd.DataFrame(tech_list).to_excel(writer, sheet_name="tech_costs", index=False)

        # 4. Sinks Economic & Physical
        sink_list = []
        for (i, b, t), opex_val in model_data["c_sink_opex"].items():
            sink_list.append({
                "region": i,
                "block_id": b,
                "year": t,
                "storage_type": model_data["sink_type_map"].get((i, b)),
                "timedelay_td": model_data["sink_timedelay"].get((i, b)),
                "max_injection_yr": model_data["sink_injection_rate"].get((i, b)),
                "max_capacity": model_data["sink_block_cap"].get((i, b)),
                "capex": model_data["c_sink_capex"].get((i, b, t)),
                "opex_with_transport": opex_val
            })
        pd.DataFrame(sink_list).to_excel(writer, sheet_name="sinks", index=False)

        # 5. NG Flows Inter-Region Trade (i != j)
        ng_trade_list = [{"region_i": i, "region_j": j, "sector": s, "year": t, "volume": v} for (i, j, s, t), v in
                         model_data["f_NG_intern"].items()]
        pd.DataFrame(ng_trade_list).to_excel(writer, sheet_name="ng_trade", index=False)

        # 6. NG Flows Domestic/Imports (i == j)
        ng_import_list = [{"region_i": i, "sector": s, "year": t, "volume": v} for (i, s, t), v in
                          model_data["f_NG_import"].items()]
        pd.DataFrame(ng_import_list).to_excel(writer, sheet_name="ng_domestic_imports", index=False)

        # 7. Transport Costs
        trans_list = [{"region_i": i, "region_j": j, "c_trans": v} for (i, j), v in model_data["c_trans"].items()]
        pd.DataFrame(trans_list).to_excel(writer, sheet_name="transport_costs", index=False)

        # 8. Capacity Expansion Limits (cc_tech processed)
        exp_list = []
        for key in model_data["Q_init"].keys():
            i, s, k = key
            exp_list.append({
                "region": i,
                "sector": s,
                "technology": k,
                "cap_init": model_data["Q_init"].get(key),
                "delta_cap": model_data["Q_delta_abs"].get(key),
                "ir_cap": model_data["Q_IR_cap"].get(key)
            })
        pd.DataFrame(exp_list).to_excel(writer, sheet_name="cc_expansion_limits", index=False)

    print(f"Successfully exported verified model data to: {output_path}")

# Execution
raw = load_data()
model_data = prepare_model_data(raw)
export_model_data_to_excel(model_data)