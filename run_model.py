import os
import argparse
import pyomo.environ as pyo
import pandas as pd

from input import load_data, prepare_model_data
from initialization import build_base_model
from model import apply_constraints
from results import export_results

# NEW: Import the dynamic dictionary we created
from scenarios import SCENARIO_DICT

def analyze_model_lcos(m, discount_rate=0.08, default_lifetime=20):
    """Calculates the Levelized Cost of CCUS directly from the Pyomo model parameters

    and compares it against the ETS carbon price.
    """
    # ---------------------------------------------------------
    # 1. EXTRACT DATA DIRECTLY FROM MODEL INSTANCE
    # ---------------------------------------------------------
    c_cap_capex = m.c_cap_capex.extract_values()
    c_cap_opex = m.c_cap_opex.extract_values()
    c_trans = m.c_trans.extract_values()
    c_sink_capex = m.c_sink_capex.extract_values()
    c_inj_opex = m.c_inj_opex.extract_values()
    sink_cap = m.sink_block_cap.extract_values()
    sink_inj_rate = m.sink_injection_rate.extract_values()
    c_ets = m.c_ETS.extract_values()

    # Capture efficiency (eta) if present
    eta = pyo.value(m.eta) if hasattr(m, "eta") else 1.0

    base_year = m.y.first()
    crf = (discount_rate * (1 + discount_rate) ** default_lifetime) / (
        (1 + discount_rate) ** default_lifetime - 1
    )

    print("=" * 80)
    print(
        f"CCUS LEVELIZED COST DIAGNOSTIC (Discount Rate: {discount_rate*100:.1f}%, Lifetime: {default_lifetime} yrs, CRF: {crf:.4f})"
    )
    print("=" * 80)

    # ---------------------------------------------------------
    # 2. LEVELIZED CAPTURE COST BY SECTOR & TECH
    # ---------------------------------------------------------
    capture_summary = {}
    print("\n[1] LEVELIZED CAPTURE COSTS (€/tCO2)")
    print(
        f"{'Sector':<22} {'Tech':<16} {'CAPEX(€/t)':<12} {'Ann.CAPEX':<12} {'OPEX(€/t)':<12} {'LCO-Capture'}"
    )
    print("-" * 80)

    for s in m.sector:
        for k in m.tech:
            # Look at base_year values
            capex = c_cap_capex.get((s, k, base_year), 0.0)
            # Take average or base_year opex
            opex_vals = [
                v
                for (sec, tech, vin), v in c_cap_opex.items()
                if sec == s and tech == k
            ]
            opex = opex_vals[0] if opex_vals else 0.0

            ann_capex = capex * crf
            lco_capture = ann_capex + opex
            capture_summary[(s, k)] = {
                "capex": capex,
                "ann_capex": ann_capex,
                "opex": opex,
                "lco_capture": lco_capture,
            }

            print(
                f"{s:<22} {k:<16} {capex:<12.1f} {ann_capex:<12.2f} {opex:<12.2f} €{lco_capture:.2f}/t"
            )

    # ---------------------------------------------------------
    # 3. LEVELIZED SINK COSTS BY BLOCK
    # ---------------------------------------------------------
    sink_summary = {}
    print("\n[2] LEVELIZED SINK COSTS (€/tCO2)")
    print(
        f"{'Region':<12} {'Block':<12} {'Block CAPEX(M€)':<16} {'Inj.Rate(Mt/y)':<16} {'Inj.OPEX':<10} {'LCO-Sink'}"
    )
    print("-" * 80)

    for (i, b), tot_cap in sink_cap.items():
        if tot_cap == 0:
            continue
        inj_rate = sink_inj_rate.get((i, b), 0.0)
        # Sinks have capex across years, take year 1 capex or first non-zero
        block_capex_list = [
            val
            for (reg, blk, yr), val in c_sink_capex.items()
            if reg == i and blk == b and val > 0
        ]
        block_capex = block_capex_list[0] if block_capex_list else 0.0

        inj_op_list = [
            val
            for (reg, blk, yr), val in c_inj_opex.items()
            if reg == i and blk == b
        ]
        inj_op = inj_op_list[0] if inj_op_list else 0.0

        # Effective lifetime of block = total capacity / annual injection rate
        effective_life = (
            min(default_lifetime, tot_cap / inj_rate)
            if inj_rate > 0
            else default_lifetime
        )
        crf_sink = (discount_rate * (1 + discount_rate) ** effective_life) / (
            (1 + discount_rate) ** effective_life - 1
        )

        # Levelized sink CAPEX per tonne = (Overnight CAPEX * CRF) / annual injection volume
        ann_sink_capex_per_t = (
            (block_capex * crf_sink) / inj_rate if inj_rate > 0 else 0.0
        )
        lco_sink = ann_sink_capex_per_t + inj_op
        sink_summary[(i, b)] = lco_sink

        print(
            f"{i:<12} {b:<12} {block_capex:<16.1f} {inj_rate:<16.2f} {inj_op:<10.2f} €{lco_sink:.2f}/t"
        )

    # ---------------------------------------------------------
    # 4. FULL CHAIN CCUS vs. ETS BENCHMARK
    # ---------------------------------------------------------
    print("\n[3] FULL VALUE CHAIN BREAKDOWN vs. ETS BENCHMARK (Base Year)")
    print("-" * 80)
    base_ets = c_ets.get(base_year, 80.0)
    print(f"--> Base Year ({base_year}) ETS Carbon Price: €{base_ets:.2f}/tCO2\n")

    full_chain_rows = []
    # Test specific key sink regions (e.g., NO, UK, or domestic EU)
    sink_regions = set(i for (i, b) in sink_summary.keys())

    for (s, k), cap_data in capture_summary.items():
        for s_reg in sink_regions:
            # Find the cheapest block in this sink region
            blocks_in_reg = [
                (b, cost)
                for (r, b), cost in sink_summary.items()
                if r == s_reg
            ]
            if not blocks_in_reg:
                continue
            best_block, best_sink_cost = min(blocks_in_reg, key=lambda x: x[1])

            # For each emitting region
            for e_reg in m.region:
                trans_cost = (
                    c_trans.get((e_reg, s_reg), 0.0) if e_reg != s_reg else 0.0
                )

                # Total CCUS Cost
                # Note: If 1/eta is applied in network balance, adjust sink/transport accordingly
                total_ccus = cap_data["lco_capture"] + trans_cost + (best_sink_cost / eta)
                diff = total_ccus - base_ets

                full_chain_rows.append(
                    {
                        "Sector": s,
                        "Origin": e_reg,
                        "Sink_Reg": s_reg,
                        "Capture": round(cap_data["lco_capture"], 1),
                        "Trans": round(trans_cost, 1),
                        "Sink": round(best_sink_cost, 1),
                        "Total_CCUS": round(total_ccus, 1),
                        "ETS": round(base_ets, 1),
                        "Diff": round(diff, 1),
                        "Model_Choice": "CCUS CHEAPER"
                        if diff < 0
                        else "ETS CHEAPER",
                    }
                )

    df_res = pd.DataFrame(full_chain_rows)

    # Show a concise preview of key combinations
    # Filter to show only unique sector-to-sink combinations
    preview = (
        df_res.groupby(["Sector", "Sink_Reg"])
        .agg(
            {
                "Capture": "mean",
                "Trans": "mean",
                "Sink": "mean",
                "Total_CCUS": "mean",
                "ETS": "first",
                "Diff": "mean",
                "Model_Choice": lambda x: x.iloc[0],
            }
        )
        .reset_index()
    )

    print(
        preview.to_string(
            index=False,
            columns=[
                "Sector",
                "Sink_Reg",
                "Capture",
                "Trans",
                "Sink",
                "Total_CCUS",
                "ETS",
                "Diff",
                "Model_Choice",
            ],
        )
    )

    return df_res

parser = argparse.ArgumentParser(description="Run SCM model scenarios using Gurobi.")
parser.add_argument(
    "--scenario",
    default="base",
    help='Scenario name to run (e.g., "base", "high_ets", "low_quota", or "all")',
)
args = parser.parse_args()


def execute_scenario(scenario_name, scenario_func):
    print(f"\n========================================")
    print(f"RUNNING SCENARIO: {scenario_name.upper()}")
    print(f"========================================")

    # Step 1: Load raw data
    raw_data = load_data()

    # Step 2: Apply scenario mutations (if any)
    if scenario_func is not None:
        raw_data = scenario_func(raw_data)

    # Step 3: Convert DataFrames to Pyomo-ready dictionaries
    model_data = prepare_model_data(raw_data)


    # Step 4: Build model
    base_model = build_base_model(model_data)
    final_model = apply_constraints(base_model)

    # =========================================================================
    # CALL DIAGNOSTIC FUNCTION HERE (Before solving)
    # =========================================================================
    #analyze_model_lcos(final_model, discount_rate=0.08, default_lifetime=20)
    # =========================================================================

    # Step 4: Build model
    base_model = build_base_model(model_data)
    final_model = apply_constraints(base_model)

    # =========================================================================
    # DEBUG OUTPUT: Check if scenario parameters applied correctly
    # =========================================================================
    print("\n--- DEBUG: CHECKING SCENARIO PARAMETERS ---")
    print("1. Parameter 'g' (CSU Obligation Rate):")
    final_model.g.pprint()

    print("\n2. Parameter 'c_ETS' (ETS Price):")
    final_model.c_ETS.pprint()
    print("-------------------------------------------\n")

    # Fetch duals (shadow prices)
    final_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    print("Solving model with Gurobi...")
    solver = pyo.SolverFactory("gurobi")
    solver.options['MIPGap'] = 1e-6

    # --- INITIAL MIP SOLVE ---
    results = solver.solve(final_model, tee=True)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("\nOptimal MIP found. Fixing binaries to extract dual variables...")

        # ==========================================
        # FIX-AND-RELAX PROCEDURE
        # ==========================================
        # 1. Fix all binary/integer variables to their optimal values
        for v in final_model.component_data_objects(ctype=pyo.Var):
            if v.domain in (pyo.Binary, pyo.Integers, pyo.NonNegativeIntegers):
                if v.value is not None:
                    v.fix(round(v.value))
                else:
                    v.fix(0)
                # Relax domain to Continuous so Gurobi treats it as an LP
                v.domain = pyo.Reals

        # 2. Re-solve the model as a pure LP
        solver.solve(final_model, tee=False)
        print("LP re-solve complete. Dual variables generated.")
        # ==========================================

        # Export results (now containing duals)
        result_dir = os.path.join("results", scenario_name)
        os.makedirs(result_dir, exist_ok=True)
        out_path = os.path.join(result_dir, f"SCM_RESULTS_{scenario_name}.xlsx")

        export_results(final_model, output_filename=out_path)
        print(f"SUCCESS: Results saved to {out_path}")
    else:
        print(f"FAILURE: Optimal solution not found for {scenario_name}.")


if __name__ == "__main__":
    # Use the dynamic dictionary from scenarios.py
    scenarios_registry = SCENARIO_DICT

    if args.scenario == "all":
        for name, func in scenarios_registry.items():
            execute_scenario(name, func)
    else:
        # Check if the requested scenario exists in our dictionary
        if args.scenario in scenarios_registry:
            func = scenarios_registry[args.scenario]
            execute_scenario(args.scenario, func)
        else:
            print(f"Error: Unknown scenario '{args.scenario}' specified.")
            print(f"Available scenarios are:")
            for s in scenarios_registry.keys():
                print(f"  - {s}")

print("\nSimulation process finished.")
