import os
import argparse
import pyomo.environ as pyo

# 1. FIXED: Imported prepare_model_data
from input import load_data, prepare_model_data
from initialization import build_base_model
from model import apply_constraints
from results import export_results
import scenarios

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
    # 2. FIXED: Pass model_data instead of raw_data
    base_model = build_base_model(model_data)
    final_model = apply_constraints(base_model)

    print("Solving model with Gurobi...")
    solver = pyo.SolverFactory("gurobi")
    results = solver.solve(final_model, tee=True)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        result_dir = os.path.join("results", scenario_name)
        os.makedirs(result_dir, exist_ok=True)
        out_path = os.path.join(result_dir, f"SCM_RESULTS_{scenario_name}.xlsx")
        export_results(final_model, output_filename=out_path)
        print(f"SUCCESS: Results saved to {out_path}")
    else:
        print(f"FAILURE: Optimal solution not found for {scenario_name}.")


if __name__ == "__main__":
    scenarios_registry = {
        "base": None,
        "high_ets": scenarios.scenario_high_ets,
        "low_quota": scenarios.scenario_low_ets,
    }

    if args.scenario == "all":
        for name, func in scenarios_registry.items():
            execute_scenario(name, func)
    else:
        func = scenarios_registry.get(args.scenario, None)
        if func is not None or args.scenario == "base":
            execute_scenario(args.scenario, func)
        else:
            print(f"Error: Unknown scenario '{args.scenario}' specified.")

print("\nSimulation process finished.")
