from datetime import datetime
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo


def extract_variable_data(var, scenario_label=None):
    """Extracts data from a single Pyomo variable and converts it into a pandas DataFrame."""
    if not var.is_indexed():
        df = pd.DataFrame({"Value": [pyo.value(var)]})
    else:
        records = []
        for index in var:
            row = list(index) if isinstance(index, tuple) else [index]
            try:
                val = pyo.value(var[index])
            except ValueError:
                val = None
            row.append(val)
            records.append(row)

        dim = var.dim()
        cols = [f"Index_{i + 1}" for i in range(dim)] + ["Value"]
        df = pd.DataFrame(records, columns=cols)

    if scenario_label is not None:
        df.insert(0, "Scenario", scenario_label)

    return df


def extract_constraint_duals(m, constraint, scenario_label=None, undiscount_rate=None):
    """Extracts shadow prices (duals) from a Pyomo constraint and converts to a DataFrame.
  Optionally reverses the NPV discounting to yield Nominal prices.
  """
    records = []
    base_year = m.y.first() if undiscount_rate is not None else None

    # Ensure duals exist
    if not hasattr(m, "dual"):
        return pd.DataFrame()

    if not constraint.is_indexed():
        raw_dual = m.dual.get(constraint, None)
        records.append([raw_dual])
        cols = ["Raw_Dual_NPV"]
    else:
        dim = constraint.dim()
        cols = [f"Index_{i + 1}" for i in range(dim)] + ["Raw_Dual_NPV"]

        # Add column for real-world prices if we are undiscounting
        if undiscount_rate is not None:
            cols.append("Nominal_Price")

        for index in constraint:
            row = list(index) if isinstance(index, tuple) else [index]
            c_obj = constraint[index]

            # Fetch the dual value safely
            raw_dual = m.dual.get(c_obj, None)
            row.append(raw_dual)

            # If we need to undiscount (e.g., for Market Prices)
            if undiscount_rate is not None and raw_dual is not None:
                # Assume the time index 't' is the last index in the constraint
                t = index[-1] if isinstance(index, tuple) else index
                try:
                    nominal = abs(raw_dual) * ((1 + undiscount_rate) ** (t - base_year))
                    row.append(nominal)
                except Exception:
                    row.append(None)
            elif undiscount_rate is not None:
                row.append(None)

            records.append(row)

    df = pd.DataFrame(records, columns=cols)
    if scenario_label is not None:
        df.insert(0, "Scenario", scenario_label)
    return df


def export_results(m, output_filename="SCM_RESULTS.xlsx", scenario_label=None, discount_rate=0.05):
    """Iterates through active variables and selected constraints, writing them
  to an Excel file inside a timestamped directory.
  """
    path = Path(output_filename)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if path.suffix.lower() == ".xlsx":
        parent_dir = path.parent if path.parent != Path("") else Path(".")
        output_dir = parent_dir / f"{path.stem}_{timestamp}"
        final_filepath = output_dir / "model_results.xlsx"
    else:
        output_dir = Path(f"{output_filename}_{timestamp}")
        final_filepath = output_dir / "model_results.xlsx"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting results to {final_filepath}...")

    with pd.ExcelWriter(final_filepath, engine="openpyxl") as writer:

        # 1. Export all Variables
        for var in m.component_objects(pyo.Var, active=True):
            var_name = var.name
            df = extract_variable_data(var, scenario_label=scenario_label)
            sheet_name = var_name[:31]  # Excel 31-character limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 2. Export specific Duals (Shadow Prices)
        if hasattr(m, "dual"):
            if hasattr(m, "csu_market_clearing"):
                df_prices = extract_constraint_duals(
                    m,
                    m.csu_market_clearing,
                    scenario_label=scenario_label,
                    undiscount_rate=discount_rate
                )
                if not df_prices.empty:
                    # Rename columns to match what the plotting script expects ('region', 'year', 'value')
                    # Assumes Index_1 is Region and Index_2 is Year
                    rename_dict = {}
                    if "Index_2" in df_prices.columns:
                        rename_dict.update({"Index_1": "region", "Index_2": "year"})
                    elif "Index_1" in df_prices.columns:
                        rename_dict.update({"Index_1": "year"})

                    if "Nominal_Price" in df_prices.columns:
                        rename_dict["Nominal_Price"] = "value"
                    else:
                        rename_dict["Raw_Dual_NPV"] = "value"

                    df_dual_csu = df_prices.rename(columns=rename_dict)

                    # Filter down to just the columns we need for plotting
                    valid_cols = [c for c in ["Scenario", "region", "year", "value"] if c in df_dual_csu.columns]
                    df_dual_csu = df_dual_csu[valid_cols]

                    # Output to the sheet name the plotting script is looking for
                    df_dual_csu.to_excel(writer, sheet_name="dual_csu", index=False)

                    # Keep the original detailed output as well just in case
                    df_prices.to_excel(writer, sheet_name="CSU_Market_Prices", index=False)

    print(f"--- Export Complete --- Saved in folder: {output_dir}")