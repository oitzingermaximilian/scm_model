import pyomo.environ as pyo
import pandas as pd


def extract_variable_data(var):
    """
    Extracts data from a single Pyomo variable and converts it into a pandas DataFrame.
    Handles both scalar variables (like total costs) and indexed variables.
    """
    # 1. Handle unindexed (scalar) variables like m.costs
    if not var.is_indexed():
        return pd.DataFrame({"Value": [pyo.value(var)]})

    # 2. Handle indexed variables
    records = []
    for index in var:
        # Check if the variable is multi-dimensional (returns a tuple) or 1D
        if isinstance(index, tuple):
            row = list(index)
        else:
            row = [index]

        # Extract the optimized value (use pyo.value to avoid Pyomo object errors)
        try:
            val = pyo.value(var[index])
        except ValueError:
            val = None  # In case a variable wasn't initialized or solved

        row.append(val)
        records.append(row)

    # Generate generic column names based on the variable's dimensions
    dim = var.dim()
    cols = [f"Index_{i + 1}" for i in range(dim)] + ["Value"]

    df = pd.DataFrame(records, columns=cols)
    return df


def export_results(m, output_filename="SCM_RESULTS.xlsx"):
    """
    Iterates through all active variables in the model and writes them to an Excel file.
    """
    print(f"Exporting results to {output_filename}...")

    # Use pandas ExcelWriter to write multiple sheets
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        # Iterate over all Variable objects in the Pyomo model
        for var in m.component_objects(pyo.Var, active=True):
            var_name = var.name
            df = extract_variable_data(var)

            # Excel has a strict 31-character limit for sheet names
            sheet_name = var_name[:31]

            # Write the DataFrame to its own sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("--- Export Complete ---")
