from datetime import datetime
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo


def extract_variable_data(var, scenario_label=None):
  """Extracts data from a single Pyomo variable and converts it into a pandas DataFrame.

  Handles both scalar variables and indexed variables, optionally adding a
  scenario label column.
  """
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


def export_results(m, output_filename="SCM_RESULTS.xlsx", scenario_label=None):
  """Iterates through all active variables in the model and writes them

  to an Excel file inside a timestamped directory, supporting the scenario label.
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
    for var in m.component_objects(pyo.Var, active=True):
      var_name = var.name
      df = extract_variable_data(var, scenario_label=scenario_label)
      sheet_name = var_name[:31]  # Excel 31-character limit
      df.to_excel(writer, sheet_name=sheet_name, index=False)

  print(f"--- Export Complete --- Saved in folder: {output_dir}")