import pandas as pd

file_path = "NG_flows_scm.xlsx"
demand = pd.read_excel(file_path, sheet_name="demand")
supply_share = pd.read_excel(file_path, sheet_name="supply_share")

merged = pd.merge(demand, supply_share, on="region_j", how="inner")
merged["ng_volume"] = merged["ng_volume"] * merged["supply_fraction"]
merged = merged.rename(columns={"sector_s": "sector_i"})

template_data = []
sectors = ["Power", "Industry", "Transport", "Buildings"]
years = [2026, 2030, 2035, 2040, 2045, 2050]

pairs = [
    ("NO", "UK"),
    ("EU27", "UK"),
    ("EU27", "NO"),
    ("UK", "NO"),
    ("UK", "EU27"),
    ("NO", "EU27"),
    ("NO", "NO"),
    ("EU27", "EU27"),
    ("UK", "UK"),
]

for sector in sectors:
  for year in years:
    for region_i, region_j in pairs:
      template_data.append({
          "year": year,
          "region_i": region_i,
          "region_j": region_j,
          "sector_i": sector,
      })

template_df = pd.DataFrame(template_data)

output_df = pd.merge(
    template_df,
    merged[["year", "region_i", "region_j", "sector_i", "ng_volume"]],
    on=["year", "region_i", "region_j", "sector_i"],
    how="left",
)

output_df["ng_volume"] = output_df["ng_volume"].fillna(0)
output_df.to_excel("NG_flows_scm_output.xlsx", index=False)
print("Updated script executed successfully. Output saved to NG_flows_scm_output.xlsx")