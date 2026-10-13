import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# Setup & Aesthetics
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 12,
})

# Colors for the Value Chain (Colorblind-friendly)
C_CAP = '#8A7E94'  # Muted Plum (Capture)
C_STO = '#D4A373'  # Warm Clay (Storage)
C_TRA = '#457B9D'  # Steel Blue (Transport)

# Transport Cost Rates (EUR/tCO2 per km)
RATE_INTER_NODE = 0.07
RATE_INTRA_NODE = 0.0345

TARGET_REGIONS = ["EU27", "NO", "UK"]

# Hardcoded Distances
DISTANCES = {
    ("EU27", "UK"): 1200,
    ("EU27", "NO"): 1300,
    ("NO", "UK"): 1100,
    ("NO", "EU27"): 1300,
    ("UK", "NO"): 1100,
    ("UK", "EU27"): 1200,
    ("EU27", "EU27"): 818,
    ("NO", "NO"): 250,
    ("UK", "UK"): 297
}


# ==========================================
# Data Loading & Processing
# ==========================================
def load_lcos_data(file_path="scm_input_cleaned.xlsx"):
    """Loads and sanitizes the required sheets from the Excel file."""
    sinks = pd.read_excel(file_path, sheet_name="sink_blocks")
    capture = pd.read_excel(file_path, sheet_name="economic_cc_tech")

    for df in [sinks, capture]:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

    return sinks, capture


def get_transport_cost_hardcoded(origin, host):
    """Calculates transport cost from origin to host sink."""
    dist_internal = DISTANCES.get((host, host), 0.0)
    cost_internal = dist_internal * RATE_INTRA_NODE

    if origin != host:
        dist_cross = DISTANCES.get((origin, host), DISTANCES.get((host, origin), 0.0))
        cost_cross = dist_cross * RATE_INTER_NODE
        return cost_cross + cost_internal
    else:
        return cost_internal


def calculate_lcos_database(sinks, capture, year=2026):
    """Creates a master dataframe with the LCOS for every valid pathway."""
    records = []

    columns = [
        "Host_Region", "Origin_Region", "Sector", "Technology",
        "Storage_Type", "Cost_Capture", "Cost_Transport", "Cost_Storage", "Total_LCOS"
    ]

    # --- FINANCIAL ASSUMPTIONS ---
    DISCOUNT_RATE = 0.075  # 7.5% Discount Rate
    MODEL_END_YEAR = 2050
    LIFETIME_YEARS = max(1, MODEL_END_YEAR - year)

    # Capital Recovery Factor (CRF)
    crf = (DISCOUNT_RATE * (1 + DISCOUNT_RATE) ** LIFETIME_YEARS) / ((1 + DISCOUNT_RATE) ** LIFETIME_YEARS - 1)

    if f"capex_{year}" not in sinks.columns or f"opex_{year}" not in sinks.columns:
        print(f"Warning: capex_{year} or opex_{year} missing in sink_blocks sheet.")
        return pd.DataFrame(columns=columns)

    # Safely convert to numeric
    sinks[f"capex_{year}"] = pd.to_numeric(sinks[f"capex_{year}"], errors='coerce')
    sinks[f"opex_{year}"] = pd.to_numeric(sinks[f"opex_{year}"], errors='coerce')

    sink_costs = (
        sinks.groupby(["region_i", "storage_type"])[
            [f"capex_{year}", f"opex_{year}"]
        ]
        .mean()
        .reset_index()
    )

    for host in TARGET_REGIONS:
        host_sinks = sink_costs[sink_costs["region_i"] == host]
        if host_sinks.empty:
            continue

        for _, sink_row in host_sinks.iterrows():
            storage_type = sink_row["storage_type"]

            # ==========================================
            # --- FIX: PARTNER STORAGE COST LOGIC ---
            # ==========================================
            unit_capex = sink_row[f"capex_{year}"]
            LIFETIME_PARTNER = 40
            capex_partner_logic = unit_capex * LIFETIME_PARTNER
            annualized_storage_capex = capex_partner_logic * crf
            c_storage = annualized_storage_capex + sink_row[f"opex_{year}"]
            # ==========================================

            for origin in TARGET_REGIONS:
                c_trans = get_transport_cost_hardcoded(origin, host)

                origin_caps = capture[(capture["region"] == origin) | (capture["region"] == "All")]
                if origin_caps.empty:
                    continue

                for _, cap_row in origin_caps.iterrows():
                    sector = cap_row["sector_i"]
                    tech = cap_row["technology_k"]

                    c_cap_capex = pd.to_numeric(cap_row[f"capex_{year}"], errors='coerce')
                    c_cap_opex_fix = pd.to_numeric(cap_row[f"opex_fix_{year}"], errors='coerce')
                    c_cap_opex_var = pd.to_numeric(cap_row[f"opex_var_{year}"], errors='coerce')

                    annualized_capture_capex = c_cap_capex * crf
                    c_capture = annualized_capture_capex + c_cap_opex_fix + c_cap_opex_var

                    records.append({
                        "Host_Region": host,
                        "Origin_Region": origin,
                        "Sector": sector,
                        "Technology": tech,
                        "Storage_Type": storage_type,
                        "Cost_Capture": c_capture,
                        "Cost_Transport": c_trans,
                        "Cost_Storage": c_storage,
                        "Total_LCOS": c_capture + c_trans + c_storage,
                    })

    return pd.DataFrame(records, columns=columns)


# ==========================================
# Plotting Functions
# ==========================================
def plot_lcos_grid(df_lcos, host_region, year=2026, output_dir="figures_lcos"):
    """Creates a grid of subplots for a country (Rows = Sink Types, Cols = Sectors)."""
    os.makedirs(output_dir, exist_ok=True)

    df_host = df_lcos[df_lcos["Host_Region"] == host_region].copy()
    if df_host.empty:
        print(f"No data for Host: {host_region}")
        return

    # Extract unique sectors and sink types to build the grid
    sectors = sorted(df_host["Sector"].unique())
    sink_types = sorted(df_host["Storage_Type"].unique())

    nrows = len(sink_types)
    ncols = len(sectors)

    # Dynamic figure size based on grid dimensions
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3.5 * nrows), sharey=True)

    # Ensure axes is a 2D array even if it's a 1xN or Nx1 grid
    axes = np.atleast_2d(axes)
    if nrows == 1 and ncols > 1:
        axes = axes.T if axes.shape[0] != 1 else axes  # handle 1D edge cases gracefully

    max_lcos = df_host["Total_LCOS"].max()

    for r, s_type in enumerate(sink_types):
        for c, sec in enumerate(sectors):
            # Safe indexing for 2D axes
            ax = axes[r, c] if nrows > 1 else (axes[0, c] if ncols > 1 else axes[0, 0])

            df_plot = df_host[(df_host["Storage_Type"] == s_type) & (df_host["Sector"] == sec)]

            if df_plot.empty:
                ax.axis('off')
                continue

            # Average costs across differing technologies to get 1 value per origin
            df_plot = df_plot.groupby("Origin_Region", as_index=False).agg({
                "Cost_Capture": "mean",
                "Cost_Storage": "mean",
                "Cost_Transport": "mean",
                "Total_LCOS": "mean"
            })

            # Sort origins cleanly
            df_plot["Origin_Region"] = pd.Categorical(df_plot["Origin_Region"], categories=TARGET_REGIONS, ordered=True)
            df_plot = df_plot.sort_values("Origin_Region")

            x = np.arange(len(df_plot))
            width = 0.55

            # Stacking logic
            bottom_storage = df_plot["Cost_Capture"]
            bottom_transport = bottom_storage + df_plot["Cost_Storage"]

            # Plot bars: Capture (Bottom) -> Storage (Middle) -> Transport (Top)
            ax.bar(x, df_plot["Cost_Capture"], width, color=C_CAP, edgecolor="black", linewidth=0.5, zorder=3)
            ax.bar(x, df_plot["Cost_Storage"], width, bottom=bottom_storage, color=C_STO, edgecolor="black",
                   linewidth=0.5, zorder=3)
            ax.bar(x, df_plot["Cost_Transport"], width, bottom=bottom_transport, color=C_TRA, edgecolor="black",
                   linewidth=0.5, zorder=3)

            # Labels on top
            for i, total in enumerate(df_plot["Total_LCOS"]):
                if not pd.isna(total):
                    ax.text(x[i], total + (max_lcos * 0.03), f"€{total:.0f}", ha="center", va="bottom", fontsize=9,
                            fontweight="bold")

            # X-axis ticks
            ax.set_xticks(x)
            ax.set_xticklabels(df_plot["Origin_Region"])
            ax.grid(True, linestyle=":", alpha=0.6, axis="y", zorder=0)

            # Add x-axis label only for the bottom row of the grid
            if r == nrows - 1:
                ax.set_xlabel("CO$_2$ Origin", fontweight="bold", labelpad=8)

            # Setup Titles (Columns = Sectors, Rows = Sink Types)
            if r == 0:
                # Custom logic for formatting Sector titles
                clean_sec = sec.replace("Industry_", "")
                if clean_sec == "iron_steel":
                    clean_sec = "Iron & Steel"
                else:
                    clean_sec = clean_sec.replace("_", " ").title()

                ax.set_title(clean_sec, pad=10, fontweight="bold")

            if c == 0:
                clean_sink = s_type.replace("_", " ").title()
                ax.set_ylabel(f"{clean_sink}\nCost (€/tCO$_2$)", labelpad=10, fontweight="bold")

    # Limit shared Y axis
    plt.ylim(0, max_lcos * 1.15)

    # Master Title has been removed per request
    # fig.suptitle(...)

    # Global Legend below the entire grid
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=C_CAP, edgecolor='black', label='Capture Cost'),
        mpatches.Patch(facecolor=C_STO, edgecolor='black', label='Storage Cost'),
        mpatches.Patch(facecolor=C_TRA, edgecolor='black', label='Transport Cost')
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=True, edgecolor="black",
               fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"LCOS_Grid_{host_region}_{year}.pdf")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved LCOS grid plot to: {plot_path}")


# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    excel_file = "scm_input_cleaned.xlsx"

    print("Loading input data...")
    sinks, capture = load_lcos_data(excel_file)

    print("Calculating LCOS master database for 2026...")
    df_lcos_2026 = calculate_lcos_database(sinks, capture, year=2026)

    # Generate the comprehensive grid plot for each target region
    for region in TARGET_REGIONS:
        plot_lcos_grid(df_lcos_2026, host_region=region, year=2026)

    print("All LCOS grid plots generated successfully.")