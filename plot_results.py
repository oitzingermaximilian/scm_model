import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scienceplots
from input import load_data, prepare_model_data

# ==========================================
# Setup matplotlib style
# ==========================================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "pgf.rcfonts": False,
})
try:
    plt.style.use(["science", "ieee", "no-latex"])
except:
    pass

_fontsize = 10
plt.rcParams["xtick.labelsize"] = _fontsize
plt.rcParams["ytick.labelsize"] = _fontsize
plt.rcParams["axes.labelsize"] = _fontsize
plt.rcParams["axes.titlesize"] = _fontsize + 2

RESULTS_PATH = "results/base/model_results.xlsx"
OUTPUT_DIR = "figures"

# ------------------------------------------
# Create Structured Directories
# ------------------------------------------
DIR_FLOWS = os.path.join(OUTPUT_DIR, "flows")
DIR_SOC = os.path.join(OUTPUT_DIR, "state_of_charge")
DIR_BALANCE = os.path.join(OUTPUT_DIR, "csu_balance")
DIR_ETS = os.path.join(OUTPUT_DIR, "ets_vs_injected")
DIR_MARKET = os.path.join(OUTPUT_DIR, "market_flows")
DIR_PRICE = os.path.join(OUTPUT_DIR, "shadow_prices")

for directory in [DIR_FLOWS, DIR_SOC, DIR_BALANCE, DIR_ETS, DIR_MARKET, DIR_PRICE]:
    os.makedirs(directory, exist_ok=True)

# 1. Load parameters for capacity (Converted to Megatons)
raw = load_data()
model_data = prepare_model_data(raw)
sink_cap = {k: v / 1000000.0 for k, v in model_data["sink_block_cap"].items()}

# 2. Load result variables
q_trans = pd.read_excel(RESULTS_PATH, sheet_name="q_CO2_trans")
csu_buy = pd.read_excel(RESULTS_PATH, sheet_name="CSU_buy")
csu_sell = pd.read_excel(RESULTS_PATH, sheet_name="CSU_sell")
csu_gen = pd.read_excel(RESULTS_PATH, sheet_name="CSU_generate")
csu_use = pd.read_excel(RESULTS_PATH, sheet_name="CSU_use")
q_inj = pd.read_excel(RESULTS_PATH, sheet_name="q_CO2_inj")
q_ets = pd.read_excel(RESULTS_PATH, sheet_name="q_CO2_ETS")


def rename_cols(df, cols):
    if "Scenario" in df.columns:
        df = df.drop(columns=["Scenario"])
    df.columns = cols
    return df


q_trans = rename_cols(q_trans, ["region_from", "region_to", "year", "value"])
csu_buy = rename_cols(csu_buy, ["region", "year", "value"])
csu_sell = rename_cols(csu_sell, ["region", "year", "value"])
csu_gen = rename_cols(csu_gen, ["region", "year", "value"])
csu_use = rename_cols(csu_use, ["region", "year", "value"])
q_inj = rename_cols(q_inj, ["region", "block", "year", "value"])
q_ets = rename_cols(q_ets, ["region", "sector", "year", "value"])

# ==========================================
# Apply Global Year Cutoff (<= 2050)
# ==========================================
MAX_YEAR = 2050

q_trans = q_trans[q_trans["year"] <= MAX_YEAR]
csu_buy = csu_buy[csu_buy["year"] <= MAX_YEAR]
csu_sell = csu_sell[csu_sell["year"] <= MAX_YEAR]
csu_gen = csu_gen[csu_gen["year"] <= MAX_YEAR]
csu_use = csu_use[csu_use["year"] <= MAX_YEAR]
q_inj = q_inj[q_inj["year"] <= MAX_YEAR]
q_ets = q_ets[q_ets["year"] <= MAX_YEAR]

# 3. Convert ALL quantities from tonnes to Megatons (Mt)
for df in [q_trans, q_inj, q_ets, csu_buy, csu_sell, csu_gen, csu_use]:
    df.loc[:, "value"] /= 1000000.0

# ==========================================
# Define Custom Node Palette
# ==========================================
custom_palette = ['#A9907E', '#F3DEBA', '#ABC4AA', '#675D50']

# Extract all unique regions to map them consistently to colors
all_regions = sorted(
    set(q_trans["region_from"].unique()) | set(q_trans["region_to"].unique()) | set(q_inj["region"].unique()))
region_color_map = {region: custom_palette[i % len(custom_palette)] for i, region in enumerate(all_regions)}

# Specific years to display on x-axis (Up to 2050)
target_years = [2025, 2030, 2035, 2040, 2045, 2050]


# ==========================================
# Helper for Periods
# ==========================================
def get_period(year):
    if 2026 <= year <= 2030:
        return "2026-2030"
    elif 2031 <= year <= 2035:
        return "2031-2035"
    elif 2036 <= year <= 2040:
        return "2036-2040"
    elif 2041 <= year <= 2045:
        return "2041-2045"
    elif 2046 <= year <= 2050:
        return "2046-2050"
    return None


# ==========================================
# Figure 1: Flows of CO2 between nodes and CSUs
# ==========================================
fig1, axes = plt.subplots(1, 2, figsize=(10, 4))

trans_agg = q_trans.groupby(["region_from", "region_to"])["value"].sum().reset_index()
trans_agg = trans_agg[trans_agg["region_from"] != trans_agg["region_to"]]
trans_pivot = trans_agg.pivot(index="region_from", columns="region_to", values="value").fillna(0)

bar_colors = [region_color_map[col] for col in trans_pivot.columns]
trans_pivot.plot(kind="bar", stacked=True, ax=axes[0], color=bar_colors, edgecolor='black', linewidth=0.5)
axes[0].set_title(f"Total Physical CO$_2$ Transport Between Nodes\n(Up to {MAX_YEAR})")
axes[0].set_ylabel("CO$_2$ (Mt)")
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelrotation=0)
axes[0].grid(True, linestyle=":", alpha=0.6, axis='y')

csu_net = csu_buy.groupby("region")["value"].sum() - csu_sell.groupby("region")["value"].sum()
net_colors = [region_color_map[r] for r in csu_net.index]
csu_net.plot(kind="bar", ax=axes[1], color=net_colors, edgecolor='black', linewidth=0.5)
axes[1].set_title(f"Net CSU Position (Bought - Sold) per Node\n(Up to {MAX_YEAR})")
axes[1].set_ylabel("CSU (Mt)")
axes[1].set_xlabel("")
axes[1].tick_params(axis='x', labelrotation=0)
axes[1].axhline(0, color='black', lw=1)
axes[1].grid(True, linestyle=":", alpha=0.6, axis='y')

plt.tight_layout()
fig1.savefig(f"{DIR_FLOWS}/Figure1_Flows.png", dpi=600, bbox_inches='tight')
plt.close(fig1)

# ==========================================
# Figure 2: State of Charge of Sinks
# ==========================================
q_inj_sorted = q_inj.sort_values(by=["region", "block", "year"])
q_inj_sorted["cum_inj"] = q_inj_sorted.groupby(["region", "block"])["value"].cumsum()

soc_agg = q_inj_sorted.groupby(["region", "year"])["cum_inj"].sum().reset_index()
unique_years_bar = sorted(soc_agg["year"].unique())  # Force uniform x-axis length

region_caps = {}
for (r, b), c in sink_cap.items():
    region_caps[r] = region_caps.get(r, 0) + c

plt.figure(figsize=(8, 4))
ax2 = plt.gca()

cap_df = soc_agg.copy()
cap_df["total_cap"] = cap_df["region"].map(region_caps)

sns.barplot(data=cap_df, x="year", y="total_cap", hue="region", palette=region_color_map, alpha=0.3, errorbar=None,
            edgecolor='none', order=unique_years_bar)
sns.barplot(data=soc_agg, x="year", y="cum_inj", hue="region", palette=region_color_map, alpha=1.0, errorbar=None,
            edgecolor='black', linewidth=0.5, order=unique_years_bar)

handles, labels = ax2.get_legend_handles_labels()
n = len(region_caps)
plt.legend(handles[n:2 * n], labels[:n], title="Region", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False,
           fontsize=_fontsize)

tick_pos = [unique_years_bar.index(y) for y in target_years if y in unique_years_bar]
tick_lab = [y for y in target_years if y in unique_years_bar]
ax2.set_xticks(tick_pos)
ax2.set_xticklabels(tick_lab)

plt.title("State of Charge of Sinks over Time\n(Total Available Capacity vs Cumulative Injection)")
plt.ylabel("CO$_2$ (Mt)")
plt.xlabel("")
plt.grid(True, linestyle=":", alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig(f"{DIR_SOC}/Figure2_SOC_Overall.png", dpi=600, bbox_inches='tight')
plt.close()

# ==========================================
# Figure 3: CSU Balance (Two Variants)
# ==========================================
try:
    regions = q_inj["region"].unique()
    bar_chart_data = {}  # Dictionary to store data for the combined Variant B chart

    for region in regions:
        # Generate clean aligned dataframes
        df_gen = csu_gen[csu_gen["region"] == region].groupby("year")["value"].sum()
        df_buy = csu_buy[csu_buy["region"] == region].groupby("year")["value"].sum()
        df_sell = csu_sell[csu_sell["region"] == region].groupby("year")["value"].sum()
        df_use = csu_use[csu_use["region"] == region].groupby("year")["value"].sum()

        df_bal = pd.DataFrame({"Gen": df_gen, "Buy": df_buy, "Sell": df_sell, "Use": df_use}).fillna(0)
        df_bal = df_bal.loc[df_bal.index <= MAX_YEAR]

        if df_bal.empty:
            continue

        years = df_bal.index.tolist()

        # ------------------------------------------
        # Variant A: Continuous Filled Area Chart
        # ------------------------------------------
        plt.figure(figsize=(7, 4))
        ax_area = plt.gca()

        ax_area.fill_between(years, 0, df_bal["Use"], color='#675D50', alpha=0.15, label='Use')
        ax_area.plot(years, df_bal["Use"], color='#675D50', linestyle='--', linewidth=1.5, label='Obligation')

        ax_area.stackplot(years, df_bal["Gen"], df_bal["Buy"], labels=['Generation', 'Buy'],
                          colors=['#ABC4AA', '#F3DEBA'], alpha=0.9, edgecolor='black', linewidth=0.5)
        ax_area.fill_between(years, 0, -df_bal["Sell"], color='#A9907E', alpha=0.9, label='Sell',
                             edgecolor='black', linewidth=0.5)

        ax_area.axhline(0, color='black', linewidth=1)
        ax_area.set_title(f"CSU Balance & Fulfillment: {region} (Area)")
        ax_area.set_ylabel("CSU (Mt)")
        ax_area.set_xlabel("")
        ax_area.set_xticks([y for y in target_years if y in years])
        ax_area.grid(True, linestyle=":", alpha=0.6, axis='y')

        # Formatting legend (Framed, single row below)
        handles, labels = ax_area.get_legend_handles_labels()
        desired_order = ['Obligation', 'Use', 'Generation', 'Buy', 'Sell']
        label_map = dict(zip(labels, handles))
        ordered_handles = [label_map[lbl] for lbl in desired_order if lbl in label_map]
        ordered_labels = [lbl for lbl in desired_order if lbl in label_map]

        ax_area.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=len(desired_order), frameon=True, edgecolor='black', fontsize=_fontsize)

        for spine in ax_area.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(f"{DIR_BALANCE}/Figure3_CSU_Balance_Area_{region}.png", dpi=600, bbox_inches='tight')
        plt.close()

        # ------------------------------------------
        # Prepare Data for Variant B (Bar Chart up to 2040)
        # ------------------------------------------
        df_bal_2040 = df_bal.loc[df_bal.index <= 2040].copy()
        df_bal_2040['Period'] = df_bal_2040.index.map(get_period)
        df_period = df_bal_2040.groupby('Period').sum()

        standard_periods_2040 = ['2026-2030', '2031-2035', '2036-2040']
        df_period = df_period.reindex(standard_periods_2040).fillna(0)

        bar_chart_data[region] = df_period

    # ------------------------------------------
    # Variant B: Cumulative Stacked Bar Chart (Combined) - Horizontal Red Obligation Line across Bars
    # ------------------------------------------
    if bar_chart_data:
        num_regions = len(bar_chart_data)
        fig_bar, axes_bar = plt.subplots(1, num_regions, figsize=(3.5 * num_regions, 4.5), sharey=True)

        if num_regions == 1:
            axes_bar = [axes_bar]

        for ax_bar, region in zip(axes_bar, bar_chart_data.keys()):
            df_period = bar_chart_data[region]
            bar_width = 0.45  # Narrow bars

            # Stacked bars for positive and negative values
            ax_bar.bar(df_period.index, df_period["Gen"], width=bar_width, color='#ABC4AA', label='Generation',
                       edgecolor='black', linewidth=0.5)
            ax_bar.bar(df_period.index, df_period["Buy"], bottom=df_period["Gen"], width=bar_width, color='#F3DEBA',
                       label='Buy', edgecolor='black', linewidth=0.5)
            ax_bar.bar(df_period.index, -df_period["Sell"], width=bar_width, color='#A9907E', label='Sell',
                       edgecolor='black', linewidth=0.5)

            # Draw obligation (Use) as red horizontal lines across the width of each bar
            for i, (period_name, row) in enumerate(df_period.iterrows()):
                use_val = row["Use"]
                ax_bar.hlines(y=use_val, xmin=i - bar_width / 2, xmax=i + bar_width / 2, color='red', linewidth=2.5,
                              zorder=5)

            # Add a proxy line to handle the Legend entry for Obligation cleanly
            ax_bar.plot([], [], color='red', linewidth=2.5, label='Obligation')

            ax_bar.axhline(0, color='black', linewidth=1)
            ax_bar.set_title(f"{region}")
            ax_bar.set_xlabel("")
            ax_bar.grid(True, linestyle=":", alpha=0.6, axis='y')

            # Explicitly horizontal period text labels
            ax_bar.tick_params(axis='x', rotation=0)

            for spine in ax_bar.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)

        axes_bar[0].set_ylabel("CSU (Mt)")
        fig_bar.suptitle("Cumulative CSU Balance (Up to 2040)", y=0.98, fontsize=_fontsize + 2)

        handles, labels = axes_bar[0].get_legend_handles_labels()
        desired_order_bar = ['Obligation', 'Generation', 'Buy', 'Sell']
        label_map_bar = dict(zip(labels, handles))
        ordered_handles_bar = [label_map_bar[lbl] for lbl in desired_order_bar if lbl in label_map_bar]
        ordered_labels_bar = [lbl for lbl in desired_order_bar if lbl in label_map_bar]

        fig_bar.legend(ordered_handles_bar, ordered_labels_bar, loc='upper center', bbox_to_anchor=(0.5, 0.03),
                       ncol=len(desired_order_bar), frameon=True, edgecolor='black', fontsize=_fontsize)

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(f"{DIR_BALANCE}/Figure3_CSU_Balance_Bar_Combined.png", dpi=600, bbox_inches='tight')
        plt.close(fig_bar)

except Exception as e:
    print(f"Skipping Figure 3 (CSU Balance). Error: {e}")

#== == == == == == == == == == == == == == == == == == == == ==
# Figure: Market Clearing Price (Shadow Price)
# ==========================================
try:
    # Adjust sheet name here if it's called something else in your workbook
    market_clearing_sheet = "CSU_Market_Prices"

    mc_df = pd.read_excel(RESULTS_PATH, sheet_name=market_clearing_sheet)

    if "Scenario" in mc_df.columns:
        mc_df = mc_df.drop(columns=["Scenario"])

    mc_df.columns = ["Index_1", "Raw_Dual_NPV", "Nominal_Price"]

    import re


    def extract_year(val):
        match = re.search(r'(20\d\d)', str(val))
        return int(match.group(1)) if match else None


    mc_df["year"] = mc_df["Index_1"].apply(extract_year)
    mc_df = mc_df[mc_df["year"].notna()]
    mc_df["year"] = mc_df["year"].astype(int)
    mc_df = mc_df[mc_df["year"] <= MAX_YEAR]

    if not mc_df.empty:
        # Plot Nominal Price (System-wide)
        plt.figure(figsize=(7, 4))
        ax_price = plt.gca()

        sns.lineplot(data=mc_df, x="year", y="Nominal_Price", marker="o", color="#675D50", ax=ax_price)
        ax_price.set_title("CSU Market Clearing Price (Nominal Price)")
        ax_price.set_ylabel("Nominal Price ($ / CSU)")
        ax_price.set_xlabel("")
        ax_price.set_xticks(target_years)
        ax_price.grid(True, linestyle=":", alpha=0.6)

        for spine in ax_price.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(f"{DIR_PRICE}/Market_Clearing_Nominal_Price.png", dpi=600, bbox_inches='tight')
        plt.close()

        # Plot Raw Dual NPV (System-wide)
        plt.figure(figsize=(7, 4))
        ax_dual = plt.gca()

        sns.lineplot(data=mc_df, x="year", y="Raw_Dual_NPV", marker="o", color="#ABC4AA", ax=ax_dual)
        ax_dual.set_title("CSU Market Clearing Price (Raw Dual NPV)")
        ax_dual.set_ylabel("Raw Dual NPV")
        ax_dual.set_xlabel("")
        ax_dual.set_xticks(target_years)
        ax_dual.grid(True, linestyle=":", alpha=0.6)

        for spine in ax_dual.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(f"{DIR_PRICE}/Market_Clearing_Raw_Dual_NPV.png", dpi=600, bbox_inches='tight')
        plt.close()

except Exception as e:
    print(f"Skipping Market Clearing Price plot. Error: {e}")

# ==========================================
# Figure 4: ETS vs Injected
# ==========================================
ets_agg = q_ets.groupby(["region", "year"])["value"].sum().reset_index()
inj_agg = q_inj.groupby(["region", "year"])["value"].sum().reset_index()

ets_agg["type"] = "Emitted CO$_2$"
inj_agg["type"] = "Injected"

df_fig4 = pd.concat([ets_agg, inj_agg])

type_palette = {"Emitted CO$_2$": "#675D50", "Injected": "#ABC4AA"}

unique_regions = df_fig4["region"].unique()
for region in unique_regions:
    df_region = df_fig4[df_fig4["region"] == region]
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df_region, x="year", y="value", hue="type", palette=type_palette, marker="o")
    plt.title(f"{region}")
    plt.ylabel("CO$_2$ (Mt)")
    plt.xlabel("")
    plt.xticks(target_years)
    plt.grid(True, linestyle=":", alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, frameon=False, fontsize=_fontsize)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig(f"{DIR_ETS}/Figure4_Node_{region}.png", dpi=600, bbox_inches='tight')
    plt.close()

overall_agg = df_fig4.groupby(["year", "type"])["value"].sum().reset_index()
plt.figure(figsize=(7, 4))
ax = sns.lineplot(data=overall_agg, x="year", y="value", hue="type", palette=type_palette, marker="o")
plt.title(f"Overall Emitted CO$_2$ vs Injected (Up to {MAX_YEAR})")
plt.ylabel("CO$_2$ (Mt)")
plt.xlabel("")
plt.xticks(target_years)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=False, fontsize=_fontsize)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.savefig(f"{DIR_ETS}/Figure4_Overall.png", dpi=600, bbox_inches='tight')
plt.close()

# ==========================================
# Figure 4b: CO2 Capture Fraction by Sector
# ==========================================
try:
    q_cap = pd.read_excel(RESULTS_PATH, sheet_name="q_CO2_cap")
    if "Scenario" in q_cap.columns:
        q_cap = q_cap.drop(columns=["Scenario"])
    q_cap.columns = ["region", "sector", "tech", "vintage", "year", "value"]
    q_cap = q_cap[q_cap["year"] <= MAX_YEAR]
    q_cap.loc[:, "value"] /= 1000000.0

    q_cap_agg = q_cap.groupby(["region", "sector", "year"])["value"].sum().reset_index()
    df_frac = pd.merge(q_ets, q_cap_agg, on=["region", "sector", "year"], suffixes=("_ets", "_cap"),
                       how="outer").fillna(0)
    df_frac["total_co2"] = df_frac["value_ets"] + df_frac["value_cap"]
    df_frac["capture_pct"] = 0.0
    mask = df_frac["total_co2"] > 0
    df_frac.loc[mask, "capture_pct"] = (df_frac.loc[mask, "value_cap"] / df_frac.loc[mask, "total_co2"]) * 100

    unique_regions_frac = df_frac["region"].unique()
    for region in unique_regions_frac:
        df_reg = df_frac[df_frac["region"] == region]
        g = sns.FacetGrid(df_reg, col="sector", col_wrap=3, height=3, aspect=1.2, sharey=True)
        g.map(sns.lineplot, "year", "capture_pct", color="#ABC4AA", marker="o")
        g.fig.suptitle(f"CO$_2$ Captured (% of Sector Total): {region}", y=1.05)
        g.set_titles(col_template="{col_name}")
        for ax in g.axes.flat:
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_ylabel("Capture Fraction (%)")
            ax.set_xlabel("")
            ax.set_xticks(target_years)
            ax.set_ylim(-5, 105)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)
        plt.tight_layout()
        g.savefig(f"{DIR_ETS}/Figure4b_Sector_CapturePct_{region}.png", dpi=600, bbox_inches='tight')
        plt.close()

except Exception as e:
    print(f"Skipping Figure 4b (Capture Fraction). Error: {e}")

# ==========================================
# Figure 4c: Absolute CO2 Dynamics per Sector
# ==========================================
try:
    unique_sectors = df_frac["sector"].unique()
    df_frac["total_co2_scaled"] = df_frac["total_co2"]
    df_frac["value_ets_scaled"] = df_frac["value_ets"]
    df_frac["value_cap_scaled"] = df_frac["value_cap"]

    for sector in unique_sectors:
        df_sec = df_frac[df_frac["sector"] == sector]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        sns.lineplot(data=df_sec, x="year", y="total_co2_scaled", hue="region", palette=region_color_map, marker="o",
                     ax=axes[0])
        axes[0].set_title("Total Generated CO$_2$")

        sns.lineplot(data=df_sec, x="year", y="value_ets_scaled", hue="region", palette=region_color_map, marker="o",
                     ax=axes[1])
        axes[1].set_title("Emitted CO$_2$")

        sns.lineplot(data=df_sec, x="year", y="value_cap_scaled", hue="region", palette=region_color_map, marker="o",
                     ax=axes[2])
        axes[2].set_title("Captured CO$_2$")

        for ax in axes:
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_ylabel("CO$_2$ (Mt)")
            ax.set_xlabel("")
            ax.set_xticks(target_years)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)

        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles=handles, labels=labels, frameon=False, fontsize=_fontsize, title="Region")

        fig.suptitle(f"Absolute CO$_2$ Dynamics: {sector}", y=1.05, fontsize=_fontsize + 4)
        plt.tight_layout()
        fig.savefig(f"{DIR_ETS}/Figure4c_{sector}_Absolute.png", dpi=600, bbox_inches='tight')
        plt.close(fig)

except Exception as e:
    print(f"Skipping Figure 4c (Absolute Emissions). Error: {e}")

# ==========================================
# Figure 5: Inter-Node Market Flows
# ==========================================
try:
    periods = {
        "2026-2030": (0, 2030), "2031-2035": (2031, 2035), "2036-2040": (2036, 2040),
        "2041-2045": (2041, 2045), "2046-2050": (2046, 2050)
    }

    fig_market = make_subplots(rows=1, cols=5, subplot_titles=list(periods.keys()),
                               specs=[[{"type": "sankey"}] * 5])

    all_market_nodes = sorted(all_regions)
    node_indices = {region: idx for idx, region in enumerate(all_market_nodes)}
    node_colors = [region_color_map.get(r, "#000000") for r in all_market_nodes]

    for idx, (period_name, (y_start, y_end)) in enumerate(periods.items()):
        buy_period = csu_buy[(csu_buy["year"] >= y_start) & (csu_buy["year"] <= y_end)]
        sell_period = csu_sell[(csu_sell["year"] >= y_start) & (csu_sell["year"] <= y_end)]

        net_pos = buy_period.groupby("region")["value"].sum() - sell_period.groupby("region")["value"].sum()
        net_sellers = net_pos[net_pos < 0].abs()
        net_buyers = net_pos[net_pos > 0]
        total_market_vol = net_buyers.sum()

        source, target, values, link_colors = [], [], [], []

        if total_market_vol > 0:
            for seller, sell_vol in net_sellers.items():
                for buyer, buy_vol in net_buyers.items():
                    buyer_share = buy_vol / total_market_vol
                    flow_vol = sell_vol * buyer_share
                    if flow_vol > 0.01:
                        source.append(node_indices[seller])
                        target.append(node_indices[buyer])
                        values.append(flow_vol)
                        hex_color = region_color_map[seller].lstrip('#')
                        link_colors.append(
                            f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, 0.4)")

        fig_market.add_trace(go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=all_market_nodes,
                      color=node_colors),
            link=dict(source=source, target=target, value=values, color=link_colors)
        ), row=1, col=idx + 1)

    fig_market.update_layout(title_text="Net Inter-Node CSU Market Flows", font_size=12, font_family="serif",
                             width=2000, height=500)
    fig_market.write_html(f"{DIR_MARKET}/Figure5_Market_Flows.html")
    try:
        fig_market.write_image(f"{DIR_MARKET}/Figure5_Market_Flows.png", scale=2)
    except:
        pass
except Exception as e:
    print(f"Skipping Figure 5. Error: {e}")

print(f"Figures generated successfully and organized in the '{OUTPUT_DIR}' directory.")