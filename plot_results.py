import os
import pandas as pd
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
DIR_SANKEY = os.path.join(OUTPUT_DIR, "sankey_csu")
DIR_ETS = os.path.join(OUTPUT_DIR, "ets_vs_injected")

for directory in [DIR_FLOWS, DIR_SOC, DIR_SANKEY, DIR_ETS]:
    os.makedirs(directory, exist_ok=True)

# 1. Load parameters for capacity
raw = load_data()
model_data = prepare_model_data(raw)
sink_cap = {k: v / 1000.0 for k, v in model_data["sink_block_cap"].items()}

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
# Apply Global Year Cutoff (<= 2040)
# ==========================================
MAX_YEAR = 2040

q_trans = q_trans[q_trans["year"] <= MAX_YEAR]
csu_buy = csu_buy[csu_buy["year"] <= MAX_YEAR]
csu_sell = csu_sell[csu_sell["year"] <= MAX_YEAR]
csu_gen = csu_gen[csu_gen["year"] <= MAX_YEAR]
csu_use = csu_use[csu_use["year"] <= MAX_YEAR]
q_inj = q_inj[q_inj["year"] <= MAX_YEAR]
q_ets = q_ets[q_ets["year"] <= MAX_YEAR]

# 3. Convert ONLY CO2 quantities from tonnes to kilotonnes (CSU stays unitless)
for df in [q_trans, q_inj, q_ets]:
    df.loc[:, "value"] /= 1000.0

# ==========================================
# Define Custom Node Palette
# ==========================================
custom_palette = ['#A9907E', '#F3DEBA', '#ABC4AA', '#675D50']

# Extract all unique regions to map them consistently to colors
all_regions = sorted(
    set(q_trans["region_from"].unique()) | set(q_trans["region_to"].unique()) | set(q_inj["region"].unique()))
region_color_map = {region: custom_palette[i % len(custom_palette)] for i, region in enumerate(all_regions)}

# Specific years to display on x-axis (Cut to 2040)
target_years = [2025, 2030, 2035, 2040]

# ==========================================
# Figure 1: Flows of CO2 between nodes and CSUs
# ==========================================
fig1, axes = plt.subplots(1, 2, figsize=(10, 4))

# Physical CO2 transport
trans_agg = q_trans.groupby(["region_from", "region_to"])["value"].sum().reset_index()
trans_agg = trans_agg[trans_agg["region_from"] != trans_agg["region_to"]]
trans_pivot = trans_agg.pivot(index="region_from", columns="region_to", values="value").fillna(0)

bar_colors = [region_color_map[col] for col in trans_pivot.columns]
trans_pivot.plot(kind="bar", stacked=True, ax=axes[0], color=bar_colors, edgecolor='black', linewidth=0.5)
axes[0].set_title(f"Total Physical CO$_2$ Transport Between Nodes\n(Up to {MAX_YEAR})")
axes[0].set_ylabel("CO$_2$ (kt)")
axes[0].set_xlabel("")
axes[0].tick_params(axis='x', labelrotation=0)
axes[0].grid(True, linestyle=":", alpha=0.6, axis='y')

# CSU net flows
csu_net = csu_buy.groupby("region")["value"].sum() - csu_sell.groupby("region")["value"].sum()
net_colors = [region_color_map[r] for r in csu_net.index]
csu_net.plot(kind="bar", ax=axes[1], color=net_colors, edgecolor='black', linewidth=0.5)
axes[1].set_title(f"Net CSU Position (Bought - Sold) per Node\n(Up to {MAX_YEAR})")
axes[1].set_ylabel("CSU")
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
plt.ylabel("CO$_2$ (kt)")
plt.xlabel("")
plt.grid(True, linestyle=":", alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig(f"{DIR_SOC}/Figure2_SOC_Overall.png", dpi=600, bbox_inches='tight')
plt.close()

# ==========================================
# Figure 2b: State of Charge vs Active Sinks (Overall)
# ==========================================
sink_bdv = pd.read_excel(RESULTS_PATH, sheet_name="sink_bdv")
sink_bdv = rename_cols(sink_bdv, ["region", "block", "year", "value"])
sink_bdv = sink_bdv[sink_bdv["year"] <= MAX_YEAR]  # Apply filter to investment logic
sink_bdv_sorted = sink_bdv.sort_values(by=["region", "block", "year"])
sink_bdv_sorted["is_active"] = sink_bdv_sorted.groupby(["region", "block"])["value"].cumsum()

active_cap_records = []
for _, row in sink_bdv_sorted.iterrows():
    cap = sink_cap.get((row["region"], row["block"]), 0)
    active_cap_records.append({
        "region": row["region"],
        "block": row["block"],
        "year": row["year"],
        "active_cap": cap if row["is_active"] >= 1 else 0
    })

active_cap_df = pd.DataFrame(active_cap_records)
active_cap_agg = active_cap_df.groupby(["region", "year"])["active_cap"].sum().reset_index()

plt.figure(figsize=(8, 4))
ax2b = plt.gca()

sns.barplot(data=active_cap_agg, x="year", y="active_cap", hue="region", palette=region_color_map, alpha=0.3,
            errorbar=None, edgecolor='none', order=unique_years_bar)
sns.barplot(data=soc_agg, x="year", y="cum_inj", hue="region", palette=region_color_map, alpha=1.0, errorbar=None,
            edgecolor='black', linewidth=0.5, order=unique_years_bar)

handles, labels = ax2b.get_legend_handles_labels()
plt.legend(handles[n:2 * n], labels[:n], title="Region", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False,
           fontsize=_fontsize)

ax2b.set_xticks(tick_pos)
ax2b.set_xticklabels(tick_lab)

plt.title("State of Charge of Sinks over Time\n(Active Sink Capacity vs Cumulative Injection)")
plt.ylabel("CO$_2$ (kt)")
plt.xlabel("")
plt.grid(True, linestyle=":", alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig(f"{DIR_SOC}/Figure2b_SOC_Active_Overall.png", dpi=600, bbox_inches='tight')
plt.close()

# ==========================================
# Figure 2c: State of Charge vs Active Sinks (Per Node)
# ==========================================
for region in all_regions:
    df_active_reg = active_cap_agg[active_cap_agg["region"] == region]
    df_soc_reg = soc_agg[soc_agg["region"] == region]

    if df_active_reg.empty and df_soc_reg.empty:
        continue

    plt.figure(figsize=(6, 4))
    ax2c = plt.gca()
    r_color = region_color_map[region]

    sns.barplot(data=df_active_reg, x="year", y="active_cap", color=r_color, alpha=0.3, errorbar=None, edgecolor='none',
                label="Active Sink Capacity", order=unique_years_bar)
    sns.barplot(data=df_soc_reg, x="year", y="cum_inj", color=r_color, alpha=1.0, errorbar=None, edgecolor='black',
                linewidth=0.5, label="Cumulative Injection", order=unique_years_bar)

    handles, labels = ax2c.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=_fontsize)

    ax2c.set_xticks(tick_pos)
    ax2c.set_xticklabels(tick_lab)

    plt.title(f"State of Charge vs Active Capacity: {region}")
    plt.ylabel("CO$_2$ (kt)")
    plt.xlabel("")
    plt.grid(True, linestyle=":", alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig(f"{DIR_SOC}/Figure2b_SOC_Active_{region}.png", dpi=600, bbox_inches='tight')
    plt.close()

# ==========================================
# Figure 3: CSU Sourcing Sankey per Node
# (Total AND Snapshot formats)
# ==========================================
regions = q_inj["region"].unique()
sankey_years = [2030, 2035, 2040]

for region in regions:

    labels = ["Gen", "Buy", f"{region} Pool", "Use", "Sell"]
    node_colors = ['#ABC4AA', '#A9907E', region_color_map.get(region, '#F3DEBA'), '#675D50', '#A9907E']
    source = [0, 1, 2, 2]
    target = [2, 2, 3, 4]

    # --- 1. Total Sankey (aggregated over all years up to MAX_YEAR) ---
    gen_tot = csu_gen[csu_gen["region"] == region]["value"].sum()
    buy_tot = csu_buy[csu_buy["region"] == region]["value"].sum()
    use_tot = csu_use[csu_use["region"] == region]["value"].sum()
    sell_tot = csu_sell[csu_sell["region"] == region]["value"].sum()
    val_tot = [gen_tot, buy_tot, use_tot, sell_tot]

    fig_total = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=labels, color=node_colors
        ),
        link=dict(
            source=source, target=target, value=val_tot,
            color='rgba(169, 144, 126, 0.4)'
        )
    )])
    fig_total.update_layout(
        title_text=f"Total CSU Balance Sankey for {region} (Up to {MAX_YEAR})",
        font_size=12, font_family="serif", width=700, height=450
    )

    fig_total.write_html(f"{DIR_SANKEY}/Figure3_{region}_CSU_Sankey_Total.html")
    try:
        fig_total.write_image(f"{DIR_SANKEY}/Figure3_{region}_CSU_Sankey_Total.png", scale=2)
    except:
        pass

    # --- 2. Snapshot Subplots (1x3 grid for chosen years) ---
    fig_snap = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Year {y}" for y in sankey_years],
        specs=[[{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}]]
    )

    for idx, year in enumerate(sankey_years):
        gen_yr = csu_gen[(csu_gen["region"] == region) & (csu_gen["year"] == year)]["value"].sum()
        buy_yr = csu_buy[(csu_buy["region"] == region) & (csu_buy["year"] == year)]["value"].sum()
        use_yr = csu_use[(csu_use["region"] == region) & (csu_use["year"] == year)]["value"].sum()
        sell_yr = csu_sell[(csu_sell["region"] == region) & (csu_sell["year"] == year)]["value"].sum()

        val_yr = [gen_yr, buy_yr, use_yr, sell_yr]

        sankey_trace = go.Sankey(
            node=dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5),
                label=labels, color=node_colors
            ),
            link=dict(
                source=source, target=target, value=val_yr,
                color='rgba(169, 144, 126, 0.4)'
            )
        )
        fig_snap.add_trace(sankey_trace, row=1, col=idx + 1)

    fig_snap.update_layout(
        title_text=f"CSU Balance Sankey Snapshots for {region}",
        font_size=12, font_family="serif", width=1400, height=450
    )

    fig_snap.write_html(f"{DIR_SANKEY}/Figure3_{region}_CSU_Sankey_Snapshots.html")
    try:
        fig_snap.write_image(f"{DIR_SANKEY}/Figure3_{region}_CSU_Sankey_Snapshots.png", scale=2)
    except:
        pass

# ==========================================
# Figure 4: ETS vs Injected
# ==========================================
ets_agg = q_ets.groupby(["region", "year"])["value"].sum().reset_index()
inj_agg = q_inj.groupby(["region", "year"])["value"].sum().reset_index()

ets_agg["type"] = "Paid ETS"
inj_agg["type"] = "Injected"

df_fig4 = pd.concat([ets_agg, inj_agg])
df_fig4["value"] /= 1000.0

type_palette = {"Paid ETS": "#675D50", "Injected": "#ABC4AA"}

# Plot per node
unique_regions = df_fig4["region"].unique()

for region in unique_regions:
    df_region = df_fig4[df_fig4["region"] == region]

    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df_region, x="year", y="value", hue="type", palette=type_palette, marker="o")

    plt.title(f"{region}")
    plt.ylabel("CO$_2$ ($10^3$ kt)")
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

# Plot overall
overall_agg = df_fig4.groupby(["year", "type"])["value"].sum().reset_index()
plt.figure(figsize=(7, 4))
ax = sns.lineplot(data=overall_agg, x="year", y="value", hue="type", palette=type_palette, marker="o")
plt.title("Overall CO$_2$ Paid ETS vs Injected")
plt.ylabel("CO$_2$ ($10^3$ kt)")
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

print(f"Figures generated successfully and organized in the '{OUTPUT_DIR}' directory.")