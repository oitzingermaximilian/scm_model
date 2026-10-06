import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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

_fontsize = 12
plt.rcParams["xtick.labelsize"] = _fontsize
plt.rcParams["ytick.labelsize"] = _fontsize
plt.rcParams["axes.labelsize"] = _fontsize
plt.rcParams["axes.titlesize"] = _fontsize + 2

MAX_YEAR = 2040
TARGET_YEARS = [2026, 2030, 2035, 2040]
DESIRED_REGION_ORDER = ["UK", "EU27", "NO"]

# Global Region Colors
REGION_COLOR_MAP = {
    "EU27": '#A9907E',
    "NO": '#F3DEBA',
    "UK": '#ABC4AA'
}
C_GEN, C_BUY, C_SELL, C_OB = '#A8D8EA', '#AA96DA', '#FFFFD2', '#FCBAD3'

# Load Base Data Once (Needed for mapping ETS prices)
RAW = load_data()
MODEL_DATA = prepare_model_data(RAW)
SINK_CAP = {k: v / 1000000.0 for k, v in MODEL_DATA["sink_block_cap"].items()}

# Incline Scenario Years
INCLINE_YEARS = [2030, 2035, 2040, 2045, 2050]
# Colors for the trajectories
INCLINE_COLORS = {
    2030: '#D65A9C',  # Base Case (Steil ab 2030)
    2035: '#14B9DC',
    2040: '#76C68F',
    2045: '#C4E17F',
    2050: '#F8ED6F'  # Low Case (Flach bis 2050)
}


# ==========================================
# Helper Functions
# ==========================================
def get_latest_result_path(scenario_name):
    """Finds the model results file for a given scenario name."""
    path1 = os.path.join("results", scenario_name, "model_results.xlsx")
    path2 = os.path.join("results", scenario_name, f"SCM_RESULTS_{scenario_name}.xlsx")

    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        search_pattern = f"results/{scenario_name}/**/*.xlsx"
        files = glob.glob(search_pattern, recursive=True)
        if files:
            return max(files, key=os.path.getctime)
        print(f"Warning: No results found for scenario '{scenario_name}'")
        return None


def sort_regions(region_list):
    return sorted(region_list, key=lambda x: DESIRED_REGION_ORDER.index(x) if x in DESIRED_REGION_ORDER else 999)


def rename_cols(df, cols):
    if "Scenario" in df.columns:
        df = df.drop(columns=["Scenario"])
    df.columns = cols
    return df


def get_period(year):
    if 2026 <= year <= 2030:
        return "2026-2030"
    elif 2031 <= year <= 2035:
        return "2031-2035"
    elif 2036 <= year <= 2040:
        return "2036-2040"
    return None


def get_all_folders(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]


# ==========================================
# 1. Standard Plotting Function
# ==========================================
def generate_standard_plots(scenario_name):
    result_path = get_latest_result_path(scenario_name)
    if not result_path:
        return

    base_output_dir = f"figures_{scenario_name}"
    print(f"\n--- Generating Standard Plots for {scenario_name} ---")

    q_trans = rename_cols(pd.read_excel(result_path, sheet_name="q_CO2_trans"),
                          ["region_from", "region_to", "year", "value"])
    csu_buy = rename_cols(pd.read_excel(result_path, sheet_name="CSU_buy"), ["region", "year", "value"])
    csu_sell = rename_cols(pd.read_excel(result_path, sheet_name="CSU_sell"), ["region", "year", "value"])
    csu_gen = rename_cols(pd.read_excel(result_path, sheet_name="CSU_generate"), ["region", "year", "value"])
    csu_use = rename_cols(pd.read_excel(result_path, sheet_name="CSU_use"), ["region", "year", "value"])
    q_inj = rename_cols(pd.read_excel(result_path, sheet_name="q_CO2_inj"), ["region", "block", "year", "value"])
    q_ets = rename_cols(pd.read_excel(result_path, sheet_name="q_CO2_ETS"), ["region", "sector", "year", "value"])

    dfs = [q_trans, csu_buy, csu_sell, csu_gen, csu_use, q_inj, q_ets]
    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i]["year"] <= MAX_YEAR]
        dfs[i].loc[:, "value"] /= 1000000.0
    q_trans, csu_buy, csu_sell, csu_gen, csu_use, q_inj, q_ets = dfs

    all_regions = sort_regions(
        list(set(q_trans["region_from"].unique()) | set(q_trans["region_to"].unique()) | set(q_inj["region"].unique())))
    custom_palette = ['#675D50', '#A8D8EA', '#AA96DA', '#FFFFD2']
    for i, r in enumerate(all_regions):
        if r not in REGION_COLOR_MAP:
            REGION_COLOR_MAP[r] = custom_palette[i % len(custom_palette)]

    for with_titles in [True, False]:
        folder_suffix = "with_titles" if with_titles else "without_titles"
        output_dir = os.path.join(base_output_dir, folder_suffix)
        dirs = {
            "FLOWS": os.path.join(output_dir, "flows"),
            "SOC": os.path.join(output_dir, "state_of_charge"),
            "BAL": os.path.join(output_dir, "csu_balance"),
            "ETS": os.path.join(output_dir, "ets_vs_injected"),
            "MKT": os.path.join(output_dir, "market_flows"),
            "PRC": os.path.join(output_dir, "shadow_prices")
        }
        for d in dirs.values(): os.makedirs(d, exist_ok=True)

        # ---------------- Figure 1a ----------------
        plt.figure(figsize=(5, 5))
        ax1a = plt.gca()
        trans_agg = q_trans.groupby(["region_from", "region_to"])["value"].sum().reset_index()
        trans_agg = trans_agg[trans_agg["region_from"] != trans_agg["region_to"]]
        trans_pivot = trans_agg.pivot(index="region_from", columns="region_to", values="value").fillna(0)
        sorted_cols = [c for c in DESIRED_REGION_ORDER if c in trans_pivot.columns]
        trans_pivot = trans_pivot[sorted_cols]
        bar_colors = [REGION_COLOR_MAP[col] for col in trans_pivot.columns]
        trans_pivot = trans_pivot.reindex([r for r in DESIRED_REGION_ORDER if r in trans_pivot.index])

        if not trans_pivot.empty:
            trans_pivot.plot(kind="bar", stacked=True, ax=ax1a, color=bar_colors, edgecolor='black', linewidth=0.5)
            if with_titles: ax1a.set_title("Total Physical CO$_2$ Transport Between Nodes")
            ax1a.set_ylabel("CO$_2$ Transported (MtCO$_2$)")
            ax1a.set_xlabel("")
            ax1a.tick_params(axis='x', labelrotation=0)
            ax1a.grid(True, linestyle=":", alpha=0.6, axis='y')
            ax1a.legend(title="To Region" if with_titles else None, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                        ncol=len(trans_pivot.columns), frameon=True, edgecolor='black', fontsize=_fontsize)
            plt.tight_layout(rect=[0, 0.12, 1, 1])
            plt.savefig(os.path.join(dirs["FLOWS"], "Figure1a_CO2_Transport.pdf"), dpi=600, bbox_inches='tight')
        plt.close()

        # ---------------- Figure 1b ----------------
        plt.figure(figsize=(5, 5))
        ax1b = plt.gca()
        csu_net = csu_buy.groupby("region")["value"].sum() - csu_sell.groupby("region")["value"].sum()
        csu_net = csu_net.reindex([r for r in DESIRED_REGION_ORDER if r in csu_net.index])
        if not csu_net.empty and csu_net.sum() != 0:
            net_colors = [C_BUY if val > 0 else C_SELL for val in csu_net]
            csu_net.plot(kind="bar", ax=ax1b, color=net_colors, edgecolor='black', linewidth=0.5, width=0.4)
            if with_titles: ax1b.set_title("Net CSU Trading Balance per Node")
            ax1b.set_ylabel("Net CSU trade balance (MtCO$_2$)")
            ax1b.set_xlabel("")
            ax1b.tick_params(axis='x', labelrotation=0)
            ax1b.axhline(0, color='black', lw=1)
            ax1b.grid(True, linestyle=":", alpha=0.6, axis='y')
            legend_handles_1b = [mpatches.Patch(facecolor=C_BUY, edgecolor='black', label='Net Importer'),
                                 mpatches.Patch(facecolor=C_SELL, edgecolor='black', label='Net Exporter')]
            ax1b.legend(handles=legend_handles_1b, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                        frameon=True, edgecolor='black', fontsize=_fontsize)
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["FLOWS"], "Figure1b_CSU_Trade_Balance.pdf"), dpi=600, bbox_inches='tight')
        plt.close()

        # ---------------- Figures 2, 2b, 2c (State of Charge) ----------------
        import matplotlib.ticker as ticker
        all_years = sorted(set(q_inj["year"].unique()))
        if not all_years: all_years = TARGET_YEARS
        inj_added = q_inj.groupby(["region", "year"])["value"].sum().reset_index()
        inj_pivot = inj_added.pivot(index="year", columns="region", values="value").fillna(0).reindex(all_years).fillna(
            0)
        soc_agg = inj_pivot.cumsum().reset_index().melt(id_vars="year", var_name="region", value_name="cum_inj")

        cap_file_path = "potential_sink_cap_plot.xlsx"
        if os.path.exists(cap_file_path):
            cap_raw = pd.read_excel(cap_file_path)
            if "Year" in cap_raw.columns:
                cap_raw = cap_raw.rename(columns={"Year": "year"})
            elif "year" not in cap_raw.columns:
                cap_raw = cap_raw.rename(columns={cap_raw.columns[0]: "year"})
            cap_melted = cap_raw.melt(id_vars="year", var_name="region", value_name="total_cap")
            cap_melted["total_cap"] /= 1e6
            cap_df = pd.merge(soc_agg, cap_melted, on=["year", "region"], how="left").fillna({"total_cap": 0})

            plt.figure(figsize=(9, 4.8))
            ax2 = plt.gca()
            sns.barplot(data=cap_df, x="year", y="total_cap", hue="region", hue_order=DESIRED_REGION_ORDER,
                        palette=REGION_COLOR_MAP, alpha=0.3, edgecolor='none', order=all_years)
            sns.barplot(data=soc_agg, x="year", y="cum_inj", hue="region", hue_order=DESIRED_REGION_ORDER,
                        palette=REGION_COLOR_MAP, alpha=1.0, edgecolor='black', linewidth=0.5, order=all_years)
            handles, labels = ax2.get_legend_handles_labels()
            n = len(cap_df["region"].unique())
            plt.legend(handles[n:2 * n], labels[:n], title="Region" if with_titles else None, loc='upper center',
                       bbox_to_anchor=(0.5, -0.18), ncol=n, frameon=True, edgecolor='black', fontsize=_fontsize)
            ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
            current_target_years = sorted(list(set(TARGET_YEARS + ([2026] if 2026 in all_years else []))))
            tick_pos = [all_years.index(y) for y in current_target_years if y in all_years]
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels([y for y in current_target_years if y in all_years])
            if with_titles: plt.title("State of Charge of Sinks over Time")
            plt.ylabel("CO$_2$ (MtCO$_2$)")
            plt.xlabel("")
            plt.grid(True, linestyle=":", alpha=0.6, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["SOC"], "Figure2_SOC_Overall.pdf"), dpi=600, bbox_inches='tight')
            plt.close()

            # ---------------- Figure 3: CSU Balance ----------------
            try:
                # 1. Load capture data for local generation tracing
                q_cap = pd.read_excel(result_path, sheet_name="q_CO2_cap")
                if "Scenario" in q_cap.columns:
                    q_cap = q_cap.drop(columns=["Scenario"])
                q_cap.columns = ["region", "sector", "tech", "vintage", "year", "value"]
                q_cap = q_cap[q_cap["year"] <= MAX_YEAR]
                q_cap.loc[:, "value"] /= 1000000.0

                # Map 5-year periods
                q_cap["Period"] = q_cap["year"].map(get_period)
                q_trans["Period"] = q_trans["year"].map(get_period)

                periods_order = ['2026-2030', '2031-2035', '2036-2040']
                regions = sort_regions(q_inj["region"].unique())
                bar_chart_data = {}

                for region in regions:
                    df_gen = csu_gen[csu_gen["region"] == region].groupby("year")["value"].sum()
                    df_buy = csu_buy[csu_buy["region"] == region].groupby("year")["value"].sum()
                    df_sell = csu_sell[csu_sell["region"] == region].groupby("year")["value"].sum()
                    df_use = csu_use[csu_use["region"] == region].groupby("year")["value"].sum()
                    df_bal = pd.DataFrame({"Gen": df_gen, "Buy": df_buy, "Sell": df_sell, "Use": df_use}).fillna(0)
                    df_bal = df_bal.loc[df_bal.index <= MAX_YEAR]
                    if df_bal.empty:
                        continue
                    years = df_bal.index.tolist()

                    # Area Chart (Annual Overview)
                    plt.figure(figsize=(7, 4))
                    ax_area = plt.gca()
                    ax_area.fill_between(years, 0, df_bal["Use"], color=C_OB, alpha=0.15, label='Use')
                    ax_area.plot(years, df_bal["Use"], color=C_OB, linestyle='--', linewidth=2.0, label='Obligation')
                    ax_area.stackplot(years, df_bal["Gen"], df_bal["Buy"], labels=['Generation', 'Buy'],
                                      colors=[C_GEN, C_BUY], alpha=0.9, edgecolor='black', linewidth=0.5)
                    ax_area.fill_between(years, 0, -df_bal["Sell"], color=C_SELL, alpha=0.9, label='Sell',
                                         edgecolor='black', linewidth=0.5)
                    ax_area.axhline(0, color='black', linewidth=1)
                    if with_titles:
                        ax_area.set_title(f"CSU Balance: {region} (Area)")
                    ax_area.set_ylabel("CSU Balance (MtCO$_2$)")
                    ax_area.set_xlabel("")
                    ax_area.set_xticks([y for y in TARGET_YEARS if y in years])
                    ax_area.grid(True, linestyle=":", alpha=0.6, axis='y')
                    handles, labels = ax_area.get_legend_handles_labels()
                    desired_order = ['Obligation', 'Use', 'Generation', 'Buy', 'Sell']
                    lmap = dict(zip(labels, handles))
                    ax_area.legend([lmap[l] for l in desired_order if l in lmap],
                                   [l for l in desired_order if l in lmap],
                                   loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=True,
                                   edgecolor='black')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dirs["BAL"], f"Figure3_CSU_Balance_Area_{region}.pdf"), dpi=600,
                                bbox_inches='tight')
                    plt.close()

                    # Store data for combined bar chart
                    df_bal['Period'] = df_bal.index.map(get_period)
                    bar_chart_data[region] = df_bal.groupby('Period').sum().reindex(periods_order).fillna(0)

                # Combined Stacked Bar Chart for CSU Balance with Node Origin Breakdown
                if bar_chart_data:
                    num_regions = len(bar_chart_data)
                    fig_bar, axes_bar = plt.subplots(1, num_regions, figsize=(3.5 * num_regions, 4.5), sharey=False)
                    if num_regions == 1:
                        axes_bar = [axes_bar]

                    # --- CORRECTED MAX Y MATH ---
                    max_y = max(
                        [(d["Gen"] + d["Buy"]).max() for d in bar_chart_data.values()] +
                        [d["Use"].max() for d in bar_chart_data.values()]
                    )
                    min_y = min([-d["Sell"].max() for d in bar_chart_data.values()])

                    plotted_origins = set()

                    for ax_bar, region in zip(axes_bar,
                                              [r for r in DESIRED_REGION_ORDER if r in bar_chart_data.keys()]):
                        df_period = bar_chart_data[region]

                        # --- FIX APPLIED: Use Actual Generation ---
                        real_gen = df_period["Gen"]

                        # -------------------------------------------------------------
                        # PHYSICAL NODE ORIGIN TRACING
                        # -------------------------------------------------------------
                        df_gen_nodes = pd.DataFrame(0.0, index=periods_order, columns=all_regions)

                        for p in periods_order:
                            # 1. Local Capture in this node
                            cap_local = q_cap[(q_cap["region"] == region) & (q_cap["Period"] == p)]["value"].sum()

                            # 2. Physical Imports from each originating node j != region
                            inflows = {}
                            for r_origin in all_regions:
                                if r_origin == region:
                                    inflows[r_origin] = cap_local
                                else:
                                    imp = q_trans[(q_trans["region_from"] == r_origin) &
                                                  (q_trans["region_to"] == region) &
                                                  (q_trans["Period"] == p)]["value"].sum()
                                    inflows[r_origin] = imp

                            total_pool = sum(inflows.values())
                            gen_vol = real_gen.loc[p] if p in real_gen.index else 0.0

                            if total_pool > 0 and gen_vol > 0:
                                for r_origin in all_regions:
                                    share = inflows[r_origin] / total_pool
                                    df_gen_nodes.loc[p, r_origin] = share * gen_vol
                            elif gen_vol > 0:
                                df_gen_nodes.loc[p, region] = gen_vol

                        # Plot stacked Generation by Origin Node using TRUE generation
                        bottom_gen = np.zeros(len(periods_order))
                        ordered_origins = [r for r in DESIRED_REGION_ORDER if r in all_regions] + \
                                          [r for r in all_regions if r not in DESIRED_REGION_ORDER]

                        for r_origin in ordered_origins:
                            vals = df_gen_nodes[r_origin].values
                            if (vals > 0.001).any():
                                plotted_origins.add(r_origin)
                            c = REGION_COLOR_MAP.get(r_origin, '#CCCCCC')
                            ax_bar.bar(periods_order, vals, bottom=bottom_gen, width=0.45,
                                       color=c, edgecolor='black', linewidth=0.5)
                            bottom_gen += vals

                        # Plot CSU Buy stacked on top of TRUE Generation
                        ax_bar.bar(periods_order, df_period["Buy"], bottom=real_gen, width=0.45,
                                   color=C_BUY, edgecolor='black', linewidth=0.5)

                        # Plot CSU Sell below 0
                        ax_bar.bar(periods_order, -df_period["Sell"], width=0.45,
                                   color=C_SELL, edgecolor='black', linewidth=0.5)

                        # Plot Obligation tick mark
                        for i, row in enumerate(df_period.itertuples()):
                            ax_bar.hlines(y=row.Use, xmin=i - 0.225, xmax=i + 0.225,
                                          color=C_OB, linewidth=2.5, zorder=5)

                        ax_bar.set_title(region)
                        ax_bar.set_ylim(min_y * 1.15 if min_y < 0 else 0, max_y * 1.15)
                        ax_bar.axhline(0, color='black', linewidth=1)
                        ax_bar.grid(True, linestyle=":", alpha=0.6, axis='y')

                    if with_titles:
                        fig_bar.suptitle("Cumulative CSU Balance by CO$_2$ Origin Node", y=0.98, fontsize=_fontsize + 2)

                    # Legend with origin nodes
                    custom_handles = [
                        mlines.Line2D([], [], color=C_OB, linewidth=2.5, label='Obligation'),
                        mpatches.Patch(facecolor=C_BUY, edgecolor='black', label='CSU Buy'),
                        mpatches.Patch(facecolor=C_SELL, edgecolor='black', label='CSU Sell')
                    ]
                    for r_origin in [r for r in DESIRED_REGION_ORDER if r in plotted_origins]:
                        custom_handles.append(
                            mpatches.Patch(facecolor=REGION_COLOR_MAP.get(r_origin, '#CCCCCC'),
                                           edgecolor='black', label=f'CSU generate ({r_origin} CO$_2$)')
                        )

                    fig_bar.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                                   ncol=min(6, len(custom_handles)), frameon=True, edgecolor='black',
                                   fontsize=_fontsize)

                    plt.tight_layout(rect=[0, 0.12, 1, 1])
                    plt.savefig(os.path.join(dirs["BAL"], "Figure3_CSU_Balance_Bar_Combined.pdf"), dpi=600,
                                bbox_inches='tight')
                    plt.close(fig_bar)
            except Exception as e:
                print(f"Skipping Figure 3. Error: {e}")

        # ---------------- Figure 4: ETS vs Injected ----------------
        try:
            q_cap = pd.read_excel(result_path, sheet_name="q_CO2_cap")
            if "Scenario" in q_cap.columns: q_cap = q_cap.drop(columns=["Scenario"])
            q_cap.columns = ["region", "sector", "tech", "vintage", "year", "value"]
            q_cap = q_cap[q_cap["year"] <= MAX_YEAR]
            q_cap.loc[:, "value"] /= 1000000.0

            q_cap_agg = q_cap.groupby(["region", "sector", "year"])["value"].sum().reset_index()
            df_frac = pd.merge(q_ets, q_cap_agg, on=["region", "sector", "year"], suffixes=("_ets", "_cap"),
                               how="outer").fillna(0)
            df_frac["total_co2"] = df_frac["value_ets"] + df_frac["value_cap"]
            sector_mapping = {"Industry_cement": "Cement", "Industry_chemicals": "Chemicals",
                              "Industry_iron_steel": "Iron & Steel", "Industry_other": "Other",
                              "Industry_refining": "Refining"}
            df_frac["sector"] = df_frac["sector"].replace(sector_mapping)

            all_sectors = sorted(df_frac["sector"].unique())
            for fig_name, is_global in [("Figure4x_Grid_RowSharedY", False), ("Figure4z_Grid_GlobalSharedY", True)]:
                fig, axes = plt.subplots(len(all_sectors), 3, figsize=(10, 2.5 * len(all_sectors)), sharex=True)
                if len(all_sectors) == 1: axes = np.array([axes])
                glob_max = df_frac[df_frac["region"].isin(DESIRED_REGION_ORDER)]["total_co2"].max()

                for r, sector in enumerate(all_sectors):
                    row_max = df_frac[(df_frac["sector"] == sector) & (df_frac["region"].isin(DESIRED_REGION_ORDER))][
                        "total_co2"].max()
                    for c, region in enumerate(DESIRED_REGION_ORDER):
                        ax = axes[r, c]
                        df_plot = df_frac[(df_frac["sector"] == sector) & (df_frac["region"] == region)].sort_values(
                            "year")
                        df_plot = df_plot[df_plot["year"] >= 2026]
                        if not df_plot.empty:
                            ax.stackplot(df_plot["year"], df_plot["value_cap"], df_plot["value_ets"],
                                         labels=["Captured", "Uncaptured"], colors=["#ABC4AA", "#675D50"], alpha=0.8)
                            ax.plot(df_plot["year"], df_plot["total_co2"], color="black", linestyle="--", linewidth=1.5,
                                    label="Total")
                        ax.set_ylim(0, glob_max * 1.1 if is_global else row_max * 1.1)
                        ax.set_xlim(2025, 2041)
                        ax.set_xticks(TARGET_YEARS)
                        if r == 0: ax.set_title(region)
                        if c == 0: ax.set_ylabel(f"{sector}")
                        ax.grid(True, linestyle=":", alpha=0.6)

                if with_titles: fig.suptitle("Sectoral CO$_2$ Distribution", y=1.02, fontsize=_fontsize + 2)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=True)
                plt.tight_layout()
                plt.savefig(os.path.join(dirs["ETS"], f"{fig_name}.pdf"), dpi=600, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"Skipping Figure 4x/z. Error: {e}")

        # ---------------- Figure: Market Clearing Price ----------------
        try:
            mc_df = rename_cols(pd.read_excel(result_path, sheet_name="CSU_Market_Prices"),
                                ["Index_1", "Raw_Dual", "Nominal_Price"])
            mc_df["year"] = mc_df["Index_1"].apply(
                lambda val: int(re.search(r'(20\d\d)', str(val)).group(1)) if pd.notna(val) else None)
            mc_df = mc_df.dropna(subset=['year'])
            mc_df["year"] = mc_df["year"].astype(int)
            mc_df = mc_df[mc_df["year"] <= MAX_YEAR]
            mc_df["ETS_Price"] = mc_df["year"].map(MODEL_DATA.get("c_ETS", {}))

            if not mc_df.empty:
                plt.figure(figsize=(7, 4))
                ax_price = plt.gca()
                sns.lineplot(data=mc_df, x="year", y="Nominal_Price", color="#1F3A5F", linewidth=2.5, label="CSU Price",
                             ax=ax_price)
                if not mc_df["ETS_Price"].isna().all():
                    sns.lineplot(data=mc_df, x="year", y="ETS_Price", color="#D6604D", linestyle="--", linewidth=2.5,
                                 label="ETS Price", ax=ax_price)

                if with_titles: ax_price.set_title("Market Clearing Price vs ETS")
                ax_price.set_ylabel("€/tCO$_2$")
                ax_price.set_xticks(TARGET_YEARS)
                ax_price.grid(True, linestyle=":", alpha=0.6)
                ax_price.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True, edgecolor='black')
                plt.tight_layout()
                plt.savefig(os.path.join(dirs["PRC"], "Market_Clearing_Prices.pdf"), dpi=600, bbox_inches='tight')
                plt.close()
        except Exception:
            pass

        # ---------------- Figure 5: Sankey ----------------
        try:
            periods = {"2026-2030": (0, 2030), "2031-2035": (2031, 2035), "2036-2040": (2036, 2040)}
            fig_market = make_subplots(rows=1, cols=3, subplot_titles=list(periods.keys()),
                                       specs=[[{"type": "sankey"}] * 3])
            node_indices = {r: idx for idx, r in enumerate(all_regions)}
            node_colors = [REGION_COLOR_MAP.get(r, "#000000") for r in all_regions]

            for idx, (p_name, (y_start, y_end)) in enumerate(periods.items()):
                b_prof = csu_buy[(csu_buy["year"] >= y_start) & (csu_buy["year"] <= y_end)]
                s_prof = csu_sell[(csu_sell["year"] >= y_start) & (csu_sell["year"] <= y_end)]
                net_pos = b_prof.groupby("region")["value"].sum() - s_prof.groupby("region")["value"].sum()
                net_sellers, net_buyers = net_pos[net_pos < 0].abs(), net_pos[net_pos > 0]
                tot_vol = net_buyers.sum()
                source, target, values, link_colors = [], [], [], []

                if tot_vol > 0:
                    for seller, s_vol in net_sellers.items():
                        for buyer, b_vol in net_buyers.items():
                            f_vol = s_vol * (b_vol / tot_vol)
                            if f_vol > 0.01:
                                source.append(node_indices[seller])
                                target.append(node_indices[buyer])
                                values.append(f_vol)
                                hx = REGION_COLOR_MAP[seller].lstrip('#')
                                link_colors.append(
                                    f"rgba({int(hx[0:2], 16)}, {int(hx[2:4], 16)}, {int(hx[4:6], 16)}, 0.4)")

                fig_market.add_trace(go.Sankey(node=dict(pad=20, thickness=20, label=all_regions, color=node_colors),
                                               link=dict(source=source, target=target, value=values,
                                                         color=link_colors)), row=1, col=idx + 1)

            fig_market.update_layout(title_text="Net Inter-Node CSU Market Flows" if with_titles else None,
                                     font_size=12, width=1400, height=500)
            fig_market.write_html(os.path.join(dirs["MKT"], "Figure5_Market_Flows.html"))
        except Exception:
            pass

        print(f"  -> Generated {folder_suffix} in {output_dir}")


# ==========================================
# 2. ETS Trajectory Plot
# ==========================================
def create_co2_price_plot():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # 1. Calculate the trajectories
    scenario_trajectories = {}
    years = np.arange(2026, 2051)

    for start_year in INCLINE_YEARS:
        prices = []
        current_price = 80.00
        for y in years:
            if y == 2026:
                current_price = 80.00
            elif y <= start_year:
                current_price += 4.37
            elif y <= start_year + 10:
                current_price += 20.01
            else:
                current_price += 20.52
            prices.append(round(current_price, 2))

        scenario_trajectories[start_year] = pd.Series(prices, index=years)

    # 2. Plotting Setup
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    output_trajectories = {}

    for start_year in sorted(scenario_trajectories.keys()):
        trajectory = scenario_trajectories[start_year]

        # Scenario labels
        if start_year == 2030:
            lbl = "Target ETS Pathway"
        elif start_year == 2035:
            lbl = "Near-Term Delay"
        elif start_year == 2040:
            lbl = "Mid-Term Delay"
        elif start_year == 2045:
            lbl = "Long-Term Delay"
        elif start_year == 2050:
            lbl = "Stagnant Policy"
        else:
            lbl = f"Incline {start_year}"

        col_name = lbl
        output_trajectories[col_name] = trajectory

        # Line plot without scatter markers
        ax.plot(
            trajectory.index,
            trajectory.values,
            color=INCLINE_COLORS.get(start_year, "black"),
            linestyle="-",
            linewidth=2.5 if start_year == 2030 else 1.5,
            label=lbl,
        )

    ax.set_xlabel("")
    ax.set_ylabel("EUR/tCO$_2$", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(2025, 2051)
    ax.set_xticks(range(2025, 2051, 5))
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.legend(frameon=True, edgecolor="black", fontsize=9, loc="upper left", handlelength=1.5)

    plt.tight_layout()
    plot_path = os.path.join(script_directory, "co2_price_trajectories_shifted.pdf")
    plt.savefig(plot_path, bbox_inches="tight", dpi=600)
    print(f"Saved CO2 price trajectory plot: {plot_path}")
    plt.close(fig)

    # 3. Export data as DataFrame and Excel
    df_output = pd.DataFrame(output_trajectories)
    df_output.index.name = "Year"

    print("\n" + "=" * 60)
    print("CO2 PRICE TRAJECTORIES (2025 - 2050) [EUR/tCO2]")
    print("=" * 60)
    print(df_output.loc[2026:2040].round(2).to_string())
    print("... (up to 2050 exported to Excel)")
    print("=" * 60 + "\n")

    excel_path = os.path.join(script_directory, "co2_price_trajectories_2030_2050.xlsx")
    df_output.round(2).to_excel(excel_path)
    print(f"Saved numerical data to: {excel_path}")

    return df_output


# ==========================================
# 3. Advanced Sensitivity Plot Functions
# ==========================================
def get_capture_data(scenario_name):
    res_path = get_latest_result_path(scenario_name)
    if not res_path:
        return None

    q_cap = pd.read_excel(res_path, sheet_name="q_CO2_cap")
    if "Scenario" in q_cap.columns: q_cap = q_cap.drop(columns=["Scenario"])
    q_cap.columns = ["region", "sector", "tech", "vintage", "year", "value"]

    return q_cap[q_cap["year"] <= MAX_YEAR].groupby("year")["value"].sum() / 1e6


def plot_sensitivity_group(prefix, incline_years, colors_map, title, out_filename, output_dir="figures_sensitivity"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for y in incline_years:
        scen_name = f"{prefix}_ets_incline_start_year_{y}"
        s_data = get_capture_data(scen_name)
        if s_data is not None:
            color = colors_map.get(y, 'black')
            lbl = f"ETS Low (Flat)" if y == 2050 else f"Incline {y}"
            linewidth = 2.5 if y == 2030 else 1.5
            ax.plot(s_data.index, s_data.values, color=color, linewidth=linewidth, label=lbl)

    ax.set_title(title)
    ax.set_ylabel("Total Captured CO$_2$ (MtCO$_2$)")
    ax.set_xticks(TARGET_YEARS)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, edgecolor="black", fontsize=_fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_filename), dpi=600, bbox_inches='tight')
    plt.close()


def plot_total_capture_comparison(incline_years, output_dir="figures_sensitivity"):
    os.makedirs(output_dir, exist_ok=True)

    data = []

    # 1. Gather Total Cumulative Capture Data (2026-2040)
    for y in incline_years:
        dw = get_capture_data(f"With_Club_ets_incline_start_year_{y}")
        dwo = get_capture_data(f"Without_Club_ets_incline_start_year_{y}")

        # Sum from 2026 to 2040 inclusive
        sum_w = dw[(dw.index >= 2026) & (dw.index <= 2040)].sum() if dw is not None else 0
        sum_wo = dwo[(dwo.index >= 2026) & (dwo.index <= 2040)].sum() if dwo is not None else 0

        data.append({
            "Incline": y,
            "With": sum_w,
            "Without": sum_wo
        })

    df = pd.DataFrame(data)
    if df.empty or (df["With"].sum() == 0 and df["Without"].sum() == 0):
        print("Skipping total capture plot: No capture data found.")
        return

    # Colors exactly as requested
    c_with = '#2A9D8F'  # Teal
    c_without = '#E76F51'  # Orange-Red

    # 2. Setup the single plot (no subplots)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(incline_years))
    width = 0.35

    # 3. Plot grouped bars (Z-order 3 puts them in front of the grid)
    ax.bar(x - width / 2, df["With"], width, label='w/ club',
           color=c_with, edgecolor='black', linewidth=1, zorder=3)

    ax.bar(x + width / 2, df["Without"], width, label='w/o club',
           color=c_without, edgecolor='black', linestyle='--', linewidth=1.5, zorder=3)

    # 4. Map Labels
    x_labels = []
    for y in incline_years:
        if y == 2030:
            x_labels.append("Target ETS Pathway")
        elif y == 2035:
            x_labels.append("Near-Term Delay")
        elif y == 2040:
            x_labels.append("Mid-Term Delay")
        elif y == 2045:
            x_labels.append("Long-Term Delay")
        elif y == 2050:
            x_labels.append("Stagnant Policy")
        else:
            x_labels.append(f"Inc. {y}")

    # 5. Add text labels directly on top of the bars with exact amounts
    max_val = max(df["With"].max(), df["Without"].max())
    y_offset = max_val * 0.02

    for i in range(len(df)):
        val_w = df["With"].iloc[i]
        ax.text(x[i] - width / 2, val_w + y_offset,
                f"{val_w:.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color=c_with)

        val_wo = df["Without"].iloc[i]
        ax.text(x[i] + width / 2, val_wo + y_offset,
                f"{val_wo:.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color=c_without)

        # 6. Formatting
        ax.set_ylabel("Cumulative Captured CO$_2$ (MtCO$_2$)", fontsize=_fontsize)
        ax.set_xticks(x)
        # CHANGED: ha="right" makes the rotated text align properly with the tick marks
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=_fontsize)

        # Add headroom to the Y-axis so the text labels don't get clipped by the ceiling
        ax.set_ylim(0, max_val * 1.15)
        ax.grid(True, linestyle=":", alpha=0.6, axis='y', zorder=0)

        # 7. Shared Legend (Centered below the x-axis)
        custom_handles = [
            mpatches.Patch(facecolor=c_with, edgecolor='black', label='w/ club'),
            mpatches.Patch(facecolor=c_without, edgecolor='black', linestyle='--', linewidth=1.5, label='w/o club')
        ]
        # CHANGED: Adjusted bbox_to_anchor from -0.15 to -0.30 to clear the rotated labels
        ax.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, -0.30),
                  ncol=2, frameon=True, edgecolor='black', fontsize=_fontsize)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "Total_Captured_CO2_Comparison.pdf")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  -> Saved total capture plot to: {plot_path}")

def plot_club_capture_comparison(incline_years, output_dir="figures_sensitivity"):
    os.makedirs(output_dir, exist_ok=True)
    periods = ["2026-2030", "2031-2035", "2036-2040"]
    data = []

    # 1. Gather Cumulative Capture Data
    for y in incline_years:
        dw = get_capture_data(f"With_Club_ets_incline_start_year_{y}")
        dwo = get_capture_data(f"Without_Club_ets_incline_start_year_{y}")

        for p in periods:
            y_start, y_end = int(p[:4]), int(p[5:])
            sum_w = dw[(dw.index >= y_start) & (dw.index <= y_end)].sum() if dw is not None else 0
            sum_wo = dwo[(dwo.index >= y_start) & (dwo.index <= y_end)].sum() if dwo is not None else 0
            data.append({
                "Incline": y,
                "Period": p,
                "With": sum_w,
                "Without": sum_wo,
                "Delta": sum_w - sum_wo
            })

    df = pd.DataFrame(data)
    if df.empty or df["With"].sum() == 0 and df["Without"].sum() == 0:
        print("Skipping club capture comparison: No capture data found.")
        return

    # Colors
    c_with = '#2A9D8F'  # Teal
    c_without = '#E76F51'  # Orange-Red

    # ---------------------------------------------------------
    # Figure A: Compact Bar Chart (1xN Subplots for each Scenario)
    # ---------------------------------------------------------
    num_scenarios = len(incline_years)
    fig, axes = plt.subplots(1, num_scenarios, figsize=(2.8 * num_scenarios, 4.5), sharey=True)
    if num_scenarios == 1: axes = [axes]

    x = np.arange(len(periods))
    width = 0.35

    for i, y in enumerate(incline_years):
        ax = axes[i]
        df_sub = df[df["Incline"] == y]

        # Bars
        ax.bar(x - width / 2, df_sub["With"], width, label='w/ club', color=c_with, edgecolor='black', linewidth=1)
        # Without club has a dashed edge pattern
        ax.bar(x + width / 2, df_sub["Without"], width, label='w/o club', color=c_without,
               edgecolor='black', linestyle='--', linewidth=1.5)

        # Map labels
        if y == 2030:
            lbl = "Target ETS Pathway"
        elif y == 2035:
            lbl = "Near-Term Delay"
        elif y == 2040:
            lbl = "Mid-Term Delay"
        elif y == 2045:
            lbl = "Long-Term Delay"
        elif y == 2050:
            lbl = "Stagnant Policy"
        else:
            lbl = f"Inc. {y}"

        ax.set_title(lbl, fontsize=_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45, ha="right")
        ax.grid(True, linestyle=":", alpha=0.6, axis='y')

        if i == 0:
            ax.set_ylabel("Cumulative Captured CO$_2$ (MtCO$_2$)")

    # Shared Legend below figure
    custom_handles = [
        mpatches.Patch(facecolor=c_with, edgecolor='black', label='w/ club'),
        mpatches.Patch(facecolor=c_without, edgecolor='black', linestyle='--', linewidth=1.5,
                       label='w/o club')
    ]
    fig.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
               frameon=True, edgecolor='black', fontsize=_fontsize)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, "Captured_CO2_Comparison_Bars.pdf"), dpi=600, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Figure B: Delta Diverging Bar Chart (4x1: 3 Periods + 1 Total)
    # ---------------------------------------------------------

    # 1. Calculate the total offset across 2026-2040 for each scenario
    total_data = []
    for y in incline_years:
        df_y = df[df["Incline"] == y]
        total_data.append({
            "Incline": y,
            "Period": "Total (2026-2040)",
            "With": df_y["With"].sum(),
            "Without": df_y["Without"].sum(),
            "Delta": df_y["Delta"].sum()
        })
    df_total = pd.DataFrame(total_data)
    df_extended = pd.concat([df, df_total], ignore_index=True)

    # 2. Setup a 4x1 vertical grid
    plot_periods = ["2026-2030", "2031-2035", "2036-2040", "Total (2026-2040)"]
    fig2, axes2 = plt.subplots(4, 1, figsize=(8.5, 12), sharex=True)

    # Map the better, descriptive labels
    x_labels = []
    for y in incline_years:
        if y == 2030:
            x_labels.append("Target ETS Pathway")
        elif y == 2035:
            x_labels.append("Near-Term Delay")
        elif y == 2040:
            x_labels.append("Mid-Term Delay")
        elif y == 2045:
            x_labels.append("Long-Term Delay")
        elif y == 2050:
            x_labels.append("Stagnant Policy")
        else:
            x_labels.append(str(y))

    x_pos = np.arange(len(incline_years))

    for i, p in enumerate(plot_periods):
        ax = axes2[i]
        df_sub = df_extended[df_extended["Period"] == p]

        # Determine colors: Teal for positive, Red for negative, Black for 0
        colors = [c_with if val > 0 else c_without if val < 0 else 'black' for val in df_sub["Delta"]]

        # Plot Diverging Bars
        bars = ax.bar(x_pos, df_sub["Delta"], width=0.5, color=colors, edgecolor='black', zorder=3)

        # Bold Zero Line
        ax.axhline(0, color='black', linewidth=1.5, zorder=4)

        # Auto-scale Y-axis symmetrically so the 0-line stays centered
        max_abs_val = max(abs(df_sub["Delta"].max()), abs(df_sub["Delta"].min()))
        y_lim = max_abs_val * 1.35 if max_abs_val != 0 else 1
        ax.set_ylim(-y_lim, y_lim)

        # Add exact values above/below the bars
        for bar in bars:
            val = bar.get_height()
            y_offset = y_lim * 0.08 if val >= 0 else -y_lim * 0.08

            if val == 0:
                # Black 0.0 if there is no delta
                ax.text(bar.get_x() + bar.get_width() / 2, y_offset,
                        "0.0", ha='center', va='bottom', fontsize=10,
                        color='black', fontweight='bold')
            else:
                va = 'bottom' if val > 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset,
                        f"{val:+.1f}", ha='center', va=va, fontsize=10,
                        color=c_with if val > 0 else c_without, fontweight='bold')

        # Formatting
        is_total = "Total" in p
        ax.set_title(f"{p}", fontsize=_fontsize + (2 if is_total else 0),
                     fontweight='bold' if is_total else 'normal')
        ax.set_ylabel("$\Delta$ CO$_2$\n(MtCO$_2$)")
        ax.grid(True, linestyle=":", alpha=0.6, axis='y', zorder=0)

    # Set custom X-axis labels on the bottom-most plot
    axes2[-1].set_xticks(x_pos)
    axes2[-1].set_xticklabels(x_labels, rotation=45, ha="center")

    # Shared Legend (Replacing the text boxes)
    custom_handles = [
        mpatches.Patch(facecolor=c_with, edgecolor='black', label='Higher w/ Club'),
        mpatches.Patch(facecolor=c_without, edgecolor='black', label='Higher w/o Club')
    ]
    fig2.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2,
                frameon=True, edgecolor='black', fontsize=_fontsize)

    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Slightly raised bottom rect to fit legend
    plt.savefig(os.path.join(output_dir, "Captured_CO2_Delta_Bars.pdf"), dpi=600, bbox_inches='tight')
    plt.close()

def plot_combined_sensitivity(incline_years, colors_map, output_dir="figures_sensitivity"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    for y in incline_years:
        color = colors_map.get(y, 'black')
        linewidth = 2.5 if y == 2030 else 1.5

        # With Club
        dw = get_capture_data(f"With_Club_ets_incline_start_year_{y}")
        if dw is not None:
            ax.plot(dw.index, dw.values, color=color, linestyle="-", linewidth=linewidth)

        # Without Club
        dwo = get_capture_data(f"Without_Club_ets_incline_start_year_{y}")
        if dwo is not None:
            ax.plot(dwo.index, dwo.values, color=color, linestyle="--", linewidth=linewidth)

    # Removed the title as requested
    ax.set_ylabel("Total Captured CO$_2$ (MtCO$_2$)")
    ax.set_xticks(TARGET_YEARS)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 1. Legend elements for Line Styles (Below Plot)
    style_elements = [
        mlines.Line2D([0], [0], color='black', lw=2.0, linestyle='-', label='w/ club'),
        mlines.Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='w/o club '),
    ]

    # 2. Legend elements for Colors (Inside Plot, Top Left)
    color_elements = []
    for y in incline_years:
        # Renaming scenarios based on the year
        if y == 2030:
            lbl = "Target ETS Pathway"
        elif y == 2035:
            lbl = "Near-Term Delay"
        elif y == 2040:
            lbl = "Mid-Term Delay"
        elif y == 2045:
            lbl = "Long-Term Delay"
        elif y == 2050:
            lbl = "Stagnant Policy"
        else:
            lbl = f"Inc. {y}"

        linewidth = 2.5 if y == 2030 else 1.5
        color_elements.append(
            mlines.Line2D([0], [0], color=colors_map.get(y, 'black'), lw=linewidth, label=lbl)
        )

    # Add Color Legend (Top Left)
    color_legend = ax.legend(
        handles=color_elements,
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fontsize=_fontsize
    )
    # Add it to the axes manually so the next legend call doesn't overwrite it
    ax.add_artist(color_legend)

    # Add Style Legend (Below Plot)
    ax.legend(
        handles=style_elements,
        bbox_to_anchor=(0.5, -0.12),  # Centers it below the x-axis
        loc="upper center",
        ncol=2,  # Splits items into two columns
        frameon=True,
        edgecolor="black",
        fontsize=_fontsize
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Sensitivity_Combined.pdf"), dpi=600, bbox_inches='tight')
    plt.close()


# ==========================================
# 4. Execution
# ==========================================
if __name__ == "__main__":

    print("\n--- Generating CO2 Price Trajectories ---")
    create_co2_price_plot()

    # Generate Standard Plots for all Base Cases
    for scenario in ["With_Club_ets_incline_start_year_2030", "Without_Club_ets_incline_start_year_2030"]:
        generate_standard_plots(scenario)

    print("\n--- Generating Sensitivity Plots ---")

    plot_sensitivity_group(
        prefix="With_Club",
        incline_years=INCLINE_YEARS,
        colors_map=INCLINE_COLORS,
        title="CO$_2$ Captured: Policy Delay Sensitivities (With Club)",
        out_filename="Sensitivity_With_Club.pdf"
    )

    plot_sensitivity_group(
        prefix="Without_Club",
        incline_years=INCLINE_YEARS,
        colors_map=INCLINE_COLORS,
        title="CO$_2$ Captured: Policy Delay Sensitivities (Without Club)",
        out_filename="Sensitivity_Without_Club.pdf"
    )

    plot_combined_sensitivity(INCLINE_YEARS, INCLINE_COLORS)

    # NEW CALL: Plots Figure A (Bars) and Figure B (Delta Lines)
    plot_club_capture_comparison(INCLINE_YEARS)

    plot_total_capture_comparison(INCLINE_YEARS)

    print("\nAll plotting tasks completed successfully.")