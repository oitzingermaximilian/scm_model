import os
import glob
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
        # Search recursively just in case there's a timestamp folder
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
        inj_pivot = inj_added.pivot(index="year", columns="region", values="value").fillna(0)
        inj_pivot = inj_pivot.reindex(all_years).fillna(0)
        inj_cum = inj_pivot.cumsum().reset_index()
        soc_agg = inj_cum.melt(id_vars="year", var_name="region", value_name="cum_inj")

        cap_file_path = "potential_sink_cap_plot.xlsx"
        if os.path.exists(cap_file_path):
            cap_raw = pd.read_excel(cap_file_path)
            if "Year" in cap_raw.columns:
                cap_raw = cap_raw.rename(columns={"Year": "year"})
            elif "year" not in cap_raw.columns:
                cap_raw = cap_raw.rename(columns={cap_raw.columns[0]: "year"})
            cap_melted = cap_raw.melt(id_vars="year", var_name="region", value_name="total_cap")
            cap_melted["total_cap"] = cap_melted["total_cap"] / 1e6
            cap_df = pd.merge(soc_agg, cap_melted, on=["year", "region"], how="left")
            cap_df["total_cap"] = cap_df["total_cap"].fillna(0)

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

        # Active Capacity Plots
        try:
            q_sink_active = rename_cols(pd.read_excel(result_path, sheet_name="sink_active_cap"),
                                        ["region", "block", "year", "value"])
            q_sink_active = q_sink_active[q_sink_active["year"] <= MAX_YEAR]
            q_sink_active.loc[:, "value"] /= 1000000.0

            extended_years = sorted(set(all_years) | set(q_sink_active["year"].unique()))
            active_added = q_sink_active.groupby(["region", "year"])["value"].sum().reset_index()
            active_pivot = active_added.pivot(index="year", columns="region", values="value").fillna(0).reindex(
                extended_years).fillna(0)
            active_cum = active_pivot.cumsum().reset_index()
            active_cap_agg = active_cum.melt(id_vars="year", var_name="region", value_name="active_cap")

            inj_pivot_ext = inj_added.pivot(index="year", columns="region", values="value").fillna(0).reindex(
                extended_years).fillna(0)
            soc_agg_ext = inj_pivot_ext.cumsum().reset_index().melt(id_vars="year", var_name="region",
                                                                    value_name="cum_inj")
            df_active_soc = pd.merge(soc_agg_ext, active_cap_agg, on=["region", "year"], how="inner")

            # Figure 2b
            plt.figure(figsize=(8, 4))
            ax2b = plt.gca()
            sns.barplot(data=df_active_soc, x="year", y="cum_inj", hue="region", hue_order=DESIRED_REGION_ORDER,
                        palette=REGION_COLOR_MAP, order=extended_years, ax=ax2b)
            sns.barplot(data=df_active_soc, x="year", y="active_cap", hue="region", hue_order=DESIRED_REGION_ORDER,
                        palette=REGION_COLOR_MAP, order=extended_years, ax=ax2b)
            half = len(ax2b.patches) // 2
            for p in ax2b.patches[:half]: p.set_edgecolor('black'); p.set_linewidth(0.5)
            for p in ax2b.patches[half:]: p.set_facecolor('none'); p.set_edgecolor('black'); p.set_linewidth(0.5)
            present_regions = sort_regions(df_active_soc["region"].unique())
            legend_handles = [mpatches.Patch(facecolor=REGION_COLOR_MAP[r], edgecolor='black', linewidth=0.5, label=r)
                              for r in present_regions]
            ax2b.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                        ncol=len(present_regions), frameon=True, edgecolor='black')
            tick_pos_ext = [extended_years.index(y) for y in TARGET_YEARS if y in extended_years]
            ax2b.set_xticks(tick_pos_ext)
            ax2b.set_xticklabels([y for y in TARGET_YEARS if y in extended_years])
            if with_titles: plt.title("State of Charge (Actively Developed vs Cumulative)")
            plt.ylabel("CO$_2$ (MtCO$_2$)")
            plt.xlabel("")
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["SOC"], "Figure2b_SOC_ActiveCap_Hollow.pdf"), dpi=600, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

        # ---------------- Figure 3: CSU Balance ----------------
        try:
            regions = sort_regions(q_inj["region"].unique())
            bar_chart_data = {}
            for region in regions:
                df_gen = csu_gen[csu_gen["region"] == region].groupby("year")["value"].sum()
                df_buy = csu_buy[csu_buy["region"] == region].groupby("year")["value"].sum()
                df_sell = csu_sell[csu_sell["region"] == region].groupby("year")["value"].sum()
                df_use = csu_use[csu_use["region"] == region].groupby("year")["value"].sum()
                df_bal = pd.DataFrame({"Gen": df_gen, "Buy": df_buy, "Sell": df_sell, "Use": df_use}).fillna(0)
                df_bal = df_bal.loc[df_bal.index <= MAX_YEAR]
                if df_bal.empty: continue
                years = df_bal.index.tolist()

                # Area Chart
                plt.figure(figsize=(7, 4))
                ax_area = plt.gca()
                ax_area.fill_between(years, 0, df_bal["Use"], color=C_OB, alpha=0.15, label='Use')
                ax_area.plot(years, df_bal["Use"], color=C_OB, linestyle='--', linewidth=2.0, label='Obligation')
                ax_area.stackplot(years, df_bal["Gen"], df_bal["Buy"], labels=['Generation', 'Buy'],
                                  colors=[C_GEN, C_BUY], alpha=0.9, edgecolor='black', linewidth=0.5)
                ax_area.fill_between(years, 0, -df_bal["Sell"], color=C_SELL, alpha=0.9, label='Sell',
                                     edgecolor='black', linewidth=0.5)
                ax_area.axhline(0, color='black', linewidth=1)
                if with_titles: ax_area.set_title(f"CSU Balance: {region} (Area)")
                ax_area.set_ylabel("CSU Balance (MtCO$_2$)")
                ax_area.set_xlabel("")
                ax_area.set_xticks([y for y in TARGET_YEARS if y in years])
                ax_area.grid(True, linestyle=":", alpha=0.6, axis='y')
                handles, labels = ax_area.get_legend_handles_labels()
                desired_order = ['Obligation', 'Use', 'Generation', 'Buy', 'Sell']
                lmap = dict(zip(labels, handles))
                ax_area.legend([lmap[l] for l in desired_order if l in lmap], [l for l in desired_order if l in lmap],
                               loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=True, edgecolor='black')
                plt.tight_layout()
                plt.savefig(os.path.join(dirs["BAL"], f"Figure3_CSU_Balance_Area_{region}.pdf"), dpi=600,
                            bbox_inches='tight')
                plt.close()

                df_bal['Period'] = df_bal.index.map(get_period)
                bar_chart_data[region] = df_bal.groupby('Period').sum().reindex(
                    ['2026-2030', '2031-2035', '2036-2040']).fillna(0)

            if bar_chart_data:
                num_regions = len(bar_chart_data)
                fig_bar, axes_bar = plt.subplots(1, num_regions, figsize=(3.5 * num_regions, 4.5), sharey=False)
                if num_regions == 1: axes_bar = [axes_bar]
                max_y = max([(d["Gen"] + d["Buy"]).max() for d in bar_chart_data.values()] + [d["Use"].max() for d in
                                                                                              bar_chart_data.values()])
                min_y = min([-d["Sell"].max() for d in bar_chart_data.values()])

                for ax_bar, region in zip(axes_bar, [r for r in DESIRED_REGION_ORDER if r in bar_chart_data.keys()]):
                    df_period = bar_chart_data[region]
                    ax_bar.bar(df_period.index, df_period["Gen"], width=0.45, color=C_GEN, edgecolor='black',
                               linewidth=0.5)
                    ax_bar.bar(df_period.index, df_period["Buy"], bottom=df_period["Gen"], width=0.45, color=C_BUY,
                               edgecolor='black', linewidth=0.5)
                    ax_bar.bar(df_period.index, -df_period["Sell"], width=0.45, color=C_SELL, edgecolor='black',
                               linewidth=0.5)
                    for i, row in enumerate(df_period.itertuples()):
                        ax_bar.hlines(y=row.Use, xmin=i - 0.225, xmax=i + 0.225, color=C_OB, linewidth=2.5, zorder=5)
                    ax_bar.set_title(region)
                    ax_bar.set_ylim(min_y * 1.15 if min_y < 0 else 0, max_y * 1.15)
                    ax_bar.grid(True, linestyle=":", alpha=0.6, axis='y')

                if with_titles: fig_bar.suptitle("Cumulative CSU Balance (Up to 2040)", y=0.98, fontsize=_fontsize + 2)
                custom_handles = [mlines.Line2D([], [], color=C_OB, linewidth=2.5, label='Obligation'),
                                  mpatches.Patch(facecolor=C_GEN, edgecolor='black', label='Generation'),
                                  mpatches.Patch(facecolor=C_BUY, edgecolor='black', label='Buy'),
                                  mpatches.Patch(facecolor=C_SELL, edgecolor='black', label='Sell')]
                fig_bar.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4,
                               frameon=True, edgecolor='black')
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
            import re
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
                                source.append(node_indices[seller]);
                                target.append(node_indices[buyer]);
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
# 2. Advanced Sensitivity Plot Functions
# ==========================================
def get_capture_data(scenario_name):
    """Helper to load total captured CO2 for a specific scenario."""
    res_path = get_latest_result_path(scenario_name)
    if not res_path:
        return None

    q_cap = pd.read_excel(res_path, sheet_name="q_CO2_cap")
    if "Scenario" in q_cap.columns: q_cap = q_cap.drop(columns=["Scenario"])
    q_cap.columns = ["region", "sector", "tech", "vintage", "year", "value"]

    return q_cap[q_cap["year"] <= MAX_YEAR].groupby("year")["value"].sum() / 1e6


def plot_sensitivity_group(base_name, prefix, thetas, colors, title, out_filename, output_dir="figures_sensitivity"):
    """Plots a single group (With Club OR Without Club) in distinct colors."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Plot Base Case
    base_data = get_capture_data(base_name)
    if base_data is not None:
        ax.plot(base_data.index, base_data.values, color="black", linewidth=2.5, label=f"Base ETS Price (1.0x)")

    # Plot Sensitivities
    for t, color in zip(thetas, colors):
        scen_name = f"{prefix}_x{t}"
        s_data = get_capture_data(scen_name)
        if s_data is not None:
            ax.plot(s_data.index, s_data.values, color=color, linewidth=1.5, label=f"ETS Price x{t}")

    ax.set_title(title)
    ax.set_ylabel("Total Captured CO$_2$ (MtCO$_2$)")
    ax.set_xticks(TARGET_YEARS)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, edgecolor="black", fontsize=_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_filename), dpi=600, bbox_inches='tight')
    plt.close()


def plot_combined_sensitivity(thetas, colors, output_dir="figures_sensitivity"):
    """Plots both With Club and Without Club on the same chart using solid/dashed lines."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    # 1. Base Cases
    bw_data = get_capture_data("Base_With_Club")
    if bw_data is not None:
        ax.plot(bw_data.index, bw_data.values, color="black", linestyle="-", linewidth=3.0)

    bwo_data = get_capture_data("Base_Without_Club")
    if bwo_data is not None:
        ax.plot(bwo_data.index, bwo_data.values, color="black", linestyle="--", linewidth=3.0)

    # 2. Sensitivities
    for t, color in zip(thetas, colors):
        dw = get_capture_data(f"With_Club_ETS_x{t}")
        if dw is not None:
            ax.plot(dw.index, dw.values, color=color, linestyle="-", linewidth=1.5)

        dwo = get_capture_data(f"Without_Club_ETS_x{t}")
        if dwo is not None:
            ax.plot(dwo.index, dwo.values, color=color, linestyle="--", linewidth=1.5)

    ax.set_title("Total Captured CO$_2$: Combined Club vs No Club Sensitivities")
    ax.set_ylabel("Total Captured CO$_2$ (MtCO$_2$)")
    ax.set_xticks(TARGET_YEARS)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Custom Legend
    legend_elements = [
        mlines.Line2D([0], [0], color='black', lw=2.0, linestyle='-', label='With Club (Solid)'),
        mlines.Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='Without Club (Dashed)'),
        mpatches.Patch(color='white', label=''),  # Spacer
        mlines.Line2D([0], [0], color='black', lw=3.0, label='Base ETS (1.0x)')
    ]
    for t, color in zip(thetas, colors):
        legend_elements.append(mlines.Line2D([0], [0], color=color, lw=1.5, label=f'ETS x{t}'))

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, edgecolor="black",
              fontsize=_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Sensitivity_Combined.pdf"), dpi=600, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":

    # 1. Standard Plots
    for scenario in ["Base_With_Club", "Base_Without_Club"]:
        generate_standard_plots(scenario)

    # 2. Sensitivity Plots
    thetas = [0.25, 0.5, 0.75, 1.5, 2.0,3.0,4.0,5.0]

    # Color palette (blue to red)
    palette = sns.color_palette("coolwarm", n_colors=len(thetas))

    # Plot 1: Only With Club
    plot_sensitivity_group(
        base_name="Base_With_Club",
        prefix="With_Club_ETS",
        thetas=thetas,
        colors=palette,
        title="CO$_2$ Captured: ETS Price Sensitivities (With Club)",
        out_filename="Sensitivity_With_Club.pdf"
    )

    # Plot 2: Only Without Club
    plot_sensitivity_group(
        base_name="Base_Without_Club",
        prefix="Without_Club_ETS",
        thetas=thetas,
        colors=palette,
        title="CO$_2$ Captured: ETS Price Sensitivities (Without Club)",
        out_filename="Sensitivity_Without_Club.pdf"
    )

    # Plot 3: Combined
    plot_combined_sensitivity(thetas, palette)

    print("\nAll tasks completed successfully.")