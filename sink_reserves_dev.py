import pandas as pd


def compute_sink_capacity_timeline(file_path):
    # 1. Load the dataset from the specified sheet
    df = pd.read_excel(file_path, sheet_name='sink_blocks')

    # 2. Filter for target regions: EU27, UK, and NO
    target_regions = ['EU27', 'UK', 'NO']
    df_filtered = df[df['region_i'].isin(target_regions)].copy()

    # 3. Sort to ensure strict sequential order (Stage 1 -> 2 -> 3) within each block
    df_filtered = df_filtered.sort_values(by=['region_i', 'block_id', 'stage_id'])

    # 4. Calculate cumulative time delay per block
    # (e.g., stage 3 incorporates stage 1 + stage 2 + stage 3 time delays)
    df_filtered['cumulative_timedelay'] = df_filtered.groupby(['region_i', 'block_id'])['timedelay_td'].cumsum()

    # 5. Calculate finish year: base year 2027 + cumulative time delay
    df_filtered['finish_year'] = (2027 + df_filtered['cumulative_timedelay']).astype(int)

    # 6. Aggregate new capacity coming online per region and finish year
    annual_new = (
        df_filtered.groupby(['region_i', 'finish_year'])['max_capacity']
        .sum()
        .reset_index(name='new_capacity')
    )

    # 7. Pivot table: rows = years, columns = regions
    pivot_new = annual_new.pivot(index='finish_year', columns='region_i', values='new_capacity').fillna(0)

    # Ensure all target regions exist as columns even if a region has 0 entries
    for reg in target_regions:
        if reg not in pivot_new.columns:
            pivot_new[reg] = 0.0
    pivot_new = pivot_new[target_regions]

    # 8. Reindex for a continuous range of years from 2027 up to max finish year (at least up to 2050)
    min_yr = int(pivot_new.index.min()) if not pivot_new.empty else 2027
    max_yr = max(2050, int(pivot_new.index.max())) if not pivot_new.empty else 2050

    full_years = range(min_yr, max_yr + 1)
    pivot_new = pivot_new.reindex(full_years).fillna(0)

    # 9. Compute cumulative capacity across years
    # (carrying forward previous values automatically if no new deployment occurs in a given year)
    cumulative_capacity_df = pivot_new.cumsum().reset_index()

    # Handle column naming conventions after reset_index()
    if 'finish_year' in cumulative_capacity_df.columns:
        cumulative_capacity_df.rename(columns={'finish_year': 'Year'}, inplace=True)
    elif 'index' in cumulative_capacity_df.columns:
        cumulative_capacity_df.rename(columns={'index': 'Year'}, inplace=True)

    return cumulative_capacity_df

# --- Execution & Export ---
df_timeline = compute_sink_capacity_timeline('scm_input_cleaned.xlsx')

# Save dataframe to Excel
output_filename = 'potential_sink_cap_plot.xlsx'
df_timeline.to_excel(output_filename, index=False)

print(f"DataFrame successfully saved to {output_filename}")
print(df_timeline.to_string())