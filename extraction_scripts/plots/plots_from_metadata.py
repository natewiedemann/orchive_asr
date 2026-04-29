# ============================================
# ORCHIVE CALL TREND ANALYSIS + BEST PLOTS
# ============================================
# Generates:
# 1. Top calls overall barplot
# 2. Call occurrence heatmap across years
# 3. Normalized trend lines over time
# 4. Rolling average trends
# 5. Call diversity over time
# 6. Top matrilines overall
# 7. Call distribution across matrilines
# 8. Matriline diversity over time
# 9. Top matrilines through time
#
# INPUT FILE:
# orchive_metadata_summary.csv
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set(style="whitegrid", context="talk")

# ----------------------------
# SHARED COLOR PALETTE
# All plots draw from the same palette family so figures are visually consistent.
#   CMAP_HEATMAP : colormap for both heatmaps (plots 2 & 7)
#   PAL_BAR      : sequential bar palette — same family as heatmap
#   PAL_LINE     : qualitative palette for multi-series line/scatter plots
# ----------------------------
CMAP_HEATMAP = "YlOrRd"
PAL_BAR      = "YlOrRd"
PAL_LINE     = "tab10"

# ----------------------------
# LOAD DATA
# ----------------------------
FILE = "orchive_metadata_summary.csv"
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

# ----------------------------
# CLEAN no_speech
# ----------------------------
if df["no_speech"].dtype == object:
    df["no_speech"] = df["no_speech"].astype(str).str.strip().str.lower() == "true"
else:
    df["no_speech"] = df["no_speech"].astype(bool)

# ----------------------------
# VOCAL ROWS ONLY
# ----------------------------
df_vocal  = df[df["no_speech"] == False].copy()
n_total   = len(df)
n_vocal   = len(df_vocal)
n_silent  = n_total - n_vocal

print(f"Dataset: {n_total:,} rows total | {n_vocal:,} vocal | {n_silent:,} silent/no-speech")

# ----------------------------
# CANONICAL YEAR
# year_from_path is the directory year (reliable).
# The 'years' text column reflects years *mentioned* in the transcript
# and can differ from the actual recording year.
# ----------------------------
df_vocal["year"] = pd.to_numeric(df_vocal["year_from_path"], errors="coerce")
df_vocal = df_vocal.dropna(subset=["year"])
df_vocal["year"] = df_vocal["year"].astype(int)

# ----------------------------
# HELPER: clean + explode a semicolon-delimited column
# ----------------------------
def explode_col(frame, col, new_col):
    """Return a long-form dataframe with one value per row."""
    tmp = frame.copy()
    tmp[col] = tmp[col].astype(str).str.strip()
    tmp[col] = tmp[col].replace(["nan", "", "None"], np.nan)
    tmp = tmp.dropna(subset=[col])
    tmp[new_col] = tmp[col].str.split(";").apply(
        lambda lst: [v.strip() for v in lst if v.strip()]
    )
    return tmp.explode(new_col).assign(**{new_col: lambda d: d[new_col].str.strip()})

# ----------------------------
# BUILD LONG CALLS DATAFRAME
# ----------------------------
long_calls  = explode_col(df_vocal, "calls", "call")
n_call_rows = len(df_vocal[
    df_vocal["calls"].astype(str).str.strip().replace(["nan","","None"], np.nan).notna()
])
print(f"Rows with call data:      {n_call_rows:,} / {n_vocal:,} vocal")

# ----------------------------
# BUILD LONG MATRILINES DATAFRAME
# ----------------------------
long_mats  = explode_col(df_vocal, "matrilines", "matriline")
n_mat_rows = len(df_vocal[
    df_vocal["matrilines"].astype(str).str.strip().replace(["nan","","None"], np.nan).notna()
])
print(f"Rows with matriline data: {n_mat_rows:,} / {n_vocal:,} vocal")

# ----------------------------
# Recordings per year (denominator for normalisation)
# Based on all vocal rows, not just those with calls.
# ----------------------------
recordings_per_year = df_vocal.groupby("year").size().rename("n_recordings")

# ----------------------------
# SHARED BAR COLOR HELPER
# Draws bars shaded light→dark using PAL_BAR, mirroring the heatmaps.
# ----------------------------
def bar_colors(n):
    cmap = plt.get_cmap(PAL_BAR)
    return [cmap(0.25 + 0.6 * i / max(n - 1, 1)) for i in range(n)]

# ----------------------------
# PLOT HELPERS
# ----------------------------
def coverage_note(ax, n, label="recordings"):
    ax.annotate(
        f"n = {n:,} {label}",
        xy=(0.01, 0.98), xycoords="axes fraction",
        va="top", ha="left", fontsize=10, color="gray"
    )

def skip_if_empty(long_df, plot_name):
    if long_df.empty:
        print(f"  SKIPPED {plot_name}: no data after cleaning.")
        return True
    return False

# ============================================
# PLOT 1: TOP CALLS OVERALL
# ============================================
print("\nPlot 1: top calls overall …")

if not skip_if_empty(long_calls, "plot_1"):
    top_n     = 15
    top_calls = long_calls["call"].value_counts().head(top_n)
    colors    = bar_colors(len(top_calls))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top_calls.index[::-1], top_calls.values[::-1], color=colors[::-1])
    #ax.set_title(f"Top {top_n} Calls Overall")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Call Type")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    coverage_note(ax, n_call_rows, "recordings with call data")
    plt.tight_layout()
    plt.savefig("plot_1_top_calls_barplot.png", dpi=300)
    plt.close()
    print("  Saved plot_1_top_calls_barplot.png")

# ============================================
# PLOT 2: CALL OCCURRENCE HEATMAP ACROSS YEARS
# ============================================
print("Plot 2: call occurrence heatmap across years …")

if not skip_if_empty(long_calls, "plot_2"):
    top_call_names = long_calls["call"].value_counts().head(15).index.tolist()

    heat = (
        long_calls[long_calls["call"].isin(top_call_names)]
        .groupby(["call", "year"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heat, cmap=CMAP_HEATMAP, linewidths=0.5, ax=ax)
    #ax.set_title(
    #    f"Call Occurrence Heatmap Across Years  (n={n_call_rows:,} recordings with call data)"
    #)
    ax.set_xlabel("Year")
    ax.set_ylabel("Call")
    plt.tight_layout()
    plt.savefig("plot_2_heatmap_calls_by_year.png", dpi=300)
    plt.close()
    print("  Saved plot_2_heatmap_calls_by_year.png")

# ============================================
# PLOT 3: NORMALIZED CALL RATE OVER TIME
# ============================================
print("Plot 3: normalized call frequency …")

if not skip_if_empty(long_calls, "plot_3"):
    top_call_names = long_calls["call"].value_counts().head(15).index.tolist()

    yearly_counts = (
        long_calls[long_calls["call"].isin(top_call_names)]
        .groupby(["year", "call"])
        .size()
        .reset_index(name="count")
    )

    norm = yearly_counts.merge(recordings_per_year, left_on="year", right_index=True)
    norm["rate"] = norm["count"] / norm["n_recordings"]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=norm, x="year", y="rate", hue="call",
                 marker="o", palette=PAL_LINE, ax=ax)
    #ax.set_title("Normalized Call Frequency Over Time  (per vocal recording)")
    ax.set_ylabel("Calls Referenced per Speech Recording")
    ax.set_xlabel("Year")
    coverage_note(ax, n_call_rows, "recordings with call data")
    plt.tight_layout()
    plt.savefig("plot_3_normalized_trends.png", dpi=300)
    plt.close()
    print("  Saved plot_3_normalized_trends.png")

# ============================================
# PLOT 4: ROLLING 3-YEAR AVERAGE
# ============================================
print("Plot 4: rolling average …")

if not skip_if_empty(long_calls, "plot_4"):
    pivot = norm.pivot(index="year", columns="call", values="rate").fillna(0)

    if len(pivot) < 3:
        print(f"  NOTE: only {len(pivot)} years present; rolling(3) will repeat edge values.")

    roll     = pivot.rolling(window=3, min_periods=1).mean()
    cmap_q   = plt.get_cmap(PAL_LINE)
    n_cols   = len(roll.columns)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, col in enumerate(roll.columns):
        ax.plot(roll.index, roll[col], marker="o",
                color=cmap_q(i / max(n_cols - 1, 1)), label=col)

    title_note = f"  (NOTE: only {len(pivot)} years in data)" if len(pivot) < 3 else ""
    #ax.set_title(f"3-Year Rolling Average of Call Rates{title_note}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Smoothed Rate")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_4_rolling_average.png", dpi=300)
    plt.close()
    print("  Saved plot_4_rolling_average.png")

# ============================================
# PLOT 5: CALL DIVERSITY OVER TIME
# ============================================
print("Plot 5: call diversity …")

if not skip_if_empty(long_calls, "plot_5"):
    diversity = (
        long_calls.groupby("year")["call"]
        .nunique()
        .reset_index(name="unique_calls")
    )

    accent = plt.get_cmap(CMAP_HEATMAP)(0.65)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(diversity["year"], diversity["unique_calls"],
            marker="o", linewidth=3, color=accent)
    #ax.set_title(f"Call Diversity Over Time  (n={n_call_rows:,} recordings with call data)")
    ax.set_ylabel("Unique Call Types")
    ax.set_xlabel("Year")
    plt.tight_layout()
    plt.savefig("plot_5_call_diversity.png", dpi=300)
    plt.close()
    print("  Saved plot_5_call_diversity.png")

# ============================================
# PLOT 6: TOP MATRILINES OVERALL
# ============================================
print("\nPlot 6: top matrilines overall …")

if not skip_if_empty(long_mats, "plot_6"):
    top_mats = long_mats["matriline"].value_counts().head(15)
    colors   = bar_colors(len(top_mats))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top_mats.index[::-1], top_mats.values[::-1], color=colors[::-1])
    #ax.set_title("Top Matrilines in Recordings")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Matriline")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    coverage_note(ax, n_mat_rows, "recordings with matriline data")
    plt.tight_layout()
    plt.savefig("plot_6_top_matrilines.png", dpi=300)
    plt.close()
    print("  Saved plot_6_top_matrilines.png")

# ============================================
# PLOT 7: CALL DISTRIBUTION ACROSS MATRILINES
# ============================================
print("Plot 7: call distribution across matrilines …")

df_both = df_vocal.copy()
df_both["calls"]      = df_both["calls"].astype(str).str.strip().replace(["nan","","None"], np.nan)
df_both["matrilines"] = df_both["matrilines"].astype(str).str.strip().replace(["nan","","None"], np.nan)
df_both = df_both.dropna(subset=["calls", "matrilines"])
mat_long2 = pd.DataFrame()   # initialise so tables section can safely check it

if df_both.empty:
    print("  SKIPPED plot_7: no rows have both calls AND matrilines.")
else:
    df_both["call_list"] = df_both["calls"].str.split(";").apply(
        lambda x: [v.strip() for v in x if v.strip()]
    )
    df_both["mat_list"] = df_both["matrilines"].str.split(";").apply(
        lambda x: [v.strip() for v in x if v.strip()]
    )

    mat_long2 = df_both.explode("call_list").explode("mat_list")
    mat_long2 = mat_long2.rename(columns={"call_list": "call", "mat_list": "matriline"})

    top5_mats      = long_mats["matriline"].value_counts().head(5).index.tolist()
    top8_calls_mat = long_calls["call"].value_counts().head(8).index.tolist()

    mat_call = (
        mat_long2[
            mat_long2["matriline"].isin(top5_mats) &
            mat_long2["call"].isin(top8_calls_mat)
        ]
        .groupby(["matriline", "call"])
        .size()
        .reset_index(name="count")
    )

    if mat_call.empty:
        print("  SKIPPED plot_7: no overlap between top matrilines and top calls.")
    else:
        pivot7 = mat_call.pivot(index="matriline", columns="call", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(pivot7, cmap=CMAP_HEATMAP, annot=True, fmt=".0f", ax=ax)
        #ax.set_title(
        #    f"Call Distribution Across Matrilines  (n={len(df_both):,} recordings with both)"
        #)
        ax.set_xlabel("Call")
        ax.set_ylabel("Matriline")
        plt.tight_layout()
        plt.savefig("plot_7_calls_by_matriline_heatmap.png", dpi=300)
        plt.close()
        print("  Saved plot_7_calls_by_matriline_heatmap.png")

# ============================================
# PLOT 8: MATRILINE DIVERSITY OVER TIME
# ============================================
print("Plot 8: matriline diversity …")

if not skip_if_empty(long_mats, "plot_8"):
    mat_div = (
        long_mats.groupby("year")["matriline"]
        .nunique()
        .reset_index(name="unique_matrilines")
    )

    accent = plt.get_cmap(CMAP_HEATMAP)(0.65)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mat_div["year"], mat_div["unique_matrilines"],
            marker="o", linewidth=3, color=accent)
    #ax.set_title(
    #    f"Matriline Diversity Over Time  (n={n_mat_rows:,} recordings with matriline data)"
    #)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique Matrilines")
    plt.tight_layout()
    plt.savefig("plot_8_matriline_diversity.png", dpi=300)
    plt.close()
    print("  Saved plot_8_matriline_diversity.png")

# ============================================
# PLOT 9: TOP MATRILINES THROUGH TIME
# ============================================
print("Plot 9: top matrilines over time …")

if not skip_if_empty(long_mats, "plot_9"):
    top5_mats = long_mats["matriline"].value_counts().head(5).index.tolist()
    year_mat  = (
        long_mats[long_mats["matriline"].isin(top5_mats)]
        .groupby(["year", "matriline"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=year_mat, x="year", y="count", hue="matriline",
                 marker="o", palette=PAL_LINE, ax=ax)
    #ax.set_title(
    #    f"Top 5 Matrilines Over Time  (n={n_mat_rows:,} recordings with matriline data)"
    #)
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrences")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig("plot_9_matrilines_over_time.png", dpi=300)
    plt.close()
    print("  Saved plot_9_matrilines_over_time.png")

# ============================================
# TABLE FORMATTING HELPERS
# ============================================

def fmt_pct(series):
    """Convert 0–1 fraction to a percentage string with 2 decimal places, e.g. '12.34%'."""
    return (series * 100).round(2).map(lambda v: f"{v:.2f}%")

def fmt_int(series):
    """Format a numeric series as integers with thousands comma separator."""
    return series.fillna(0).astype(int).map(lambda v: f"{v:,}")

def clean_table(df_t):
    """Replace any remaining NaN / 'nan' values with an en-dash."""
    return df_t.fillna("–").replace("nan", "–")

# ============================================
# SUMMARY TABLES
# ============================================
print("\n--- Summary tables ---")

# TABLE 1: DATASET OVERVIEW BY YEAR
year_summary = (
    df.groupby("year_from_path")
    .agg(
        recordings = ("path", "count"),
        vocal      = ("no_speech", lambda x: (x == False).sum()),
        silent     = ("no_speech", lambda x: (x == True).sum()),
    )
    .reset_index()
    .rename(columns={"year_from_path": "Year"})
)
year_summary["% Speech"]   = fmt_pct(year_summary["vocal"] / year_summary["recordings"])
year_summary["recordings"] = fmt_int(year_summary["recordings"])
year_summary["vocal"]      = fmt_int(year_summary["vocal"])
year_summary["silent"]     = fmt_int(year_summary["silent"])
clean_table(year_summary).to_csv("table_1_year_summary.csv", index=False)
print("table_1_year_summary.csv")

# TABLE 2: TOP CALLS OVERALL
if not long_calls.empty:
    call_summary = long_calls["call"].value_counts().reset_index()
    call_summary.columns = ["call", "count"]
    call_summary["% of all calls"] = fmt_pct(call_summary["count"] / call_summary["count"].sum())
    call_summary["count"]          = fmt_int(call_summary["count"])
    clean_table(call_summary).to_csv("table_2_top_calls.csv", index=False)
    print("table_2_top_calls.csv")

# TABLE 3: TOP CALL PER YEAR
if not long_calls.empty:
    call_year = long_calls.groupby(["year", "call"]).size().reset_index(name="count")
    idx = call_year.groupby("year")["count"].idxmax()
    top_call_each_year = (
        call_year.loc[idx]
        .sort_values("year")
        .rename(columns={"year": "Year"})
    )
    top_call_each_year["count"] = fmt_int(top_call_each_year["count"])
    clean_table(top_call_each_year).to_csv("table_3_top_call_each_year.csv", index=False)
    print("table_3_top_call_each_year.csv")

# TABLE 4: MATRILINE SUMMARY
if not long_mats.empty:
    mat_summary = long_mats["matriline"].value_counts().reset_index()
    mat_summary.columns = ["matriline", "count"]
    mat_summary["% of all matriline mentions"] = fmt_pct(
        mat_summary["count"] / mat_summary["count"].sum()
    )
    mat_summary["count"] = fmt_int(mat_summary["count"])
    clean_table(mat_summary).to_csv("table_4_matriline_summary.csv", index=False)
    print("table_4_matriline_summary.csv")

# TABLE 5: TOP CALLS PER MATRILINE
if not mat_long2.empty:
    call_mat = (
        mat_long2.groupby(["matriline", "call"])
        .size()
        .reset_index(name="count")
        .sort_values(["matriline", "count"], ascending=[True, False])
    )
    top3_calls_per_mat = call_mat.groupby("matriline").head(3).copy()
    top3_calls_per_mat["count"] = fmt_int(top3_calls_per_mat["count"])
    clean_table(top3_calls_per_mat).to_csv("table_5_top_calls_per_matriline.csv", index=False)
    print("table_5_top_calls_per_matriline.csv")

# TABLE 6: DIVERSITY BY YEAR
div_summary = (
    df_vocal.groupby("year")
    .agg(total_recordings=("path", "count"))
    .reset_index()
    .rename(columns={"year": "Year"})
)
if not long_calls.empty:
    call_div = (
        long_calls.groupby("year")
        .agg(unique_calls=("call", "nunique"), total_calls=("call", "count"))
        .reset_index()
        .rename(columns={"year": "Year"})
    )
    div_summary = div_summary.merge(call_div, on="Year", how="left")
if not long_mats.empty:
    mat_div2 = (
        long_mats.groupby("year")
        .agg(unique_matrilines=("matriline", "nunique"))
        .reset_index()
        .rename(columns={"year": "Year"})
    )
    div_summary = div_summary.merge(mat_div2, on="Year", how="left")

# All numeric columns as integers with comma thousands; no decimals
for col in div_summary.columns:
    if col != "Year":
        div_summary[col] = fmt_int(div_summary[col])

clean_table(div_summary).to_csv("table_6_diversity_by_year.csv", index=False)
print("table_6_diversity_by_year.csv")

# TABLE 7: CALL RATE NORMALIZED BY YEAR
if not long_calls.empty:
    call_rates = (
        long_calls.groupby(["year", "call"])
        .size()
        .reset_index(name="count")
        .merge(recordings_per_year, left_on="year", right_index=True)
        .rename(columns={"year": "Year"})
    )
    call_rates["calls_per_recording"] = (
        call_rates["count"] / call_rates["n_recordings"]
    ).round(4)
    call_rates["count"]        = fmt_int(call_rates["count"])
    call_rates["n_recordings"] = fmt_int(call_rates["n_recordings"])
    clean_table(call_rates).to_csv("table_7_call_rates.csv", index=False)
    print("table_7_call_rates.csv")

print("\nDone.")