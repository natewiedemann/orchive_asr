# ============================================
# ORCHIVE CALL TREND ANALYSIS + BEST PLOTS
# ============================================
# Generates:
# 1. Top calls overall barplot
# 2. Heatmap of top calls by year
# 3. Normalized trend lines over time
# 4. Rolling average trends
# 5. Call diversity over time
# 6. Top matrilines overall
# 7. Calls associated with top matrilines
# 8. Matriline diversity over time
# 9. Top matrilines through time
#
# INPUT FILE:
# orchive_metadata_summary.csv
#
# FIXES vs original:
# - no_speech already bool in CSV; removed fragile str->map conversion
# - no filename column; path used as row identifier throughout
# - partial dates (MM-DD, no year) excluded from date analysis
# - year_from_path used as canonical year (not the 'years' text column,
#   which sometimes contains years from spoken content that differ from
#   the file's actual recording year)
# - duplicate import block and duplicate recordings_per_year removed
# - all plots/tables show n= coverage annotations
# - graceful skip if a required dataframe is empty
# ============================================

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk")

# ----------------------------
# LOAD DATA
# ----------------------------
FILE = "orchive_metadata_summary.csv"
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

# ----------------------------
# CLEAN no_speech
# ----------------------------
# Column is already bool from extraction script; handle the edge case
# where it was written as a string "True"/"False" just in case.
if df["no_speech"].dtype == object:
    df["no_speech"] = df["no_speech"].astype(str).str.strip().str.lower() == "true"
else:
    df["no_speech"] = df["no_speech"].astype(bool)

# ----------------------------
# VOCAL ROWS ONLY
# ----------------------------
df_vocal = df[df["no_speech"] == False].copy()
n_total    = len(df)
n_vocal    = len(df_vocal)
n_silent   = n_total - n_vocal

print(f"Dataset: {n_total} rows total | {n_vocal} vocal | {n_silent} silent/no-speech")

# ----------------------------
# CANONICAL YEAR
# year_from_path is the directory year (reliable).
# The 'years' text column reflects years mentioned inside the transcript
# and can differ (e.g. a 2010 file referencing events from 2009).
# All time-series analyses use year_from_path.
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
long_calls = explode_col(df_vocal, "calls", "call")
n_call_rows = len(df_vocal[df_vocal["calls"].astype(str).str.strip().replace(["nan","","None"], np.nan).notna()])
print(f"Rows with call data: {n_call_rows} / {n_vocal} vocal")

# ----------------------------
# BUILD LONG MATRILINES DATAFRAME
# ----------------------------
long_mats = explode_col(df_vocal, "matrilines", "matriline")
n_mat_rows = len(df_vocal[df_vocal["matrilines"].astype(str).str.strip().replace(["nan","","None"], np.nan).notna()])
print(f"Rows with matriline data: {n_mat_rows} / {n_vocal} vocal")

# ----------------------------
# recordings per year (denominator for normalisation)
# based on all vocal rows, not just those with calls
# ----------------------------
recordings_per_year = df_vocal.groupby("year").size().rename("n_recordings")

# -------------------------------------------
# PLOT HELPER: annotate with data coverage
# -------------------------------------------
def coverage_note(ax, n, denom, label="recordings"):
    ax.annotate(
        f"n = {n:,} {label}",
        xy=(0.01, 0.98), xycoords="axes fraction",
        va="top", ha="left", fontsize=10,
        color="gray"
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

if skip_if_empty(long_calls, "plot_1"):
    pass
else:
    top_n = 15
    top_calls = long_calls["call"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=top_calls.values, y=top_calls.index, hue=top_calls.index, palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Calls Overall")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Call Type")
    coverage_note(ax, n_call_rows, n_vocal, "recordings with call data")
    plt.tight_layout()
    plt.savefig("plot_1_top_calls_barplot.png", dpi=300)
    plt.close()
    print("  Saved plot_1_top_calls_barplot.png")

# ============================================
# PLOT 2: HEATMAP OF TOP CALLS BY YEAR
# ============================================
print("Plot 2: heatmap calls by year …")

if skip_if_empty(long_calls, "plot_2"):
    pass
else:
    top_call_names = long_calls["call"].value_counts().head(15).index.tolist()

    heat = (
        long_calls[long_calls["call"].isin(top_call_names)]
        .groupby(["call", "year"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heat, cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Top Calls by Year  (n={n_call_rows:,} recordings with call data)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Call")
    plt.tight_layout()
    plt.savefig("plot_2_heatmap_calls_by_year.png", dpi=300)
    plt.close()
    print("  Saved plot_2_heatmap_calls_by_year.png")

# ============================================
# PLOT 3: NORMALIZED CALL RATE OVER TIME
# Normalised by number of vocal recordings per year
# (not total recordings, so silent years don't dilute rates)
# ============================================
print("Plot 3: normalized call frequency …")

if skip_if_empty(long_calls, "plot_3"):
    pass
else:
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
    sns.lineplot(data=norm, x="year", y="rate", hue="call", marker="o", ax=ax)
    ax.set_title(f"Normalized Call Frequency Over Time  (per vocal recording)")
    ax.set_ylabel("Calls per Vocal Recording")
    ax.set_xlabel("Year")
    coverage_note(ax, n_call_rows, n_vocal, "recordings with call data")
    plt.tight_layout()
    plt.savefig("plot_3_normalized_trends.png", dpi=300)
    plt.close()
    print("  Saved plot_3_normalized_trends.png")

# ============================================
# PLOT 4: ROLLING 3-YEAR AVERAGE
# ============================================
print("Plot 4: rolling average …")

if skip_if_empty(long_calls, "plot_4"):
    pass
else:
    pivot = norm.pivot(index="year", columns="call", values="rate").fillna(0)

    # Only makes sense with 3+ years
    if len(pivot) < 3:
        print(f"  NOTE: only {len(pivot)} years present; rolling(3) will repeat edge values.")

    roll = pivot.rolling(window=3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 8))
    for col in roll.columns:
        ax.plot(roll.index, roll[col], marker="o", label=col)
    ax.set_title(
        f"3-Year Rolling Average of Call Rates"
        + (f"  (NOTE: only {len(pivot)} years in data)" if len(pivot) < 3 else "")
    )
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

if skip_if_empty(long_calls, "plot_5"):
    pass
else:
    diversity = (
        long_calls.groupby("year")["call"]
        .nunique()
        .reset_index(name="unique_calls")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=diversity, x="year", y="unique_calls", marker="o", linewidth=3, ax=ax)
    ax.set_title(f"Call Diversity Over Time  (n={n_call_rows:,} recordings with call data)")
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

if skip_if_empty(long_mats, "plot_6"):
    pass
else:
    top_mats = long_mats["matriline"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=top_mats.values, y=top_mats.index, hue=top_mats.index, palette="magma", legend=False, ax=ax)
    ax.set_title(f"Top Matrilines in Recordings")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Matriline")
    coverage_note(ax, n_mat_rows, n_vocal, "recordings with matriline data")
    plt.tight_layout()
    plt.savefig("plot_6_top_matrilines.png", dpi=300)
    plt.close()
    print("  Saved plot_6_top_matrilines.png")

# ============================================
# PLOT 7: CALLS × MATRILINES HEATMAP
# Requires both calls and matrilines on the same row.
# Cross-explode: one row per (call, matriline) pair per recording.
# ============================================
print("Plot 7: calls by matriline heatmap …")

df_both = df_vocal.copy()
df_both["calls"]      = df_both["calls"].astype(str).str.strip().replace(["nan","","None"], np.nan)
df_both["matrilines"] = df_both["matrilines"].astype(str).str.strip().replace(["nan","","None"], np.nan)
df_both = df_both.dropna(subset=["calls", "matrilines"])

if df_both.empty:
    print("  SKIPPED plot_7: no rows have both calls AND matrilines.")
else:
    df_both["call_list"] = df_both["calls"].str.split(";").apply(lambda x: [v.strip() for v in x if v.strip()])
    df_both["mat_list"]  = df_both["matrilines"].str.split(";").apply(lambda x: [v.strip() for v in x if v.strip()])

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
        sns.heatmap(pivot7, cmap="Blues", annot=True, fmt=".0f", ax=ax)
        ax.set_title(f"Call Usage by Top 5 Matrilines  (n={len(df_both):,} recordings with both)")
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

if skip_if_empty(long_mats, "plot_8"):
    pass
else:
    mat_div = (
        long_mats.groupby("year")["matriline"]
        .nunique()
        .reset_index(name="unique_matrilines")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=mat_div, x="year", y="unique_matrilines", marker="o", linewidth=3, ax=ax)
    ax.set_title(f"Matriline Diversity Over Time  (n={n_mat_rows:,} recordings with matriline data)")
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

if skip_if_empty(long_mats, "plot_9"):
    pass
else:
    top5_mats  = long_mats["matriline"].value_counts().head(5).index.tolist()
    year_mat   = (
        long_mats[long_mats["matriline"].isin(top5_mats)]
        .groupby(["year", "matriline"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=year_mat, x="year", y="count", hue="matriline", marker="o", ax=ax)
    ax.set_title(f"Top 5 Matrilines Over Time  (n={n_mat_rows:,} recordings with matriline data)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig("plot_9_matrilines_over_time.png", dpi=300)
    plt.close()
    print("  Saved plot_9_matrilines_over_time.png")

# ============================================
# SUMMARY TABLES
# ============================================
print("\n--- Summary tables ---")

# TABLE 1: DATASET OVERVIEW BY YEAR
year_summary = (
    df.groupby("year_from_path")
    .agg(
        recordings   = ("path", "count"),
        vocal        = ("no_speech", lambda x: (x == False).sum()),
        silent       = ("no_speech", lambda x: (x == True).sum()),
    )
    .reset_index()
)
year_summary["pct_vocal"] = (year_summary["vocal"] / year_summary["recordings"]).round(3)
year_summary.to_csv("table_1_year_summary.csv", index=False)
print("table_1_year_summary.csv")

# TABLE 2: TOP CALLS OVERALL
if not long_calls.empty:
    call_summary = long_calls["call"].value_counts().reset_index()
    call_summary.columns = ["call", "count"]
    call_summary["pct_of_all_calls"] = (call_summary["count"] / call_summary["count"].sum()).round(4)
    call_summary.to_csv("table_2_top_calls.csv", index=False)
    print("table_2_top_calls.csv")

# TABLE 3: TOP CALL PER YEAR
if not long_calls.empty:
    call_year = long_calls.groupby(["year", "call"]).size().reset_index(name="count")
    idx = call_year.groupby("year")["count"].idxmax()
    top_call_each_year = call_year.loc[idx].sort_values("year")
    top_call_each_year.to_csv("table_3_top_call_each_year.csv", index=False)
    print("table_3_top_call_each_year.csv")

# TABLE 4: MATRILINE SUMMARY
if not long_mats.empty:
    mat_summary = long_mats["matriline"].value_counts().reset_index()
    mat_summary.columns = ["matriline", "count"]
    mat_summary["pct_of_all_matriline_mentions"] = (mat_summary["count"] / mat_summary["count"].sum()).round(4)
    mat_summary.to_csv("table_4_matriline_summary.csv", index=False)
    print("table_4_matriline_summary.csv")

# TABLE 5: TOP CALLS PER MATRILINE
if not long_mats.empty and not long_calls.empty and not df_both.empty:
    call_mat = (
        mat_long2.groupby(["matriline", "call"])
        .size()
        .reset_index(name="count")
        .sort_values(["matriline", "count"], ascending=[True, False])
    )
    top3_calls_per_mat = call_mat.groupby("matriline").head(3)
    top3_calls_per_mat.to_csv("table_5_top_calls_per_matriline.csv", index=False)
    print("table_5_top_calls_per_matriline.csv")

# TABLE 6: DIVERSITY BY YEAR
div_summary = (
    df_vocal.groupby("year")
    .agg(total_recordings=("path", "count"))
    .reset_index()
)
if not long_calls.empty:
    call_div = long_calls.groupby("year").agg(unique_calls=("call","nunique"), total_calls=("call","count")).reset_index()
    div_summary = div_summary.merge(call_div, on="year", how="left")
if not long_mats.empty:
    mat_div2 = long_mats.groupby("year").agg(unique_matrilines=("matriline","nunique")).reset_index()
    div_summary = div_summary.merge(mat_div2, on="year", how="left")
div_summary.to_csv("table_6_diversity_by_year.csv", index=False)
print("table_6_diversity_by_year.csv")

# TABLE 7: CALL RATE NORMALIZED BY YEAR
if not long_calls.empty:
    call_rates = (
        long_calls.groupby(["year", "call"])
        .size()
        .reset_index(name="count")
        .merge(recordings_per_year, left_on="year", right_index=True)
    )
    call_rates["calls_per_recording"] = (call_rates["count"] / call_rates["n_recordings"]).round(4)
    call_rates.to_csv("table_7_call_rates.csv", index=False)
    print("table_7_call_rates.csv")

print("\nDone.")