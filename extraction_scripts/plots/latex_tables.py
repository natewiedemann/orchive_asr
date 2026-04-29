import pandas as pd
from pathlib import Path

out_dir = Path("latex_tables")
out_dir.mkdir(exist_ok=True)

# -----------------------------
def topk(df, k=15):
    """Keep only most informative rows."""
    return df.head(k)

def group_summary(df):
    """Convert large tables into aggregated summaries."""
    return df.describe(include="all").transpose()

def save_latex(df, path):
    latex = df.to_latex(
        index=False,
        escape=True,
        column_format="l " * len(df.columns)
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)

# -----------------------------
files = {
    "table_1_year_summary.csv": "year_summary",
    "table_2_top_calls.csv": "top_calls",
    "table_3_top_call_each_year.csv": "yearly_calls",
    "table_4_matriline_summary.csv": "matriline_summary",
    "table_5_top_calls_per_matriline.csv": "matriline_calls",
    "table_6_diversity_by_year.csv": "diversity_year",
    "table_7_call_rates.csv": "call_rates"
}

for file, name in files.items():
    df = pd.read_csv(file)

    # -----------------------------
    # STRATEGY PER TYPE
    # -----------------------------

    if "top_calls" in name:
        df2 = topk(df, 15)

    elif "yearly" in name:
        df2 = topk(df, 15)

    elif "matriline" in name:
        df2 = topk(df, 12)

    elif "diversity" in name:
        df2 = df  # already compact

    elif len(df) > 40:
        # fallback: compress large tables
        df2 = group_summary(df)

    else:
        df2 = df

    out_file = out_dir / f"{name}.tex"
    save_latex(df2, out_file)

    print(f"[OK] {file} → {out_file}")