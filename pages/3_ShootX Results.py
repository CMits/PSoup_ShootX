# pages/4_📊_Analyze_Results.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Analyze Results • ShootX", page_icon="📊", layout="wide")

DATA_DIR = Path("data")

# -----------------------------
# Helpers
# -----------------------------
def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names once
    df.columns = [c.strip() for c in df.columns]
    # Required columns
    needed = {"Hormones.Sustained_growth", "Hormones.SUC", "Gene.name", "Gene.kind"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Try to keep an explicit SUC band if provided, otherwise derive later
    if "Gene.SUC_band" not in df.columns:
        df["Gene.SUC_band"] = np.nan
    return df

def add_suc_band(df: pd.DataFrame, col="Hormones.SUC") -> pd.DataFrame:
    # Map exact levels if present; otherwise bucket to nearest of the 5 levels.
    target_levels = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    labels = ["very low", "low", "normal", "high", "very high"]
    if df["Gene.SUC_band"].notna().any():
        # Standardize order
        df["Gene.SUC_band"] = pd.Categorical(
            df["Gene.SUC_band"],
            categories=labels,
            ordered=True
        )
        return df
    # Derive from numeric SUC
    if col not in df.columns:
        return df
    # Assign nearest label
    def nearest_label(x):
        idx = (np.abs(target_levels - float(x))).argmin()
        return labels[idx]
    df["Gene.SUC_band"] = df[col].apply(nearest_label)
    df["Gene.SUC_band"] = pd.Categorical(df["Gene.SUC_band"], categories=labels, ordered=True)
    return df

def compute_bins(df: pd.DataFrame, eps: float = 1e-6):
    """
    Adds:
      - diff_vs_wt, bin_vs_wt
      - diff_vs_self1, bin_vs_self1   (baseline = same Gene.name at SUC = 1.0)
    Bins: Higher(+1)/Same(0)/Lower(-1) based on sign of diff with tolerance eps.
    """
    df = df.copy()
    sg = "Hormones.Sustained_growth"
    suc = "Hormones.SUC"
    name = "Gene.name"
    kind = "Gene.kind"

    # --- WT baseline at each SUC
    wt = df[df[kind] == "WT"][[suc, sg]].rename(columns={sg: "WT_SG"})
    df = df.merge(wt, on=suc, how="left")
    df["diff_vs_wt"] = df[sg] - df["WT_SG"]
    df["bin_vs_wt"] = np.where(
        df["diff_vs_wt"] > eps, 1,
        np.where(df["diff_vs_wt"] < -eps, -1, 0)
    )

    # --- self baseline at SUC = 1.0
    base = df[np.isclose(df[suc], 1.0)][[name, sg]].rename(columns={sg: "Self1_SG"})
    df = df.merge(base, on=name, how="left")
    df["diff_vs_self1"] = df[sg] - df["Self1_SG"]
    df["bin_vs_self1"] = np.where(
        df["diff_vs_self1"] > eps, 1,
        np.where(df["diff_vs_self1"] < -eps, -1, 0)
    )
    return df

def bin_label_map():
    return {-1: "Lower (-1)", 0: "Same (0)", 1: "Higher (+1)"}

# -----------------------------
# UI • Header + file picker
# -----------------------------
st.title("📊 Analyze Results")
st.caption("Explore **Hormones.Sustained_growth** bins by comparing genotypes vs **WT at the same sucrose** (default) or vs **themselves at SUC=1.0** (+Suc effect).")

# Find CSVs in /data
candidates = sorted([p for p in DATA_DIR.glob("*.csv")])
if not candidates:
    st.error("No CSV files found in `data/`. Put your PSoup results CSV there.")
    st.stop()

file_choice = st.selectbox("Results file in `data/`", options=[p.name for p in candidates], index=0)
path = DATA_DIR / file_choice

try:
    df = load_results(path)
except Exception as e:
    st.error(str(e))
    st.stop()

df = add_suc_band(df, col="Hormones.SUC")

# -----------------------------
# Controls
# -----------------------------
with st.container(border=True):
    left, mid, right = st.columns([1.6, 1, 1.4])
    with left:
        kinds = sorted(df["Gene.kind"].dropna().unique().tolist())
        default_kinds = [k for k in kinds if k != "WT"]
        sel_kinds = st.multiselect("Gene kinds", kinds, default=default_kinds)
        name_filter = st.text_input("Filter `Gene.name` (substring, optional)", "")
        hide_wt = st.checkbox("Hide WT in plots", value=True)
    with mid:
        suc_levels = sorted(df["Hormones.SUC"].dropna().unique().tolist())
        # keep the canonical ones first if present
        default_suc = [x for x in [0.1, 0.5, 1.0, 1.5, 2.0] if x in suc_levels]
        sel_suc = st.multiselect("SUC levels", suc_levels, default=default_suc or suc_levels)
        eps = st.number_input("Bin tolerance (ε)", min_value=0.0, value=0.05, step=1e-6, format="%.6f")
    with right:
        mode = st.radio(
            "Binning mode",
            ["vs WT at same SUC", "vs self at SUC=1.0"],
            horizontal=False,
            index=0
        )
        which_bin = "bin_vs_wt" if mode == "vs WT at same SUC" else "bin_vs_self1"
        which_diff = "diff_vs_wt" if mode == "vs WT at same SUC" else "diff_vs_self1"

# Recompute bins with chosen tolerance
df_b = compute_bins(df, eps=eps)

# Apply filters
mask = df_b["Hormones.SUC"].isin(sel_suc)
if sel_kinds:
    mask &= df_b["Gene.kind"].isin(sel_kinds) | (~df_b["Gene.kind"].notna() & False)
if name_filter.strip():
    s = name_filter.strip().lower()
    mask &= df_b["Gene.name"].astype(str).str.lower().str.contains(s, na=False)
if hide_wt:
    mask &= df_b["Gene.kind"] != "WT"

dff = df_b.loc[mask].copy()

# -----------------------------
# Summary header
# -----------------------------
st.write(f"**Loaded:** `{path.name}` — Rows (filtered): **{len(dff)}** of {len(df_b)}")
st.write(f"Using bin: **`{which_bin}`** | Diff column: **`{which_diff}`**")

# -----------------------------
# Key table (editable toggle)
# -----------------------------
with st.expander("🔎 View data table (toggle columns, sort, download)"):
    shown_cols = [
        "Gene.name", "Gene.kind", "Hormones.SUC", "Gene.SUC_band",
        "Hormones.Sustained_growth", which_diff, which_bin
    ]
    extra_cols = [c for c in df_b.columns if c not in shown_cols]
    col_sel = st.multiselect("Add/remove columns", shown_cols + extra_cols, default=shown_cols)
    st.dataframe(dff[col_sel], use_container_width=True, height=420)
    csv = dff[col_sel].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered table (CSV)", data=csv, file_name="shootx_results_filtered.csv", use_container_width=True)

# -----------------------------
# Chart 1 • Bin counts by SUC band / kind
# -----------------------------
st.subheader("📦 Bin counts")
counts = (
    dff
    .assign(bin_txt=dff[which_bin].map(bin_label_map()))
    .groupby(["Gene.SUC_band", "Gene.kind", "bin_txt"], dropna=False)
    .size()
    .reset_index(name="count")
)
if len(counts):
    fig_counts = px.bar(
        counts,
        x="Gene.SUC_band",
        y="count",
        color="bin_txt",
        facet_col="Gene.kind",
        category_orders={"Gene.SUC_band": ["very low","low","normal","high","very high"],
                         "bin_txt": ["Lower (-1)", "Same (0)", "Higher (+1)"]},
        barmode="group",
        height=420
    )
    fig_counts.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_counts, use_container_width=True)
else:
    st.info("No rows after filters to plot counts.")

# -----------------------------
# Chart 2 • Heatmap of bins (rows = names, cols = SUC)
# -----------------------------
st.subheader("🧊 Bin heatmap (name × SUC)")
# Pivot to heatmap-friendly
if len(dff):
    # Limit names to avoid an overly tall heatmap; provide a slider
    uniq_names = dff["Gene.name"].unique().tolist()
    max_names = st.slider("Max names on heatmap", min_value=10, max_value=max(10, len(uniq_names)), value=min(60, len(uniq_names)), step=5)
    names_subset = set(uniq_names[:max_names])

    heat_df = (
        dff[dff["Gene.name"].isin(names_subset)]
        .copy()
    )
    heat_df["SUC_txt"] = heat_df["Hormones.SUC"].astype(str)
    # Fill pivot with 0 for missing
    pvt = heat_df.pivot_table(
        index="Gene.name", columns="SUC_txt", values=which_bin, aggfunc="mean", fill_value=0
    ).sort_index()
    # Order columns numerically if possible
    try:
        col_order = sorted(pvt.columns, key=lambda x: float(x))
        pvt = pvt[col_order]
    except Exception:
        pass
    fig_heat = px.imshow(
        pvt,
        aspect="auto",
        color_continuous_scale=[ "#3b82f6", "#cbd5e1", "#22c55e" ],  # blue(−1) → gray(0) → green(+1)
        zmin=-1, zmax=1,
        height=520
    )
    fig_heat.update_layout(margin=dict(l=10, r=10, t=40, b=10), coloraxis_colorbar_title=which_bin)
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No data to draw heatmap.")

# -----------------------------
# Chart 3 • Landscape (SG vs SUC)
# -----------------------------
st.subheader("🌄 Landscape: Sustained growth vs SUC")
if len(dff):
    # Toggle WT overlay
    show_wt = st.checkbox("Overlay WT curve", value=True)
    # Base scatter/lines by name
    fig_land = px.line(
        dff.sort_values(["Gene.name", "Hormones.SUC"]),
        x="Hormones.SUC",
        y="Hormones.Sustained_growth",
        color="Gene.name",
        line_group="Gene.name",
        hover_data=["Gene.kind", which_diff, which_bin, "Gene.SUC_band"],
        markers=True,
        height=520
    )
    if show_wt and "WT" in df_b["Gene.kind"].unique():
        wt_curve = (
            df_b[df_b["Gene.kind"] == "WT"]
            .sort_values("Hormones.SUC")
            [["Hormones.SUC", "Hormones.Sustained_growth"]]
            .drop_duplicates("Hormones.SUC")
        )
        fig_land.add_scatter(
            x=wt_curve["Hormones.SUC"], y=wt_curve["Hormones.Sustained_growth"],
            mode="lines+markers", name="WT", line=dict(width=4), hoverinfo="x+y+name"
        )
    fig_land.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="Genotype")
    st.plotly_chart(fig_land, use_container_width=True)
else:
    st.info("No data to draw landscape plot.")

st.divider()
st.markdown("""
**How bins are computed**
- **vs WT at same SUC (default):** `bin = sign(SG(genotype, SUC) − SG(WT, SUC))`
- **vs self at SUC = 1.0:** `bin = sign(SG(genotype, SUC) − SG(genotype, 1.0))`  
Use **Bin tolerance (ε)** to treat tiny differences as “Same (0)”.
""")
