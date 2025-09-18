import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Creation of ShootX", page_icon="🧪", layout="wide")

st.title("🧪 Creation of ShootX")

# --- Custom CSS ---
st.markdown('''
    <style>
        .hero {
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        }
        .glass {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.25rem;
        }
        .cta-btn > button {
            border-radius: 999px !important;
            padding: 0.6rem 1.1rem !important;
            font-weight: 600 !important;
        }
        footer {visibility: hidden;}
    </style>
''', unsafe_allow_html=True)

# --- Pipeline image ---
st.title("🌿 Pipeline of ShootX Creation")
img_path = Path("data/Picture1.png")
st.image(str(img_path), use_container_width=True)

# --- Training dataset (group by unique Psoup treatment) ---
st.header("📚 Datasets")
st.subheader("Training Dataset (rolled up by Psoup treatment)")

data_path = Path("data/Training Data.xlsx")
df = pd.read_excel(data_path)

# Ensure expected columns exist
required_cols = ["Publication", "Mutant", "Psoup treatment", "Actual tiller number", "Species", "Biological bin"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns in Training Data.xlsx: {missing}")
else:
    # Clean types
    df["Biological bin"] = pd.to_numeric(df["Biological bin"], errors="coerce").fillna(0).astype(int)

    # Aggregate species & publications per treatment
    rollup_meta = (
        df.groupby("Psoup treatment")
          .agg(
              Species=("Species", lambda s: ", ".join(sorted(set(map(str, s))))),
              Publications=("Publication", lambda p: " | ".join(sorted(set(map(str, p)))))
          )
          .reset_index()
    )

    # Bin counts per treatment
    bin_counts = (
        df.assign(bin_label=df["Biological bin"].map({-1: "Bin -1", 0: "Bin 0", 1: "Bin 1"}))
          .groupby(["Psoup treatment", "bin_label"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=["Bin -1", "Bin 0", "Bin 1"], fill_value=0)
          .reset_index()
    )

    # Merge + totals
    treatment_summary = (
        rollup_meta.merge(bin_counts, on="Psoup treatment", how="left")
                   .fillna(0)
    )
    treatment_summary["Total N"] = treatment_summary[["Bin -1", "Bin 0", "Bin 1"]].sum(axis=1)
    treatment_summary = treatment_summary.sort_values(["Total N", "Psoup treatment"], ascending=[False, True])

    # Display table
    st.dataframe(treatment_summary, use_container_width=True)

    # Optional download
    st.download_button(
        "Download summary (CSV)",
        data=treatment_summary.to_csv(index=False).encode("utf-8"),
        file_name="training_summary_by_psoup_treatment.csv",
        mime="text/csv",
    )

    # --- Visualization: stacked bars per treatment (hover shows species & papers)
    import plotly.express as px
    melted = treatment_summary.melt(
        id_vars=["Psoup treatment", "Species", "Publications", "Total N"],
        value_vars=["Bin -1", "Bin 0", "Bin 1"],
        var_name="Biological Bin",
        value_name="Count"
    )
    fig = px.bar(
        melted,
        x="Psoup treatment",
        y="Count",
        color="Biological Bin",
        barmode="stack",
        text="Count",
        hover_data={"Species": True, "Publications": True, "Total N": True},
        title="Biological-bin distribution per Psoup treatment"
    )
    fig.update_layout(xaxis_title="Psoup treatment", yaxis_title="Replicate count")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🌐 Network view (clean layout + controls)
# =========================
st.subheader("🧬 Network view")

import networkx as nx
import plotly.graph_objects as go
import re
from collections import defaultdict, deque

# ---- Controls
colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
layout_choice = colA.selectbox("Layout", ["Hierarchical", "Spring", "Circular"], index=0)
layer_gap = colB.slider("Layer gap", 0.6, 3.0, 1.4, 0.1)
node_gap = colC.slider("Node gap", 0.4, 2.5, 0.9, 0.1)
node_size = colD.slider("Node size", 14, 36, 22, 1)

# ---- Load edges ----
edges_path = Path("data/edges_B.csv")
edges_df = pd.read_csv(edges_path)

# ---- Build graph ----
G = nx.DiGraph()
for _, r in edges_df.iterrows():
    src, dst, infl = str(r["From"]).strip(), str(r["To"]).strip(), str(r["Influence"]).strip()
    G.add_node(src)
    G.add_node(dst)
    G.add_edge(src, dst, influence=infl)

# ---- Helper: normalize tokens from "Psoup treatment" to match node names ----
def normalize_token(tok: str) -> str:
    t = re.sub(r"\s+", "", str(tok))
    t = t.replace("+", "").replace(",", "")
    t = re.sub(r"_n$", "", t, flags=re.I)
    t = re.sub(r"^WT$", "", t, flags=re.I)
    return t

def treatments_to_tokens(psoup_treatment: str) -> set:
    raw = re.split(r"[,+]", str(psoup_treatment))
    toks = set(filter(None, (normalize_token(x) for x in raw)))
    return toks

# ---- From training data: species/papers/bin counts per node ----
node_species = {n: set() for n in G.nodes}
node_papers  = {n: set() for n in G.nodes}
node_bin_counts = {n: {-1: 0, 0: 0, 1: 0} for n in G.nodes}

for _, row in df.iterrows():
    toks = treatments_to_tokens(row["Psoup treatment"])
    species = str(row["Species"]).strip()
    pub = str(row["Publication"]).strip()
    try:
        b = int(row["Biological bin"])
    except Exception:
        b = 0
    for n in G.nodes:
        if n in toks:
            if species:
                node_species[n].add(species)
            if pub:
                node_papers[n].add(pub)
            if b in (-1, 0, 1):
                node_bin_counts[n][b] += 1

# ---- Species colors (ensure Arabidopsis=blue)
palette = ["#1f77b4","#2ca02c","#d62728","#9467bd","#8c564b","#17becf","#ff7f0e","#bcbd22"]
species_all = sorted({s for ss in node_species.values() for s in ss})
species_color = {sp: palette[i % len(palette)] for i, sp in enumerate(species_all)}
if "Arabidopsis" in species_color:
    species_color["Arabidopsis"] = "#1f77b4"

# ---- Layout functions
def hierarchical_layout(g: nx.DiGraph, layer_gap=1.4, node_gap=1.0):
    # Topological layering; if cycles exist, break by ignoring back-edges for ranking
    indeg = {n: g.in_degree(n) for n in g.nodes}
    q = deque([n for n, d in indeg.items() if d == 0])
    rank = {n: 0 for n in g.nodes}
    seen = set(q)

    while q:
        u = q.popleft()
        for v in g.successors(u):
            rank[v] = max(rank[v], rank[u] + 1)
            if v not in seen:
                seen.add(v)
                q.append(v)

    # If graph wasn’t fully covered (cycles), assign remaining with spring-based ranks
    if len(seen) < len(g.nodes):
        extra = [n for n in g.nodes if n not in seen]
        for n in extra:
            rank[n] = max((rank.get(p, 0) for p in g.predecessors(n)), default=0) + 1

    # Nodes per layer & y-spacing
    layers = defaultdict(list)
    for n, r in rank.items():
        layers[r].append(n)
    max_width = max(len(v) for v in layers.values())
    pos = {}
    # Assign positions: x = layer * layer_gap, y evenly spaced with node_gap
    for x_layer, nodes in sorted(layers.items()):
        m = len(nodes)
        # Center y around 0
        start_y = - (m-1) * node_gap / 2
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (x_layer * layer_gap, start_y + i * node_gap)
    return pos

def spring_layout_clean(g, k=1.2):
    return nx.spring_layout(g, seed=7, k=k, iterations=200)

def circular_layout_clean(g, radius=3.0):
    base = nx.circular_layout(g)
    return {n: (radius*xy[0], radius*xy[1]) for n, xy in base.items()}

# Choose layout
if layout_choice.startswith("Hierarchical"):
    pos = hierarchical_layout(G, layer_gap=layer_gap, node_gap=node_gap)
elif layout_choice == "Spring":
    pos = spring_layout_clean(G, k=max(layer_gap, node_gap))
else:
    pos = circular_layout_clean(G, radius=layer_gap*2.0)

# ---- Utility for curved edge points (quadratic Bézier)
def quad_curve(x0, y0, x1, y1, curvature=0.25):
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    # Perpendicular control point
    dx, dy = (x1 - x0), (y1 - y0)
    cx, cy = mx - curvature * dy, my + curvature * dx
    ts = [i/20 for i in range(21)]
    xs = [(1-t)**2 * x0 + 2*(1-t)*t*cx + t**2 * x1 for t in ts]
    ys = [(1-t)**2 * y0 + 2*(1-t)*t*cy + t**2 * y1 for t in ts]
    return xs, ys

def edge_style(infl: str):
    # Color + dash by influence
    is_inhib = isinstance(infl, str) and infl.lower().startswith("inhib")
    col = "#d62728" if is_inhib else "#2ca02c"
    dash = "dash" if is_inhib else "solid"
    return col, dash

# ---- Edge traces with curvature
edge_traces = []
arrow_annotations = []
for i, (u, v, d) in enumerate(G.edges(data=True)):
    x0, y0 = pos[u]; x1, y1 = pos[v]
    color, dash = edge_style(d.get("influence", ""))
    xs, ys = quad_curve(x0, y0, x1, y1, curvature=0.18)
    edge_traces.append(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=2, color=color, dash=dash),
            hoverinfo="skip", showlegend=False
        )
    )
    # Arrowhead at end
    arrow_annotations.append(dict(
        ax=xs[-2], ay=ys[-2], x=xs[-1], y=ys[-1],
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color, standoff=4
    ))

# ---- Node scatter + hover
node_x, node_y, node_text, node_labels = [], [], [], []
for n in G.nodes:
    node_x.append(pos[n][0]); node_y.append(pos[n][1])
    bins = node_bin_counts[n]; papers = sorted(node_papers[n]); species_list = sorted(node_species[n])
    hover = (
        f"<b>{n}</b><br>"
        f"<b>Species:</b> {', '.join(species_list) if species_list else '—'}<br>"
        f"<b>Bin counts:</b> (-1) {bins[-1]}, (0) {bins[0]}, (1) {bins[1]}<br>"
        f"<b>Publications:</b><br>" + ("<br>".join(papers) if papers else "—")
    )
    node_text.append(hover)
    node_labels.append(n)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=node_labels,
    textposition="top center",
    textfont=dict(size=12),
    marker=dict(size=node_size, line=dict(width=1, color="#333"), color="#ffffff"),
    hoverinfo="text", hovertext=node_text, showlegend=False,
)

# ---- Species mini-squares under each node
badge_traces = []
y_offset = 0.12  # lower more so labels don’t collide
x_gap_sq = 0.14  # more spacing between squares
square_size = 12
for n in G.nodes:
    sx, sy = pos[n]
    sp_list = sorted(node_species[n])
    start = sx - (len(sp_list)-1) * x_gap_sq / 2 if sp_list else sx
    for i, sp in enumerate(sp_list):
        badge_traces.append(
            go.Scatter(
                x=[start + i * x_gap_sq],
                y=[sy - y_offset],
                mode="markers",
                marker=dict(symbol="square", size=square_size, color=species_color[sp],
                            line=dict(width=1, color="#111")),
                hoverinfo="skip",
                showlegend=False,
            )
        )

# ---- Species legend
legend_traces = []
for sp, col in species_color.items():
    legend_traces.append(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=12, color=col),
            name=sp
        )
    )

fig_net = go.Figure(data=edge_traces + [node_trace] + badge_traces + legend_traces)
fig_net.update_layout(
    title="Regulatory network (hover a node for details)",
    title_x=0.5, showlegend=True,
    hoverlabel=dict(bgcolor="rgba(0,0,0,0.85)", font_color="white"),
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    annotations=arrow_annotations,
)

# Pad axes to reduce crowding at edges
xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
if xs and ys:
    pad = 0.6
    fig_net.update_xaxes(range=[min(xs)-pad, max(xs)+pad])
    fig_net.update_yaxes(range=[min(ys)-pad, max(ys)+pad])

st.plotly_chart(fig_net, use_container_width=True)
