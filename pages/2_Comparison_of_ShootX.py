# pages/2_🔬_Comparison_of_ShootX.py
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.markdown(
    "<h1 style='text-align: center; margin-top: -20px;'>🔬 Comparison of ShootX</h1>",
    unsafe_allow_html=True
)

# ---------- Config ----------
# Expected image filenames inside ./data
IMAGE_CANDIDATES = {
    "ShootX": ["data/shootx.png", "data/shootx.jpg", "data/shootx.jpeg", "data/shootx.svg"],
    "JXB":    ["data/jxb.png", "data/jxb.jpg", "data/jxb.jpeg", "data/jxb.svg"],
    "Dun":    ["data/dun.png", "data/dun.jpg", "data/dun.jpeg", "data/dun.svg"],
}

DEFAULT_ACCURACY = {
    "ShootX": 82.5,
    "JXB": 20.31,
    "Dun": 42.11,
}

# ---------- Utils ----------
def first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def card_fixed(title: str, default_accuracy: float, image_candidates: list, key_prefix: str):
    with st.container(border=True):
        # Fixed title (no editing)
        st.markdown(f"### {title}")

        # Show image from data/ (fallback message if missing)
        img_path = first_existing_path(image_candidates)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning(
                f"Image not found. Place one of the expected files for **{title}** in `data/`:\n\n- "
                + "\n- ".join(image_candidates)
            )

        # Accuracy (editable)
        acc = st.slider(
            "Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_accuracy),
            step=0.01,
            key=f"{key_prefix}_acc",
        )
        st.markdown(f"**✅ Accuracy:** {acc:.2f}%")

    # Return with empty notes (to keep table schema consistent)
    return {"Variant": title, "Accuracy (%)": round(acc, 2), "Notes": ""}

# ---------- Page ----------
st.title("🔬 Comparison of ShootX")
st.caption("Side-by-side comparison of the ShootX network versus two references (JXB, Dun). Images auto-loaded from the `data/` folder.")

# Light styling
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    shootx = card_fixed(
        title="ShootX",
        default_accuracy=DEFAULT_ACCURACY["ShootX"],
        image_candidates=IMAGE_CANDIDATES["ShootX"],
        key_prefix="shootx",
    )

with c2:
    jxb = card_fixed(
        title="JXB",
        default_accuracy=DEFAULT_ACCURACY["JXB"],
        image_candidates=IMAGE_CANDIDATES["JXB"],
        key_prefix="jxb",
    )

with c3:
    dun = card_fixed(
        title="Dun",
        default_accuracy=DEFAULT_ACCURACY["Dun"],
        image_candidates=IMAGE_CANDIDATES["Dun"],
        key_prefix="dun",
    )

st.divider()

# ---------- Summary table & chart ----------
st.subheader("Summary")
df = pd.DataFrame([shootx, jxb, dun])

st.dataframe(df, use_container_width=True)

fig_bar = go.Figure()
fig_bar.add_bar(
    x=df["Variant"],
    y=df["Accuracy (%)"],
    text=[f"{v:.2f}%" for v in df["Accuracy (%)"]],
    textposition="outside",
)
fig_bar.update_layout(
    title="Accuracy Comparison",
    yaxis_title="Accuracy (%)",
    xaxis_title="Variant",
    margin=dict(l=10, r=10, t=40, b=10),
    height=380
)
st.plotly_chart(fig_bar, use_container_width=True)

# Download CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download accuracies (CSV)",
    data=csv,
    file_name="shootx_accuracy_summary.csv",
    mime="text/csv"
)

st.info("Place your images at `data/shootx.(png/jpg/jpeg/svg)`, `data/jxb.(...)`, and `data/dun.(...)`. Adjust accuracies anytime; the table and chart update automatically.")
