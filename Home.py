import streamlit as st
from pathlib import Path

st.set_page_config(page_title="ShootX", page_icon="🌿", layout="wide")

st.markdown('''
    <style>
        .hero {{
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        }}
        .glass {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.25rem;
        }}
        .cta-btn > button {{
            border-radius: 999px !important;
            padding: 0.6rem 1.1rem !important;
            font-weight: 600 !important;
        }}
        footer {{visibility: hidden;}}
    </style>
''', unsafe_allow_html=True)

st.title("🌿 ShootX")

img_path = Path("data/shootx.jpg")
st.image(str(img_path), use_container_width=True)

st.divider()
st.subheader("Get started")
st.page_link("pages/1_Creation_of_ShootX.py", label="🧪 Creation of ShootX", icon="🧪")
st.page_link("pages/2_Comparison_of_ShootX.py", label="🔬 Comparison of ShootX", icon="🔬")
st.caption("Use the sidebar to jump between pages anytime.")