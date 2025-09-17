import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Creation of ShootX", page_icon="🧪", layout="wide")

st.title("🧪 Creation of ShootX")

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

st.title("🌿 Pipeline of of ShootX Creation")

img_path = Path("data/Picture1.png")
st.image(str(img_path), use_container_width=True)