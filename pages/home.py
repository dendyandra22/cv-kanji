import streamlit as st
from pages.cards import menu_card
from pages.sidebar import get_sidebar, get_credit_footer

st.set_page_config(page_title="Kakikata - Master Kanji Writing", page_icon="✍️", layout="centered")

get_sidebar()

# Title & Subtitle
st.markdown("<h1 style='text-align: center;'>Welcome to Kakikata! ✍️</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.2rem;'>"
    "Master Japanese Kanji drawing stroke by stroke with real-time writing feedback."
    "</p>",
    unsafe_allow_html=True
)

# st.divider()

# Home Banner Image
st.image("static/img/menu1.jpg", use_container_width=True)
st.divider()

# Quick Overview Section
st.subheader("🚀 Getting Started")

cols = st.columns([1,1])
with cols[0].container(height=550, horizontal_alignment="center"):
    menu_card(
        label="📖 Step-by-Step Practice",
        image="learn.jpg",
        content="Select your level (JLPT N5 to N1) and learn proper stroke order, balance, and proportions with guided drawing animation.",
        btn_text="Start Learning",
        page_ref="learning_page.py"
    )

with cols[1].container(height=550, horizontal_alignment="center"):
    menu_card(
        label="🎯 Writing Evaluation",
        image="exercise.jpg",
        content="Test your memory! Draw characters from memory and get instant AI feedback on your stroke accuracy and handwriting score.",
        btn_text="Start Exercise",
        page_ref="exercise_page.py"
    )

# Credit Section
get_credit_footer()