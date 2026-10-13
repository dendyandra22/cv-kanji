import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
from PIL import Image
import json
import random
import time

from module.data import load_kanji_metadata
from module.preprocessing import normalize_character_bbox, resample_stroke_spatial
from module.evaluator import evaluate_character_sequence, evaluate_ssim, calculate_similarity_score
from module.svg import extract_kanjivg_strokes, svg_to_numpy, numpy_to_target_tensor, rasterize_svg_to_image, generate_stroke_order_image, render_animated_kanji
from module.cnn import preprocess_canvas_image
from module.cnn_train import SiameseKanjiNet
from module.utility import print_log
from app_pages.cards import menu_card, lesson_card
from app_pages.sidebar import get_sidebar, get_credit_footer, get_kanji_level_card

st.set_page_config(page_title="Exercise Kanji", page_icon="📋", layout="centered")

# Main Function

@st.cache_data
def load_kanji_data(shuffle: bool):
    print_log("Loading Kanji metadata...")
    metadata = load_kanji_metadata("n5.json")
    if shuffle: random.shuffle(metadata)

    return metadata

# Load Instance
print_log("Starting apps...")
kanji_metadata = load_kanji_data(shuffle=False)

# Sidebar Screen
get_sidebar()


# Exercise Welcome Screen
st.markdown(
    """
    # 📋 Kanji Exercise
    """
)
with st.container(horizontal_alignment="center"):
    st.image(
        "https://img.magnific.com/premium-vector/man-woman-sitting-stack-books-reading-intently-while-using-laptops-two-people-reading-books-using-laptops-simple-minimalist-flat-vector-illustration_538213-119036.jpg",
        width=500)

# How To Section
st.subheader("💡 How It Works")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True, height=180):
        st.markdown("#### 1. Select a Level")
        st.write("Choose a JLPT batch below to start exercising.")

with col2:
    with st.container(border=True, height=180):
        st.markdown("#### 2. Draw & Trace")
        st.write("Draw Kanji on screen to answer the question.")

with col3:
    with st.container(border=True, height=180):
        st.markdown("#### 3. Get Results")
        st.write("Review test result at the end of the exercise.")
st.divider()

# Kanji Level Section
st.subheader("Select a Exercise Level")
get_kanji_level_card(kanji_metadata, practice_route=False)

# Credit Section
get_credit_footer()