import streamlit as st
import numpy as np
from pygments.styles.dracula import background
from streamlit_drawable_canvas import st_canvas

# Import our custom modules
from module.preprocessing import normalize_character_bbox, resample_stroke_spatial
from module.evaluator import evaluate_character_sequence

# Load KanjiVG ground truth function (from Step 1)
from module.svg import extract_kanjivg_strokes

st.set_page_config(page_title="Kanji Stroke Evaluator", layout="wide")

st.title("🖊️ Real-Time Kanji Writing Evaluator")
st.write("Draw the target Kanji on the canvas below to evaluate stroke order, direction, and trajectory shape.")

# Sidebar Controls
st.sidebar.header("Target Kanji")
kanji_code = st.sidebar.text_input("KanjiVG Unicode File", value="04E09")  # Default: 三 (04E09)
target_svg_path = f"data/kanjivg/{kanji_code}.svg"

# Load Ground Truth
try:
    ref_strokes_raw = extract_kanjivg_strokes(target_svg_path)
    ref_strokes_norm = normalize_character_bbox(ref_strokes_raw)
    ref_strokes = [resample_stroke_spatial(s, num_points=30) for s in ref_strokes_norm]
    st.sidebar.success(f"Loaded target Kanji with {len(ref_strokes)} strokes.")
except Exception as e:
    st.sidebar.error(f"Failed to load target SVG ({kanji_code}.svg). Make sure the file exists in data/kanjivg/")
    ref_strokes = []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Kanji Below:")
    try:
        st.image(target_svg_path, caption=f"Target Kanji ({kanji_code})", width=200)
    except Exception:
        st.error("SVG file not found.")

    st.subheader("Drawing Canvas")
    # Interactive Drawing Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=6,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="kanji_canvas",
    )

with col2:
    st.subheader("Evaluation Results")

    # Process user JSON drawing data if canvas has content
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        print("objects:", objects)
        user_strokes_raw = []

        # Extract stroke trajectories (path points) from Streamlit Canvas
        for obj in objects:
            if obj["type"] == "Path":
                path_data = obj["path"]
                stroke_points = []
                for command in path_data:
                    print("command:", command)
                    # Capture line segments (M/L coordinates)
                    if command[0] in ["M", "L"]:
                        stroke_points.append([command[1], command[2]])
                if len(stroke_points) > 1:
                    user_strokes_raw.append(np.array(stroke_points))

        print("user_strokes_raw:", user_strokes_raw)
        # print("ref_strokes:", ref_strokes)
        if user_strokes_raw and ref_strokes:
            # Preprocess raw user input
            user_norm = normalize_character_bbox(user_strokes_raw)
            user_strokes = [resample_stroke_spatial(s, num_points=30) for s in user_norm]

            # Evaluate sequence
            results = evaluate_character_sequence(user_strokes, ref_strokes)
            print("Results:", results)

            # Display Score Overview
            c1, c2, c3 = st.columns(3)
            c1.metric("Stroke Count Match", "Yes" if results["count_match"] else "No")
            c2.metric("Avg Shape Score", f"{results['avg_shape_score']} / 100")
            c3.metric("Avg Direction Score", f"{results['avg_direction_score']} / 100")

            st.divider()

            # Display Target Kanji
            # if results["count_match"]:

            st.divider()
            # Display Per-Stroke Breakdown
            st.write("### Stroke-by-Stroke Analysis")
            for res in results["stroke_breakdown"]:
                idx = res["stroke_index"]
                dir_status = "✅ Correct" if res["direction_correct"] else "❌ Drawn Backwards"

                with st.expander(f"Stroke {idx} Evaluation"):
                    st.write(f"**Shape Match Score:** {res['shape_score']} / 100")
                    st.write(f"**Direction Status:** {dir_status}")
                    st.write(f"**DTW Distance:** {res['dtw_distance']:.2f}")
        else:
            # print(len(user_strokes_raw), len(ref_strokes))
            st.info("Draw on the canvas to see real-time evaluations.")