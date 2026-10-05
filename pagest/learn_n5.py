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
from module.svg import extract_kanjivg_strokes, svg_to_numpy, numpy_to_target_tensor, rasterize_svg_to_image, generate_stroke_order_image
from module.cnn import predict_kanji_cnn, KanjiCNN, predict_kanji_cnn2, preprocess_canvas_image
from module.cnn_train import SiameseKanjiNet
from module.utility import print_log

st.set_page_config(page_title="Practice Kanji N5", page_icon="🔰", layout="wide")

# Main Function
@st.cache_resource
def load_kanji_cnn():
    print_log("Loading Kanji CNN...")
    device = torch.device("cpu")
    model = SiameseKanjiNet()
    model.load_state_dict(torch.load("data/siamese_kanji.pth", map_location=device))
    model.eval()

    return model

@st.cache_data
def load_kanji_data(shuffle: bool):
    print_log("Loading Kanji metadata...")
    metadata = load_kanji_metadata("n5.json")
    if shuffle: random.shuffle(metadata)

    return metadata

def get_kanji_stroke(target_svg_path: str):
    print_log("Getting Kanji strokes...")
    try:
        ref_raw = extract_kanjivg_strokes(target_svg_path)
        ref_norm = normalize_character_bbox(ref_raw)
        ref_strokes = [resample_stroke_spatial(s, num_points=30) for s in ref_norm]
        print_log(f"Loaded KanjiVG target ({len(ref_strokes)} strokes).")
    except Exception:
        print_log(f"Could not load SVG file ({current_kanji['unicode'].replace('U+', '0')}.svg).")
        ref_strokes = []

    return ref_strokes

def batches_to_idx(selected_batches: list, batch_dict: dict):
    sel_idx = []
    # Get index data each batch, then concatenate as selected index
    for name in selected_batches: sel_idx.extend(list(batch_dict[name]))
    return sel_idx

# def run_countdown(seconds=10):
#     """
#     Runs a real-time countdown timer in red text from `seconds` down to 0.
#     Displays a notification toast and alert when time expires.
#     """
#     # Create a dynamic placeholder container
#     timer_placeholder = st.empty()
#
#     for t in range(seconds, -1, -1):
#         # Format as MM:SS (e.g., 00:10, 00:09) or simple seconds count
#         minutes, secs = divmod(t, 60)
#         time_str = f"{minutes:02d}:{secs:02d}"
#
#         # Render styled large red text
#         timer_placeholder.markdown(
#             f"""
#             <div style="text-align: right; margin: 10px 0;">
#                 <span style="color: #b92548; font-size: 24px; font-weight: bold; font-family: monospace;">
#                     ⏱️ {time_str}
#                 </span>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#
#         # Pause for 1 second unless we hit zero
#         if t > 0:
#             time.sleep(1)
#
#     # Trigger popup notifications when time reaches 0
#     st.toast("⏰ Time's up!", icon="🚨")




# Load Instance
print_log("Starting apps...")
cnn_model = load_kanji_cnn()
kanji_metadata = load_kanji_data(shuffle=False)

# Sidebar Screen
st.sidebar.header("Practice Control")
# n_kanji = st.sidebar.slider("Number of Kanji", value=20, max_value=103, min_value=1, step=1)
# random_kanji = st.sidebar.toggle("Randomize Kanji", value=False)

if "in_practice" not in st.session_state:
    st.session_state.in_practice = False

# VIEW 1: Welcome Screen
if not st.session_state.in_practice:
    st.markdown(
        """
        # 🔰 Practice Kanji Level N5
        In this practice, you will learn how to write **103 kanji**. This kanji is very common and mostly used in daily life.
        So, make sure to learn it properly okay 😉.
        
        ### About Practice
        In this apps You will learn writting kanji perfectly in batch. Each batch by default is contain 15 kanji.

        You can set custom setting use setting option below.
        
        When ready click button below :)
        """
    )
    st.divider()
    st.subheader("Kanji Batch Selection")
    idx_range = range(len(kanji_metadata))
    batch_size = 15
    idx_batch = {}
    for i, j in enumerate(range(0, len(idx_range), batch_size), start=1):
        batch_end = i * batch_size if i * batch_size <= len(idx_range) else len(idx_range)
        if i == 1:
            idx_batch[f"Kanji 1~{batch_end}"] = idx_range[j:j + batch_size]
            continue

        idx_batch[f"Kanji {(i-1)*(batch_size)+1}~{batch_end}"] = idx_range[j:j + batch_size]

    batch_opt = st.multiselect(
        "Select Kanji Batch",
        idx_batch,
        default=[list(idx_batch.keys())[0]],
    )

    st.divider()
    st.subheader("Practice Setting")
    with st.container():
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            sett_n_kanji = batch_size
            sett_random_kanji = False

            if st.toggle("Enable Custom Setting"):
                sett_n_kanji = st.slider("Number of Kanji Appear", value=batch_size, max_value=len(batch_opt)*batch_size, min_value=1, step=1)
                sett_random_kanji = st.toggle("Shuffle Kanji")
                timeout_opt = {"10s": 10, "15s": 15, "20s": 20, }
                # sett_answer_timeout = st.pills("Time out",
                #                                timeout_opt,
                #                                selection_mode="single",
                #                                default=["10s"]
                #                                )



    with st.container(horizontal=True):
        st.space("stretch")
        if st.button("Start Practice", type="primary", disabled=(len(batch_opt) < 1)):
            st.session_state.in_practice = True
            st.session_state.kanji_selected_idx = batches_to_idx(batch_opt, idx_batch)
            st.session_state.sett_n_kanji = sett_n_kanji #len(st.session_state.kanji_selected_idx)
            # st.session_state.sett_answer_timeout = timeout_opt[sett_answer_timeout]
            st.session_state.sett_random_kanji = sett_random_kanji
            st.session_state.sett_random_seed = random.randint(1,100)
            st.rerun()
        st.space("stretch")

# VIEW 2: Practice Screen
else:
    st.markdown(
        """
        # 🔰 Practice Kanji Level N5
        """
    )
    selected_kanji_metadata = [kanji_metadata[i] for i in st.session_state.kanji_selected_idx]

    if "kanji_index" not in st.session_state:
        st.session_state.kanji_index = 0

    if "test_score" not in st.session_state:
        st.session_state.test_score = []

    # initiate current kanji
    if st.session_state.sett_random_kanji:
        random.seed(st.session_state.sett_random_seed)
        random.shuffle(selected_kanji_metadata)

    current_kanji = selected_kanji_metadata[st.session_state.kanji_index]
    kanji_in_question = current_kanji.get('kanji', '')
    target_svg_path = f"data/kanjivg/{current_kanji['unicode'].replace('U+', '0')}.svg"
    ref_strokes = get_kanji_stroke(target_svg_path)
    target_img = generate_stroke_order_image(target_svg_path, size=350, line_thickness=3)
    # practice_container =
    with st.container():
        headcol1, headcol2, headcol3 = st.columns([1, 1, .8])

        with headcol1:
            st.subheader(f"✏️ Writting Practice! ({st.session_state.kanji_index + 1}/{st.session_state.sett_n_kanji})")

        # with headcol3:
            # run_countdown(st.session_state.sett_answer_timeout)
            # progress_text = "Operation in progress. Please wait."
            # my_bar = st.progress(0, text=progress_text)
            #
            # for percent_complete in range(100):
            #     # time.sleep(1)
            #     my_bar.progress(percent_complete + 1, text=progress_text)
            # # time.sleep(1)
            # # my_bar.empty()

        col1, col2, col_results = st.columns([1, 1, .8])

        with col1:
            st.subheader("Kanji Stroke Order")
            st.image(target_img, caption="Strokes ordered kanji", width=200)
            st.markdown(
                f"""
                ### **{current_kanji['meaning'].title()}**
                
                **Kunyomi:** {current_kanji['kunyomi'].title()}
                
                **Onyomi:** {current_kanji['onyomi'].title()} 
                
                **Sentence:** {current_kanji.get('sentence', '').title()} 
                """
            )

        with col2:
            if "canvas_key" not in st.session_state:
                st.session_state.canvas_key = 0

            st.subheader("Drawing Canvas")
            canvas_size = (300,300)
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=10,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=canvas_size[0],
                width=canvas_size[1],
                drawing_mode="freedraw",
                key=f"canvas_instance_{st.session_state.canvas_key}",
                return_image_data=True
            )

            # debug
            if canvas_result.image_data is not None:
                # 1. Generate preprocessed 28x28 PyTorch tensor
                tensor_img = preprocess_canvas_image(canvas_result.image_data)

                # 2. Display CNN Debug Preview
                st.caption("🔍 CNN Input Preview (28x28 Tensor Image)")

                # Convert tensor shape [1, 1, 28, 28] -> numpy array [28, 28]
                debug_img = tensor_img.squeeze().cpu().numpy()

                # Display small debug preview box
                st.image(debug_img, width=100, clamp=True)

        with col_results:
            st.subheader("Evaluation Dashboard")

            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                objects = canvas_result.json_data["objects"]
                user_strokes_raw = []

                # 1. Parse Trajectories (Support M, L, Q SVG commands)
                for obj in objects:
                    if obj["type"] == "Path":
                        stroke_points = []
                        for command in obj["path"]:
                            cmd_type = command[0]
                            if cmd_type in ["M", "L"]:
                                stroke_points.append([command[1], command[2]])
                            elif cmd_type == "Q":
                                stroke_points.append([command[3], command[4]])
                        if len(stroke_points) > 1:
                            user_strokes_raw.append(np.array(stroke_points))

                if user_strokes_raw and ref_strokes:
                    # --- Dynamic Processing (DTW & Direction) ---
                    user_norm = normalize_character_bbox(user_strokes_raw)
                    user_strokes = [resample_stroke_spatial(s, num_points=100) for s in user_norm]
                    seq_results = evaluate_character_sequence(user_strokes, ref_strokes)

                    dtw_score = seq_results["avg_shape_score"]
                    direction_score = seq_results["avg_direction_score"]

                    # --- Static Processing (SSIM & CNN) ---
                    canvas_img_rgba = canvas_result.image_data  # RGBA numpy array

                    # Compute SSIM score
                    ssim_score = evaluate_ssim(canvas_img_rgba, target_svg_path, canvas_size=canvas_size)

                    # Compute CNN prediction confidence
                    user_tensor = preprocess_canvas_image(canvas_img_rgba)
                    np_img = svg_to_numpy(target_svg_path, thickness=3)
                    target_tensor = numpy_to_target_tensor(np_img, "cpu")
                    user_tensor = preprocess_canvas_image(canvas_result.image_data)
                    cnn_similarity = calculate_similarity_score(cnn_model, user_tensor, target_tensor)
                    # print("user_rgba_shape", canvas_img_rgba.shape, type(canvas_img_rgba))
                    # print("user_tensor", user_tensor.shape, type(user_tensor))
                    # print("svg_to_numpy", np_img.shape, type(np_img))
                    # print("target_tensor", target_tensor.shape, type(target_tensor))


                    # --- Final Weighted Score Calculation ---
                    w1, w2, w3, w4 = 0.0, 0.425, 0.425, 0.15
                    final_score = round(
                        (w1 * dtw_score) +
                        (w2 * direction_score) +
                        (w3 * ssim_score) +
                        (w4 * cnn_similarity),
                        2
                    )

                    # --- Render Main Final Score Banner ---
                    st.metric("Overall Score", f"{final_score} / 100")
                    st.progress(min(int(final_score), 100))

                    st.divider()

                    # --- 4-Component Score Breakdown Grid ---
                    # st.write("### Metric Breakdown")
                    # m1, m2, m3, m4 = st.columns(4)
                    # st.metric("1. Stroke Order (DTW)", f"{dtw_score:.3f}%")
                    st.metric("2. Direction Score", f"{direction_score:.3f}%")
                    st.metric("3. SSIM Proportion", f"{ssim_score:.3f}%")
                    st.metric("4. CNN Similarity", f"{cnn_similarity:.3f}%")

                    # # --- Stroke-by-Stroke Details ---
                    # st.write("### Detailed Stroke Breakdown")
                    # for res in seq_results["stroke_breakdown"]:
                    #     idx = res["stroke_index"]
                    #     status = "✅ Correct" if res["direction_correct"] else "❌ Drawn Backwards"
                    #     with st.expander(f"Stroke {idx} Analysis"):
                    #         st.write(f"**Shape Similarity (DTW):** {res['shape_score']} / 100")
                    #         st.write(f"**Direction Status:** {status}")
                else:
                    st.info("Draw on the canvas to compute the live score.")

    st.divider()
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, .8])

    with nav_col1:
        if st.button("← Previous", disabled=(st.session_state.kanji_index == 0)):
            st.session_state.kanji_index -= 1
            st.session_state.canvas_key += 1
            st.rerun()

    with nav_col2:
        if st.button("End Practice"):
            st.balloons()
            st.session_state.kanji_index = 0
            st.session_state.in_practice = False
            _ = st.session_state.pop("sett_n_kanji", None)
            _ = st.session_state.pop("sett_random_kanji", None)
            _ = st.session_state.pop("sett_random_seed", None)
            st.session_state.canvas_key += 1
            st.rerun()

    with nav_col3:
        if st.session_state.kanji_index < st.session_state.sett_n_kanji - 1:
            if st.button("Next Kanji →", type="primary"):
                st.session_state.kanji_index += 1
                st.session_state.canvas_key += 1
                st.session_state.test_score.append({"kanji": kanji_in_question,
                                                    "final_score": final_score,
                                                    "direction_score": direction_score,
                                                    "dtw_score": dtw_score,
                                                    "ssim_score": ssim_score,
                                                    "cnn_similarity": cnn_similarity
                                                    })
                st.rerun()
        else:
            if st.button("Finish Practice 🎉", type="primary"):
                st.session_state.test_score.append({"kanji": kanji_in_question,
                                                    "final_score": final_score,
                                                    "direction_score": direction_score,
                                                    "dtw_score": dtw_score,
                                                    "ssim_score": ssim_score,
                                                    "cnn_similarity": cnn_similarity
                                                    })
                st.session_state.kanji_index = 0
                st.session_state.in_practice = False
                _ = st.session_state.pop("sett_n_kanji", None)
                _ = st.session_state.pop("sett_random_kanji", None)
                _ = st.session_state.pop("sett_random_seed", None)
                st.session_state.canvas_key += 1
                st.switch_page("pagest/score_page.py")