import streamlit as st
import numpy as np
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json

from module.preprocessing import normalize_character_bbox, resample_stroke_spatial
from module.evaluator import evaluate_character_sequence, evaluate_ssim, calculate_similarity_score
from module.svg import extract_kanjivg_strokes, svg_to_numpy, numpy_to_target_tensor, rasterize_svg_to_image, generate_stroke_order_image
from module.cnn import predict_kanji_cnn, KanjiCNN, predict_kanji_cnn2, preprocess_canvas_image
from module.cnn_train import SiameseKanjiNet

st.set_page_config(page_title="Kanji Writing Evaluator", layout="wide")

st.title("🖊️ Real-Time Kanji Writing Evaluator")
st.caption("Complete 4-Metric System: DTW Order, Cosine Direction, SSIM Proportion & CNN Recognition")


# --- Load PyTorch CNN Model (Cached) ---
@st.cache_resource
def load_kanji_cnn():
    # List of candidate classes (e.g., K-MNIST classes or target Kanji)
    # class_names = ["円", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    # N5_KANJI = [
    #     "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "円",
    #     "日", "月", "火", "水", "木", "金", "土", "年", "時", "分", "半", "今", "先", "毎",
    #     "山", "川", "花", "雨", "天", "空", "気", "電", "車", "名", "前", "後", "午", "間",
    #     "人", "男", "女", "子", "母", "父", "友", "生", "学", "校", "本", "語", "書", "聞",
    #     "読", "見", "行", "来", "出", "入", "会", "休", "買", "立", "言", "話", "食", "飲",
    #     "大", "小", "高", "安", "新", "古", "長", "多", "少", "白", "赤", "青", "国", "社",
    #     "上", "下", "中", "外", "右", "左", "東", "西", "南", "北", "方", "店", "道", "目",
    #     "耳", "手", "足", "口", "何"
    # ]
    # w, class_names

    device = torch.device("cpu")
    model = SiameseKanjiNet()
    model.load_state_dict(torch.load("data/siamese_kanji.pth", map_location=device))
    model.eval()

    return model


# cnn_model, cnn_classes = load_kanji_cnn()
cnn_model = load_kanji_cnn()

# Sidebar Controls
st.sidebar.header("Target Kanji")
kanji_code = st.sidebar.text_input("KanjiVG Unicode File", value="04E09")  # Default: 三 (04E09)
target_svg_path = f"data/kanjivg/{kanji_code}.svg"

# Load Ground Truth Strokes
try:
    ref_raw = extract_kanjivg_strokes(target_svg_path)
    ref_norm = normalize_character_bbox(ref_raw)
    ref_strokes = [resample_stroke_spatial(s, num_points=30) for s in ref_norm]
    st.sidebar.success(f"Loaded KanjiVG target ({len(ref_strokes)} strokes).")
except Exception:
    st.sidebar.error(f"Could not load SVG file ({kanji_code}.svg).")
    ref_strokes = []

col1, col2, col_results = st.columns([1, 1, .8])

with col1:
    st.subheader("Kanji Stroke Order")
    # target_img = rasterize_svg_to_image(target_svg_path, size=64,thickness=4, background="white")
    # st.image(target_img, width=350, clamp=True)
    target_img = generate_stroke_order_image(target_svg_path, size=350, line_thickness=3)
    st.image(target_img, caption="Strokes ordered kanji", width=350)

with col2:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="kanji_canvas",
        return_image_data=True
    )

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
            # canvas_pil = Image.fromarray(canvas_img_rgba.astype('uint8'), 'RGBA')

            # Compute SSIM score
            ssim_score = evaluate_ssim(canvas_img_rgba, target_svg_path)

            # Compute CNN prediction confidence
            # pred_char, cnn_confidence = predict_kanji_cnn(canvas_pil, cnn_model, cnn_classes)
            # pred_char, cnn_confidence = predict_kanji_cnn2(cnn_model, canvas_img_rgba, cnn_classes)
            user_tensor = preprocess_canvas_image(canvas_img_rgba)
            np_img = svg_to_numpy(target_svg_path, thickness=3)
            target_tensor = numpy_to_target_tensor(np_img, "cpu")
            user_tensor = preprocess_canvas_image(canvas_result.image_data)
            cnn_similarity = calculate_similarity_score(cnn_model, user_tensor, target_tensor)
            print("user_rgba_shape", canvas_img_rgba.shape, type(canvas_img_rgba))
            print("user_tensor", user_tensor.shape, type(user_tensor))
            print("svg_to_numpy", np_img.shape, type(np_img))
            print("target_tensor", target_tensor.shape, type(target_tensor))
            # st.write(f"Match Accuracy: {score}%")

            # --- Final Weighted Score Calculation ---
            w1, w2, w3, w4 = 0.0, 0.5, 0.5, 0.0
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
            st.write("### Metric Breakdown")
            # m1, m2, m3, m4 = st.columns(4)
            st.metric("1. Stroke Order (DTW)", f"{dtw_score:.1f}%")
            st.metric("2. Direction Score", f"{direction_score:.1f}%")
            st.metric("3. SSIM Proportion", f"{ssim_score:.1f}%")
            st.metric("4. CNN Similarity", f"{cnn_similarity:.1f}%")

            # st.caption(f"🤖 **CNN Recognition:** Predicted **'{pred_char}'** with {cnn_confidence}% confidence.")

            st.divider()

            # --- Stroke-by-Stroke Details ---
            st.write("### Detailed Stroke Breakdown")
            for res in seq_results["stroke_breakdown"]:
                idx = res["stroke_index"]
                status = "✅ Correct" if res["direction_correct"] else "❌ Drawn Backwards"
                with st.expander(f"Stroke {idx} Analysis"):
                    st.write(f"**Shape Similarity (DTW):** {res['shape_score']} / 100")
                    st.write(f"**Direction Status:** {status}")
        else:
            st.info("Draw on the canvas to compute the live score.")