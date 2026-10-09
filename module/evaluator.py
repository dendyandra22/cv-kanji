import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import cv2
from skimage.metrics import structural_similarity as ssim
# from PIL import Image
# import io
import torch
import torch.nn.functional as F
from svgpathtools import svg2paths

def evaluate_stroke_direction(user_stroke, ref_stroke):
    """
    Computes cosine similarity between overall direction vectors of two strokes.
    Returns a score between -1.0 (opposite direction) and 1.0 (identical direction).
    """
    # Vector from start point to end point
    user_vec = user_stroke[-1] - user_stroke[0]
    ref_vec = ref_stroke[-1] - ref_stroke[0]

    norm_user = np.linalg.norm(user_vec)
    norm_ref = np.linalg.norm(ref_vec)

    # Handle zero-length strokes (e.g., dots/taten)
    if norm_user < 1e-5 or norm_ref < 1e-5:
        return 1.0

    cosine_sim = np.dot(user_vec, ref_vec) / (norm_user * norm_ref)
    return float(cosine_sim)

def evaluate_single_stroke(user_stroke, ref_stroke, max_dtw_dist=1000.0):
    """
    Evaluates shape match (via FastDTW) and direction match for a single stroke.
    """
    # 1. FastDTW Shape Distance
    dtw_dist, _ = fastdtw(user_stroke, ref_stroke, dist=euclidean)

    # Normalize DTW distance into a 0 - 100 score (lower dist -> higher score)
    # shape_score = max(0.0, 100.0 * (1.0 - (dtw_dist / max_dtw_dist)))
    shape_score = 100.0 * np.exp(-dtw_dist / max_dtw_dist)

    # 2. Direction Check
    dir_sim = evaluate_stroke_direction(user_stroke, ref_stroke)

    # Penalize direction heavily if drawn backwards
    direction_correct = dir_sim > 0.0
    direction_score = max(0.0, dir_sim * 100.0)

    return {
        "dtw_distance": float(dtw_dist),
        "shape_score": round(shape_score, 5),
        "direction_score": round(direction_score, 5),
        "direction_correct": direction_correct
    }

def evaluate_character_sequence(user_strokes, ref_strokes):
    """
    Evaluates sequence order, individual strokes, and overall trajectory score.
    """
    num_user = len(user_strokes)
    num_ref = len(ref_strokes)

    results = []

    # Check total count penalty
    count_match = (num_user == num_ref)

    # Evaluate stroke-by-stroke in drawn sequence
    min_strokes = min(num_user, num_ref)
    for i in range(min_strokes):
        stroke_eval = evaluate_single_stroke(user_strokes[i], ref_strokes[i])
        stroke_eval["stroke_index"] = i + 1
        results.append(stroke_eval)

    avg_shape = np.mean([r["shape_score"] for r in results]) if results else 0.0
    avg_dir = np.mean([r["direction_score"] for r in results]) if results else 0.0

    if not count_match:
        total_stroke_score = int((num_user / num_ref) * 100) if num_user < num_ref else 0
        return {
            "count_match": count_match,
            "user_stroke_count": num_user,
            "target_stroke_count": num_ref,
            "total_stroke_score": total_stroke_score,
            "avg_shape_score": 0,  # penalize incorrect num of stroke order
            "avg_direction_score": 0, # penalize incorrect num of stroke order
            "stroke_breakdown": results
        }

    return {
        "count_match": count_match,
        "user_stroke_count": num_user,
        "target_stroke_count": num_ref,
        "total_stroke_score": int((num_user / num_ref) * 100),
        "avg_shape_score": avg_shape, #round(avg_shape, 2),
        "avg_direction_score": round(avg_dir, 2),
        "stroke_breakdown": results
    }

def rasterize_svg_to_canvas(svg_path, canvas_shape=(350, 350), line_thickness=6):
    """
    Renders SVG path curves onto a binary NumPy image canvas using OpenCV.
    No Cairo C-libraries required!
    """
    canvas = np.zeros(canvas_shape, dtype=np.uint8)

    try:
        paths, attributes = svg2paths(svg_path)

        # KanjiVG standard viewBox is 109x109, scale it to match Streamlit canvas size
        scale_x = canvas_shape[1] / 109.0
        scale_y = canvas_shape[0] / 109.0

        for path in paths:
            # Sample points along Bezier curves
            pts = []
            for t in np.linspace(0, 1, 50):
                point = path.point(t)
                x = int(point.real * scale_x)
                y = int(point.imag * scale_y)
                pts.append([x, y])

            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            # Draw line path on black canvas (white strokes = 255)
            cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=line_thickness)

    except Exception as e:
        print(f"Error rasterizing SVG template: {e}")

    return canvas

def evaluate_ssim(user_img_array, target_svg_path, canvas_size=(350, 350)):
    """
    Computes SSIM between user canvas drawing image and rendered template image.
    """
    # Convert RGBA canvas image to grayscale
    user_gray = cv2.cvtColor(user_img_array, cv2.COLOR_RGBA2GRAY)

    # Invert so background is black (0) and stroke is white (255)
    _, user_bin = cv2.threshold(user_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Convert target kanji from SVG to PNG bytes
    try:
        # Rasterize target SVG file into PNG bytes at canvas dimensions
        target_bin = rasterize_svg_to_canvas(target_svg_path, canvas_shape=canvas_size)

    except Exception as e:
        # Fallback if SVG parsing fails
        print(f"Failed to load target SVG: {e}")
        target_bin = np.zeros_like(user_bin)

    # Calculate SSIM score
    score, _ = ssim(user_bin, target_bin, full=True)

    # Normalize score between 0 and 100
    ssim_score = max(0.0, (score + 1.0) / 2.0 * 100.0)
    return round(ssim_score, 2)


def calculate_similarity_score(model, user_draw_tensor, target_svg_tensor):
    """
    Calculates similarity score between user canvas and target SVG.
    Input data is the standardized 4D PyTorch tensor format [N, C, H, W] with detail below:

    N: batch size (default is 1 batch),\n
    C: Channels (default is 1 channel since it in black-white), \n
    H: Height (default is 28 pixel), \n
    W: Width (default is 28 pixel), \n
    """
    with torch.no_grad():
        emb1, emb2 = model(user_draw_tensor, target_svg_tensor)

        # Calculate Euclidean Distance
        distance = F.pairwise_distance(emb1, emb2).item()

        # Convert distance to a 0-100% similarity score
        # (Assuming max acceptable distance is ~2.0)
        similarity = max(0.0, 100.0 * (1.0 - (distance / 2.0)))

        return round(similarity, 1)

# --- Quick Test ---
if __name__ == "__main__":
    # Create two synthetic resampled strokes (30 points each)
    ref_stroke = np.column_stack([np.linspace(0, 100, 30), np.zeros(30)]) # Horizontal line L->R

    # Test 1: Similar stroke (drawn left-to-right)
    user_stroke_good = ref_stroke + np.random.normal(0, 1, ref_stroke.shape)

    # Test 2: Reversed stroke (drawn right-to-left)
    user_stroke_backwards = ref_stroke[::-1] + np.random.normal(0, 1, ref_stroke.shape)

    print("Good Stroke Evaluation:", evaluate_single_stroke(user_stroke_good, ref_stroke))
    print("\nBackwards Stroke Evaluation:", evaluate_single_stroke(user_stroke_backwards, ref_stroke))