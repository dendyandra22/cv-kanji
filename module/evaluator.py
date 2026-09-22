import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

    # 2. Direction Check
    dir_sim = evaluate_stroke_direction(user_stroke, ref_stroke)

    # Penalize direction heavily if drawn backwards
    direction_correct = dir_sim > 0.0
    direction_score = max(0.0, dir_sim * 100.0)

    return {
        "dtw_distance": float(dtw_dist),
        "shape_score": -1, #round(shape_score, 5),
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

    return {
        "count_match": count_match,
        "user_stroke_count": num_user,
        "target_stroke_count": num_ref,
        "avg_shape_score": avg_shape, #round(avg_shape, 2),
        "avg_direction_score": round(avg_dir, 2),
        "stroke_breakdown": results
    }

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