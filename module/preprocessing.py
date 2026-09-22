import numpy as np


def normalize_character_bbox(strokes, target_size=256.0):
    """
    Rescales and centers all strokes to fit inside a target_size x target_size box.
    """
    # Combine all points across strokes to find global bounding box
    all_points = np.vstack(strokes)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = max_x - min_x
    height = max_y - min_y
    scale = target_size / max(width, height, 1e-5)

    normalized_strokes = []
    for stroke in strokes:
        # Center and scale
        norm_stroke = (stroke - [min_x, min_y]) * scale
        normalized_strokes.append(norm_stroke)

    return normalized_strokes


def resample_stroke_spatial(stroke, num_points=30):
    """
    Resamples a single stroke (N x 2 array) so points are evenly spaced along path length.
    """
    # Calculate cumulative distance along path
    distances = np.sqrt(np.sum(np.diff(stroke, axis=0) ** 2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0.0)
    total_length = cumulative_dist[-1]

    if total_length == 0:
        return np.tile(stroke[0], (num_points, 1))

    # Interpolate x and y at uniform distance steps
    uniform_dists = np.linspace(0, total_length, num_points)
    resampled_x = np.interp(uniform_dists, cumulative_dist, stroke[:, 0])
    resampled_y = np.interp(uniform_dists, cumulative_dist, stroke[:, 1])

    return np.column_stack([resampled_x, resampled_y])


# --- Quick Test ---
if __name__ == "__main__":
    # Simulated raw stroke input (e.g., fast drawing with 5 points)
    raw_user_stroke = np.array([[10, 10], [15, 40], [20, 80], [22, 120], [25, 150]])

    # Process
    resampled = resample_stroke_spatial(raw_user_stroke, num_points=30)
    print(f"Resampled stroke shape: {resampled.shape}")  # (30, 2)