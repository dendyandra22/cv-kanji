import xml.etree.ElementTree as ET
from svgpathtools import parse_path
import numpy as np
import cv2
import numpy as np
from svgpathtools import svg2paths
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def extract_kanjivg_strokes(svg_path, num_samples=30):
    """
    Parses a KanjiVG SVG file and extracts sampled (x, y) coordinates for each stroke.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVG namespaces used by KanjiVG
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    strokes = []

    # Find all path elements (each path corresponds to a stroke in order)
    for path in root.findall('.//svg:path', ns):
        d_attribute = path.get('d')
        if not d_attribute:
            continue

        # Parse Bezier curve into continuous mathematical path
        parsed_path = parse_path(d_attribute)

        # Sample equidistant points along the stroke
        points = []
        for t in np.linspace(0, 1, num_samples):
            point = parsed_path.point(t)
            points.append([point.real, point.imag])

        strokes.append(np.array(points))

    return strokes


def svg_to_numpy(svg_path, thickness=3):
    # 1. High-resolution canvas (KanjiVG natively uses 109x109 viewbox)
    canvas = np.zeros((109, 109), dtype=np.uint8)

    # 2. Extract vector paths
    paths, _ = svg2paths(svg_path)

    # 3. Draw each vector line onto OpenCV canvas
    for path in paths:
        pts = []
        for t in np.linspace(0, 1, 30):  # Sample 30 points along each stroke path
            pt = path.point(t)
            pts.append([int(pt.real), int(pt.imag)])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=thickness)

    # 4. Downsample to 28x28 (MNIST resolution)
    resized_28x28 = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    return resized_28x28  # Returns shape (28, 28) uint8 numpy array


def numpy_to_target_tensor(np_28x28_img, device):
    # 1. Convert numpy array (28, 28) to PIL Image
    pil_img = Image.fromarray(np_28x28_img, mode='L')

    # 2. Apply PyTorch Normalization (Matches training setup!)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts pixel values from [0, 255] to [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # Scales values to [-1.0, 1.0]
    ])

    # 3. Transform & add batch dimension
    tensor_2d = transform(pil_img)  # Shape: [1, 28, 28]
    tensor_4d = tensor_2d.unsqueeze(0)  # Shape: [1, 1, 28, 28]

    return tensor_4d.to(device)


def rasterize_svg_to_image(svg_path, size=28, thickness=3, background="black"):
    """
    Renders SVG paths with anti-aliasing for smooth, non-pixelated lines.
    """
    # 1. Multiplier for supersampling (4x native 109x109 viewbox = 436x436)
    scale = 4
    high_res_size = 109 * scale

    # Set colors based on desired background
    if background == "black":
        bg_val, stroke_val = 0, 255
        canvas = np.zeros((high_res_size, high_res_size), dtype=np.uint8)
    else:
        bg_val, stroke_val = 255, 0
        canvas = np.full((high_res_size, high_res_size), bg_val, dtype=np.uint8)

    try:
        paths, _ = svg2paths(svg_path)
        for path in paths:
            pts = []
            # 2. Dense sampling along the Bézier curve (100 points per stroke)
            for t in np.linspace(0, 1, 100):
                pt = path.point(t)
                # Scale coordinates up to high-res canvas
                pts.append([int(pt.real * scale), int(pt.imag * scale)])

            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

            # 3. Draw using cv2.LINE_AA (Anti-Aliased Line)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=False,
                color=stroke_val,
                thickness=thickness * scale,  # Scale thickness proportionally
                lineType=cv2.LINE_AA
            )

    except Exception as e:
        return np.full((size, size), bg_val, dtype=np.uint8)

    # 4. Smooth downsampling to target size using INTER_AREA
    if canvas.shape[0] != size:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    return canvas


def generate_stroke_order_image(svg_path, size=300, line_thickness=4):
    """
    Renders KanjiVG SVG with stroke order color gradient (Red -> Blue)
    and numbers near stroke origins.
    """
    canvas = np.full((109, 109, 3), 255, dtype=np.uint8)  # White background (RGB)

    try:
        paths, _ = svg2paths(svg_path)
        num_strokes = len(paths)

        # # Create a smooth color map gradient (e.g., jet or rainbow: 1st stroke = Red, Last = Blue)
        # cmap = plt.get_cmap("jet")

        for idx, path in enumerate(paths):
            # # Pick RGB color along the gradient based on stroke order index
            # color_rgb = cmap(idx / max(1, num_strokes - 1))[:3]
            # color_bgr = [int(c * 255) for c in reversed(color_rgb)]
            color_bgr = 0

            # 1. Sample points along stroke path
            pts = []
            for t in np.linspace(0, 1, 60):
                pt = path.point(t)
                pts.append([int(pt.real), int(pt.imag)])

            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

            # 2. Draw smooth anti-aliased stroke line
            cv2.polylines(
                canvas,
                [pts_np],
                isClosed=False,
                color=color_bgr,
                thickness=line_thickness,
                lineType=cv2.LINE_AA
            )

            # 3. Draw stroke number near starting point of stroke
            start_x, start_y = pts[0]
            cv2.putText(
                canvas,
                str(idx + 1),
                (start_x - 4, start_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (50, 50, 50),  # Dark gray text
                1,
                cv2.LINE_AA
            )

    except Exception as e:
        return np.full((size, size, 3), 255, dtype=np.uint8)

    # Convert BGR back to RGB for display
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Smooth upscale to final UI display size
    return cv2.resize(canvas_rgb, (size, size), interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    # Replace with path to any KanjiVG SVG file you downloaded
    stroke_data = extract_kanjivg_strokes("data/kanjivg/04e00.svg")
    print(f"Total strokes extracted: {len(stroke_data)}")
    print(f"First stroke shape: {stroke_data[0].shape}")  # Should be (30, 2)