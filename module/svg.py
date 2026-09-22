import xml.etree.ElementTree as ET
from svgpathtools import parse_path
import numpy as np

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

if __name__ == "__main__":
    # Replace with path to any KanjiVG SVG file you downloaded
    stroke_data = extract_kanjivg_strokes("data/kanjivg/04e00.svg")
    print(f"Total strokes extracted: {len(stroke_data)}")
    print(f"First stroke shape: {stroke_data[0].shape}")  # Should be (30, 2)