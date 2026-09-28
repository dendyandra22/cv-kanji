import json
from pathlib import Path

from .utility import print_log

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KANJI_SVG_DIR = DATA_DIR / "kanjivg"

def load_kanji_metadata(filename: str|Path, verbose: bool = False) -> list[dict]:
    filename = Path(filename) if isinstance(filename, str) else filename
    print(filename, type(filename))
    if filename.suffix != ".json":
        filename = filename.with_suffix(".json")

    file_path = DATA_DIR / f"kanji_level/{filename}"
    with open(file_path) as f:
        data = json.load(f)
    print_log(f"Loaded {filename}: {len(data)} items")

    # Validate missing data
    missing_data = []
    for item in data:
        unicode = item["unicode"].replace("U+","0")
        if not Path(KANJI_SVG_DIR / f"{unicode}.svg").exists():
            missing_data.append(item)
            data.remove(item)
    print_log(f"{len(missing_data)} missing data skipped...")
    print(missing_data)

    return data

if __name__ == "__main__":
    data = load_kanji_metadata("n5")
    data[:3]