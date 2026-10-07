from pathlib import Path

import cv2
import yaml


def preprocess_image(source: Path, destination: Path) -> tuple[int, int]:
    image = cv2.imread(str(source))

    if image is None:
        raise ValueError(f"Unable to read image: {source}")

    if not cv2.imwrite(str(destination), image):
        raise ValueError(f"Unable to write image: {destination}")

    height, width = image.shape[:2]
    return width, height


def crop_relative_roi(source: Path, destination: Path, roi: dict[str, float]) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Unable to read image: {source}")
    height, width = image.shape[:2]
    x = max(0, int(roi["x"] * width))
    y = max(0, int(roi["y"] * height))
    right = min(width, x + int(roi["width"] * width))
    bottom = min(height, y + int(roi["height"] * height))
    if right <= x or bottom <= y:
        raise ValueError(f"Invalid ROI: {roi}")
    crop = image[y:bottom, x:right]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(destination), crop)


def load_roi_config(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return raw.get("fields", {})
