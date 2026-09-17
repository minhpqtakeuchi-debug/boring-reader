from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict
import io
import fitz  # PyMuPDF
from app.utils.helper import  read_file_as_binary
import cv2
import numpy as np
from math import atan2, degrees, hypot, ceil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import uuid

try:
    from PIL import Image  # only needed for JPEG encoding
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

def pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = 200,
    fmt: str = "png",                   # "png" or "jpg"
    save_dir: Optional[Path | str] = None,
    base_filename: Optional[str] = None,  # used for saved filenames only
    jpg_quality: int = 90
) -> List[bytes]:
    """
    Render each page of a (scanned) PDF (given as bytes) to images and return a list of image bytes.
    Optionally save images to disk as <base_filename>_p<page>.<ext> (1-based page numbering).

    Args:
        pdf_bytes: Raw bytes of the PDF.
        dpi: Output DPI (controls resolution). 72 dpi == 1.0 zoom.
        fmt: "png" (native via PyMuPDF) or "jpg".
        save_dir: If provided, images are also saved to this directory.
        base_filename: If saving, the basename to use (default: "document").
        jpg_quality: JPEG quality (1–95) if fmt="jpg".

    Returns:
        List of bytes objects, one per page, in the requested format.
    """
    fmt = fmt.lower()
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError("fmt must be 'png' or 'jpg'")

    if fmt in {"jpg", "jpeg"} and not _HAS_PIL:
        raise RuntimeError("JPEG output requires Pillow (pip install Pillow)")

    # Prepare output directory (optional)
    out_dir: Optional[Path] = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not base_filename:
            base_filename = "document"

    if not base_filename:
        base_filename = "document"

    # DPI -> zoom factor
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: List[bytes] = []

    # Open from bytes
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise RuntimeError("This PDF is encrypted and needs a password.")
        page_count = doc.page_count

        for page_index, page in enumerate(doc, start=1):
            # Render to pixmap (no alpha for cleaner JPG/PNG)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            if fmt == "png":
                img_bytes = pix.tobytes(output="png")  # PyMuPDF-native PNG
                ext = "png"
            else:
                # Convert Pixmap -> PIL Image -> JPEG bytes
                # pix.samples are RGB bytes; pix.stride is bytes per row
                mode = "RGB" if pix.n in (3, 4) else "L"
                pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
                img_bytes = buf.getvalue()
                ext = "jpg"

            images.append(img_bytes)

            # Optional save
            if out_dir is not None:
                filename = f"{base_filename}_p{page_index}.{ext}"
                (out_dir / filename).write_bytes(img_bytes)

    return images

# Bounding box type: (x_min, y_min, x_max, y_max)
BBox = Tuple[int, int, int, int]


def crop_image_array(img: np.ndarray, bbox: BBox, margin=0) -> np.ndarray:
    """
    Crop an image (NumPy array) using pixel coordinates.

    Args:
        img: HxWxC or HxW image (BGR if from cv2).
        bbox: (x_min, y_min, x_max, y_max) in *pixel coordinates*.

    Returns:
        Cropped image as NumPy array.
    """
    if img is None:
        raise ValueError("crop_image_array: img is None")

    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    # Clamp to image bounds
    x_min = int(max(0, min(x_min-margin, w)))
    x_max = int(max(0, min(x_max+margin, w)))
    y_min = int(max(0, min(y_min-margin, h)))
    y_max = int(max(0, min(y_max+margin, h)))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bbox after clamping: {bbox}")

    # NumPy slices: [y_min:y_max, x_min:x_max]
    crop = img[y_min:y_max, x_min:x_max].copy()
    return crop


def crop_image_file(
    input_path: str,
    bbox: BBox,
    margin=0,
    output_path: str = None,
) -> np.ndarray:
    """
    Load an image from disk, crop by pixel bbox, optionally save.

    Args:
        input_path: path to input image.
        bbox: (x_min, y_min, x_max, y_max) in pixels.
        output_path: where to save cropped image (if not None).

    Returns:
        Cropped image as NumPy array.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    crop = crop_image_array(img, bbox, margin=margin)

    if output_path is not None:
        cv2.imwrite(output_path, crop)

    return crop

def image_array_to_png_bytes(img: np.ndarray) -> bytes:
    """
    Encode an OpenCV image (np.ndarray, BGR) as PNG bytes.
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image to PNG")
    return buf.tobytes()

PATH = "root/boring (10).pdf"
typ = "pdf"

file = read_file_as_binary(PATH)

imgs = pdf_bytes_to_images(
    file,
    dpi=400,
    fmt="png",
    save_dir="output",  # or None to skip saving
    base_filename="my_scan"           # filename prefix for saved files
)

print(f"Returned {len(imgs)} images as bytes.")