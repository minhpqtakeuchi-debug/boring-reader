from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict
import io
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import re
import os
from glob import glob
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

try:
    from PIL import Image  # only needed for JPEG encoding
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def read_file_as_binary(file_path):
    """
    Reads a local file and returns its binary content.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        bytes: Binary content of the file.
    """
    path = Path(file_path)

    with open(path, "rb") as f:
        binary_data = f.read()
    return binary_data

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

def detect_orientation_tesseract(img_bgr) -> int:
    """
    Detect orientation using Tesseract OSD by trying 4 rotations.

    For each rotation `angle` in {0, 90, 180, 270}:
      - Rotate the image.
      - Run OSD on the rotated image (Tesseract returns Rotate: R).
      - Convert that to a predicted orientation for the ORIGINAL image:
            final_angle = (angle + R) % 360
      - Vote for final_angle and accumulate confidence.

    Returns the angle in {0, 90, 180, 270} that:
      - Has the highest vote count; if tied,
      - Has the highest sum of orientation confidence.
      - Falls back to 0 if OSD completely fails.
    """
     # --- 1. Remove outer white space ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Binarize: background ~ white, content ~ dark
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so content is white (255), background black (0)
    inv = 255 - bw

    # Find bounding box of non-background (content) area
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_bgr[max(y-20, 0):y + h + 20, max(x-20,0):x + w + 20]
    else:
        # If nothing detected, fall back to original image
        cropped = img_bgr

    h, w = cropped.shape[:2]
    resize_dim = (2121, 3000)
    if h*w > resize_dim[0] * resize_dim[1]:
        if h > w and h > resize_dim[0]:
            ratio = h / resize_dim[0]
            min_h = resize_dim[0]
            min_w = int(w / ratio)
            img_bgr = cv2.resize(cropped, (min_w, min_h), interpolation=cv2.INTER_CUBIC)
        else:
            ratio = w / resize_dim[1]
            min_w = resize_dim[1]
            min_h = int(h / ratio)
            img_bgr = cv2.resize(cropped, (min_w, min_h), interpolation=cv2.INTER_CUBIC)
    else:
        img_bgr = cropped
    # img_bgr = cropped
    # cv2.imwrite("Test.png", img_bgr)
    # vote counts
    rotations: Dict[int, int] = {0: 0, 90: 0, 180: 0, 270: 0}
    # sum of confidences
    probs: Dict[int, float] = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}

    for angle in (0, 90, 180, 270):
        # Rotate image by angle
        if angle == 90:
            rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:  # 0
            rotated = img_bgr

        # Tesseract expects RGB
        img_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

        try:
            osd = pytesseract.image_to_osd(img_rgb)
        except Exception as e:
            # If OSD fails for this rotation, skip it
            # print(f"OSD failed for angle {angle}: {e}")
            continue

        # Example OSD lines:
        # "Rotate: 90\nOrientation confidence: 12.34\n..."
        match = re.search(r"Rotate:\s+(\d+)", osd)
        conf_match = re.search(r"Orientation confidence:\s+([\d.]+)", osd)

        if not match:
            continue

        rotate = int(match.group(1)) % 360
        if rotate not in (0, 90, 180, 270):
            continue

        # Map back to orientation of the *original* image
        final_angle = (angle + rotate) % 360

        if final_angle in rotations:
            rotations[final_angle] += 1
            if conf_match:
                conf = float(conf_match.group(1))
                probs[final_angle] += conf
                if conf > 3.0:
                    return final_angle

    # If everything failed, default to 0°
    if all(v == 0 for v in rotations.values()):
        return 0

    # print(rotations)
    # print(probs)
    avgs = {}
    for r in rotations:
        avgs[r] = probs[r] / rotations[r] if rotations[r] > 0 else 0
    # print(avgs)
    # 1) pick orientation with the most votes
    max_votes = max(avgs.values())
    candidates = [a for a, v in avgs.items() if v == max_votes]

    # 2) if tie, pick the one with the highest summed confidence
    if len(candidates) == 1:
        return candidates[0]

    best_angle = max(candidates, key=lambda a: rotations[a])
    return best_angle

def auto_rotate_image(img_bgr):
    """
    Detect orientation and return (rotated_image, angle).
    """
    angle = detect_orientation_tesseract(img_bgr)

    if angle == 90:
        rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated = img_bgr.copy()

    return rotated, angle

def test_orientation_on_folder(pattern: str):
    paths = sorted(glob(pattern))
    print("Pattern:", pattern)
    print("Found files:", paths)
    if not paths:
        print("No images found. Check folder and pattern.")
        return

    for path in paths:
        print("=" * 60)
        print("Image:", path)
        img = cv2.imread(path)

        if img is None:
            print("  ERROR: cv2.imread returned None. Check that this file exists and is an image.")
            continue

        rotated, angle = auto_rotate_image(img)
        print(f"  Detected rotation: {angle} degrees")

        base, ext = os.path.splitext(path)
        out_path = f"{base}_rotated.png"
        cv2.imwrite(out_path, rotated)
        print("  Saved rotated image to:", out_path)

def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes (e.g. PNG bytes) to an OpenCV BGR image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes to cv2 image")
    return img

labels = ["head", "body", "other"]

def get_boring_pdf(file_path: str):
    """
    Read a PDF, classify each page as head/body/other, and group consecutive
    head + following body pages into a list of boring logs.

    Returns:
        List of groups, e.g.:
        [
          [head1_img],
          [head2_img],
          [head3_img, body3_1_img, body3_2_img],
          ...
        ]
    where each item is a rotated OpenCV BGR image (np.ndarray).
    """
    # 1) Load classifier model once
    model = YOLO("yolo/models/boring_cls.pt")

    # 2) Read PDF as bytes and convert to image BYTES per page
    file = read_file_as_binary(file_path)
    img_bytes_list = pdf_bytes_to_images(
        file,
        dpi=400,
        fmt="png",
    )

    print(f"Returned {len(img_bytes_list)} images from PDF.")

    boring_groups: List[List[np.ndarray]] = []
    current_group: List[np.ndarray] = []

    for page_idx, img_bytes in enumerate(img_bytes_list):
        # convert bytes -> cv2 image (BGR)
        img_bgr = bytes_to_cv2_image(img_bytes)

        # auto-rotate page (no saving)
        rotated, angle = auto_rotate_image(img_bgr)
        print(f"\n[Page {page_idx}] Detected rotation: {angle} degrees")

        # run classifier on the rotated image
        results = model(rotated, verbose=False)  # or model(rotated, device="cuda"/"cpu")
        r = results[0]

        probs = r.probs
        cls_id = int(probs.top1)
        conf = float(probs.top1conf)
        name_map = r.names  # {0: 'head', 1: 'body', 2: 'other'} depending on dataset
        label = name_map[cls_id]

        print(f"[Page {page_idx}] Predicted: {label} (class {cls_id}) conf={conf:.3f}")

        # grouping logic
        if label == "head":
            # if we were collecting a group, close it
            if current_group:
                boring_groups.append(current_group)
            # start a new group with this head
            current_group = [rotated]

        elif label == "body":
            # only attach body if there's a previous head
            if current_group:
                current_group.append(rotated)
            else:
                print(f"[Page {page_idx}] 'body' without preceding 'head' -> ignored")

        else:  # "other"
            print(f"[Page {page_idx}] 'other' -> ignored")

    # flush last group if exists
    if current_group:
        boring_groups.append(current_group)

    print(f"Found {len(boring_groups)} groups")
    return boring_groups