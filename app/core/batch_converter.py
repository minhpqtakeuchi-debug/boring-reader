import json
import time
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import ValidationError

import cv2
import numpy as np
from google import genai
from google.genai import types
from ultralytics import YOLO
from math import atan2, degrees

from app.core.config import settings
from app.shared.dto import BoreholeLogEnvelope

# ----------------------------
# CONFIG
# ----------------------------
API_KEY = settings.api_key
TEMPERATURE = 1.0
TIMEOUT = 90_000  # no longer used directly for batch, but kept for completeness

# YOLO model + instruction / label maps
instructions_map = {
    0: "yolo/instructions/boring_instruction_header.txt",
    1: "yolo/instructions/boring_instruction_soil.txt",
    2: "yolo/instructions/boring_instruction_spt.txt",
}

label_map = {
    0: "header_region",
    1: "soil_region",
    2: "spt_region",
}

# Load YOLO model (adjust path to your weights)
model = YOLO("yolo_borehole_regions/yolov8n_regions_new/weights/best.pt")


# ----------------------------
# UTIL: load instruction text
# ----------------------------
def load_instruction_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# UTIL: strip code fences and parse JSON
# ----------------------------
def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a string that may contain markdown fences.
    """
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        payload = fence_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", text, flags=re.S)
        if not brace_match:
            raise ValueError("No JSON object found in the model output.")
        payload = brace_match.group(1)

    return json.loads(payload)


def _extract_text_from_gemini3_response(response: Any) -> str:
    """
    Extract text from a GenerateContentResponse in a way that works for Gemini 3.
    Priority:
      1. response.text
      2. response.output[*].content[*].text
      3. response.candidates[*].content.parts[*].text
    """
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    texts: List[str] = []

    # Gemini 3 style
    output = getattr(response, "output", None)
    if output:
        for out in output:
            content = getattr(out, "content", None)
            if not content:
                continue
            for part in content:
                t = getattr(part, "text", None)
                if isinstance(t, str) and t:
                    texts.append(t)
        if texts:
            return "\n".join(texts).strip()

    # Older candidate/parts style
    candidates = getattr(response, "candidates", None)
    if candidates:
        for c in candidates:
            content = getattr(c, "content", None)
            if not content:
                continue
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    texts.append(t)
        if texts:
            return "\n".join(texts).strip()

    return ""


# ----------------------------
# IMAGE HELPERS
# ----------------------------
def crop_image_array(
    img_bgr: np.ndarray,
    box_xyxy: List[int],
    margin: int = 0,
) -> np.ndarray:
    """
    Crop image to [xmin, ymin, xmax, ymax] box (in pixels) with optional margin.
    """
    x1, y1, x2, y2 = box_xyxy
    h, w = img_bgr.shape[:2]

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return img_bgr[y1:y2, x1:x2]


def image_array_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    """
    Encode a BGR image to PNG bytes.
    """
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return buf.tobytes()


def crop_img_to_png_bytes(img: np.ndarray, box_xyxy):
    """
    Crop image to [xmin, ymin, xmax, ymax] box (in pixels)
    and return PNG bytes for LLM.
    """
    img_bit = crop_image_array(img, box_xyxy, margin=0)
    return image_array_to_png_bytes(img_bit)


# ----------------------------
# YOLO REGION DETECTION (SEQUENTIAL)
# ----------------------------
def detect_regions_for_image(
    img_bgr: np.ndarray,
) -> List[Tuple[str, bytes, str]]:
    """
    Run YOLO on a single image (BGR), detect header/soil/spt regions,
    and return a list of:
        [(label, image_bytes, instruction_path), ...]
    where label in {"header_region", "soil_region", "spt_region"}.
    """
    results = model(img_bgr, imgsz=512, conf=0.25, save=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        raise RuntimeError("No detections from YOLO for this image")

    img = r.orig_img  # BGR

    bboxes_xyxy = r.boxes.xyxy.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    confidences = r.boxes.conf.cpu().numpy()

    desired_classes = {0, 1, 2}
    best_idx_per_class: Dict[int, int] = {}
    best_conf_per_class: Dict[int, float] = {}

    for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
        if cls_id not in desired_classes:
            continue
        if cls_id not in best_idx_per_class or conf > best_conf_per_class[cls_id]:
            best_idx_per_class[cls_id] = i
            best_conf_per_class[cls_id] = conf

    missing = desired_classes - set(best_idx_per_class.keys())
    if missing:
        raise RuntimeError(
            f"YOLO did not detect all required regions. Missing classes: {sorted(missing)}"
        )

    jobs: List[Tuple[str, bytes, str]] = []

    print("Detections after filtering (one per class):")
    for cls_id in sorted(desired_classes):
        idx = best_idx_per_class[cls_id]
        box = bboxes_xyxy[idx].astype(int).tolist()
        label = label_map[cls_id]
        instruction_path = instructions_map[cls_id]

        print(
            f"{label} (class {cls_id}, conf={best_conf_per_class[cls_id]:.3f}): "
            f"[xmin, ymin, xmax, ymax] = {box}"
        )

        image_bytes = crop_img_to_png_bytes(img, box)
        jobs.append((label, image_bytes, instruction_path))

    return jobs


# ----------------------------
# DESKEW HELPERS (unchanged)
# ----------------------------
def estimate_skew_from_vertical_lines(
    bw: np.ndarray,
    max_vertical_deviation: float = 10.0,
    min_frac_of_height: float = 0.4,
) -> float:
    """
    Estimate page skew (in degrees) using vertical-ish lines in a binary image.
    """
    if len(bw.shape) == 3:
        bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)

    h, w = bw.shape[:2]

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)

    min_line_length = int(h * min_frac_of_height)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines_p is None:
        return 0.0

    vertical_deviations: List[float] = []

    for (x1, y1, x2, y2) in lines_p[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        angle = degrees(atan2(dy, dx))  # vs x-axis

        dev_pos = abs(angle - 90.0)
        dev_neg = abs(angle + 90.0)
        dev = min(dev_pos, dev_neg)

        if dev <= max_vertical_deviation:
            if dev_pos < dev_neg:
                vertical_deviation = angle - 90.0
            else:
                vertical_deviation = angle + 90.0
            vertical_deviations.append(vertical_deviation)

    if not vertical_deviations:
        return 0.0

    skew_deg = float(np.median(vertical_deviations))
    return skew_deg


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate an image around its center by `angle_deg` degrees.
    Positive angle = counter-clockwise (OpenCV convention).
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def deskew_image_by_vertical(
    img: np.ndarray,
    max_vertical_deviation: float = 10.0,
    min_frac_of_height: float = 0.4,
):
    """
    Deskew an image using vertical lines in a binary version.
    Returns: rotated_img, skew_deg
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, bw = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    skew_deg = estimate_skew_from_vertical_lines(
        bw,
        max_vertical_deviation=max_vertical_deviation,
        min_frac_of_height=min_frac_of_height,
    )

    # print(f"Estimated skew from vertical lines (avg): {skew_deg:.3f} degrees")

    rotated = rotate_image(img, skew_deg * 0.5)
    return rotated, skew_deg


# ---------------------------------------------------------------------
# Helper: build a single inline GenerateContentRequest for batch mode
# ---------------------------------------------------------------------
def _build_inline_request_for_region(
    image_bytes: bytes,
    instruction_text: str,
    image_mime_type: str = "image/png",
    max_output_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Build a GenerateContentRequest dict suitable for BatchJobSource.inline_requests.
    """
    img_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=image_mime_type,
    )

    gen_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        max_output_tokens=max_output_tokens,
        system_instruction=instruction_text,
    )

    inline_request: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [img_part],
            }
        ],
        "config": gen_config,
    }

    return inline_request

def _build_inline_request_for_multi_images(
    image_bytes_list: Sequence[bytes],
    instruction_text: str,
    image_mime_type: str = "image/png",
    max_output_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Build a GenerateContentRequest dict for a MULTI-IMAGE batch request.

    Equivalent to calling a multi-image transform_images_to_json():
        images -> [Part.from_bytes(...), Part.from_bytes(...), ...]
    """
    parts = [
        types.Part.from_bytes(
            data=bytes(b),
            mime_type=image_mime_type,
        )
        for b in image_bytes_list
    ]

    gen_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        max_output_tokens=max_output_tokens,
        system_instruction=instruction_text,
    )

    inline_request: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": parts,  # multi-image input
            }
        ],
        "config": gen_config,
    }
    return inline_request


# ---------------------------------------------------------------------
# Function 1: create FIRST-WAVE batch (Gemini-3 only)
# ---------------------------------------------------------------------
def create_boring_batches_for_images(
    image_groups: Sequence[Sequence[np.ndarray]],
    max_output_tokens_g3: int = 4096,
    multi_instruction_path: str = "boring_instructions.txt",
) -> Dict[str, Any]:
    """
    WAVE 1, with grouped images:

      image_groups: Sequence[Sequence[np.ndarray]]
        - If len(group) == 1:
            * Run YOLO on that single image to detect header/soil/spt regions.
            * Build ONE inline request per region (same as before).
        - If len(group) > 1:
            * DO NOT run YOLO.
            * Treat the entire group as a single multi-image borehole log.
            * Build ONE inline request with all images as parts, using
              `multi_instruction_path`. Max tokens = max_output_tokens_g3 * 3.

    Returns a "batch_plan" dict containing:
      {
        "gem3_batch_name": str or None,
        "region_infos": [
          # Region-style entry (single-image, YOLO crops):
          {
            "kind": "region",
            "img_idx": int,
            "label": "header_region" | "soil_region" | "spt_region",
            "g3_index": int,            # index into gem3 batch responses
            "image_bytes": bytes,       # used for wave-2 retry
            "instruction_path": str,    # used for wave-2 retry
          },

          # Multi-image entry (group with len > 1):
          {
            "kind": "multi",
            "img_idx": int,
            "label": "full_borelog",
            "g3_index": int,
            "multi_image_bytes": [bytes, bytes, ...],  # all pages
            "instruction_path": str,    # e.g. multi_instruction_path
          },
          ...
        ],
        "num_images": int   # == len(image_groups)
      }

    NOTE: region_infos is NOT JSON-serializable because it contains bytes,
    so you should pickle it if you need to persist it.
    """
     # 1) Deskew each image in each group
    deskewed_image_groups = [
        [
            deskew_image_by_vertical(
                img,
                max_vertical_deviation=10.0,
                min_frac_of_height=0.4,
            )[0]
            for img in group
        ]
        for group in image_groups
    ]

    client = genai.Client(api_key=API_KEY)
    num_groups = len(deskewed_image_groups)

    gem3_inline_requests: List[Dict[str, Any]] = []
    region_infos: List[Dict[str, Any]] = []

    for group_idx, group_imgs in enumerate(deskewed_image_groups):
        if not group_imgs:
            print(f"[WARN] Image group {group_idx + 1} is empty; skipping.")
            continue

        # -------- CASE A: group has EXACTLY ONE image -> old YOLO behavior --------
        if len(group_imgs) == 1:
            img = group_imgs[0]
            print(f"\n=== YOLO detection for group {group_idx + 1}/{num_groups} (single image) ===")
            try:
                jobs = detect_regions_for_image(img)
            except Exception as e:
                print(f"[ERROR] YOLO failed for group {group_idx + 1}: {e}")
                jobs = []

            # jobs: [(label, image_bytes, instruction_path), ...]
            for (label, image_bytes, instruction_path) in jobs:
                instruction_text = load_instruction_text(instruction_path).strip()

                g3_req = _build_inline_request_for_region(
                    image_bytes=image_bytes,
                    instruction_text=instruction_text,
                    max_output_tokens=max_output_tokens_g3,
                )
                g3_index = len(gem3_inline_requests)
                gem3_inline_requests.append(g3_req)

                region_infos.append(
                    {
                        "kind": "region",
                        "img_idx": group_idx,
                        "label": label,
                        "g3_index": g3_index,
                        "image_bytes": image_bytes,
                        "instruction_path": instruction_path,
                    }
                )

        # -------- CASE B: group has MULTIPLE images -> multi-image batch --------
        else:
            print(
                f"\n=== Multi-image group {group_idx + 1}/{num_groups} "
                f"with {len(group_imgs)} pages ==="
            )

            # Convert each page to PNG bytes
            multi_bytes: List[bytes] = [
                image_array_to_png_bytes(img_bgr) for img_bgr in group_imgs
            ]
            instruction_text = load_instruction_text(multi_instruction_path).strip()

            # Max tokens for multi-image = 3x single
            multi_max_tokens = max_output_tokens_g3 * 3

            g3_req = _build_inline_request_for_multi_images(
                image_bytes_list=multi_bytes,
                instruction_text=instruction_text,
                max_output_tokens=multi_max_tokens,
            )
            g3_index = len(gem3_inline_requests)
            gem3_inline_requests.append(g3_req)

            # Store a single "full borelog" region for this group
            region_infos.append(
                {
                    "kind": "multi",
                    "img_idx": group_idx,
                    "label": "full_borelog",
                    "g3_index": g3_index,
                    "multi_image_bytes": multi_bytes,
                    "instruction_path": multi_instruction_path,
                }
            )

    gem3_batch_name: Optional[str] = None

    if gem3_inline_requests:
        gem3_source = {"inlined_requests": gem3_inline_requests}
        gem3_batch = client.batches.create(
            model="models/gemini-3-pro-preview",
            src=gem3_source,
        )
        gem3_batch_name = gem3_batch.name
        print(f"Created Gemini-3 batch (wave 1): {gem3_batch_name}")
    else:
        print("[WARN] No inline requests to submit for Gemini-3.")

    batch_plan = {
        "gem3_batch_name": gem3_batch_name,
        "region_infos": region_infos,
        "num_images": num_groups,
        # mark explicitly this is wave 1
        "wave": 1,
    }
    return batch_plan



# ---------------------------------------------------------------------
# Single-shot batch status check (no loop / no waiting)
# ---------------------------------------------------------------------
def _wait_for_batch_completion(
    client: genai.Client,
    batch_name: str,
):
    """
    Single-shot batch status check.

    - Fetch the batch once.
    - Print its current state.
    - If state == JOB_STATE_SUCCEEDED: return the Batch object.
    - Otherwise: return None.

    NOTE:
      This function no longer waits or loops. Callers must check for
      None and avoid parsing results if the batch is not succeeded.
    """
    batch = client.batches.get(name=batch_name)
    state = getattr(batch, "state", None)
    state_name = getattr(state, "name", str(state))

    print(f"[BATCH] {batch_name} state: {state_name}")

    if state == types.JobState.JOB_STATE_SUCCEEDED:
        return batch

    # Not succeeded yet (RUNNING, PENDING, FAILED, etc.)
    return None

# ---------------------------------------------------------------------
# Batch → JSON helper
# ---------------------------------------------------------------------
def validate_borehole_log_dict(obj: Dict[str, Any]) -> bool:
    """
    Validate a dict against the BoreholeLogEnvelope schema.

    Works for:
      - header-only responses
      - soil-only responses
      - spt-only responses
      - full borehole_log responses.

    If the object looks like a JSON *schema* (only $schema / properties, and
    no top-level "borehole_log"), we treat it as invalid data and return False
    so the fallback logic can handle it.
    """
    if not isinstance(obj, dict):
        print("[WARN] validate_borehole_log_dict: input is not a dict")
        return False

    data = dict(obj)
    # Ignore our own usage metadata
    data.pop("_usage", None)

    # If there's no 'borehole_log' at the top level, this is probably the JSON schema,
    # not an instance. Don't feed it to Pydantic; just mark it invalid.
    if "borehole_log" not in data:
        print("[WARN] Top-level JSON has no 'borehole_log' key; "
              "treating as schema / invalid instance.")
        return False

    try:
        BoreholeLogEnvelope.model_validate(data)
        return True
    except ValidationError as e:
        print("[WARN] Schema validation failed:", e)
        return False

    
def extract_jsons_from_batch(batch: Any) -> List[Optional[Dict[str, Any]]]:
    """
    Given a BatchJob that has SUCCEEDED, pull out one JSON dict
    (or None) per inlined response.

    For each response:
      - skip if finish_reason != STOP
      - extract text
      - parse JSON
      - attach usage metadata as "_usage"
      - validate against BoreholeLogEnvelope (partial schema)
      - if validation fails => return None for that index

    This supports:
      * header-only JSON
      * soil-only JSON
      * spt-only JSON
      * full borehole_log JSON (multi-image).
    """
    dest = getattr(batch, "dest", None)
    if dest is None:
        print("[WARN] Batch has no dest")
        return []

    inlined = getattr(dest, "inlined_responses", None)
    if not inlined:
        print("[WARN] Batch has no inlined_responses")
        return []

    results: List[Optional[Dict[str, Any]]] = []

    for i, inline_resp in enumerate(inlined):
        resp = getattr(inline_resp, "response", None)
        if resp is None:
            print(f"[WARN] No .response for inline_responses[{i}]")
            results.append(None)
            continue

        # --- check finish_reason ---
        finish_reason = None
        candidates = getattr(resp, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)

        if finish_reason is not None and finish_reason != types.FinishReason.STOP:
            print(
                f"[WARN] inline_responses[{i}] finished with {finish_reason}, "
                "treating as unusable (likely truncated / MAX_TOKENS)."
            )
            results.append(None)
            continue

        # --- extract text ---
        raw_text = _extract_text_from_gemini3_response(resp)
        if not raw_text:
            print(f"[WARN] No text for inline_responses[{i}]")
            results.append(None)
            continue

        # --- parse JSON ---
        try:
            parsed = extract_json(raw_text)
        except Exception as e:
            print(f"[WARN] Failed to parse JSON for inline_responses[{i}]: {e}")
            print("  Raw (first 400 chars):", raw_text[:400])
            results.append(None)
            continue

        # --- attach usage if present ---
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            parsed["_usage"] = {
                "input_tokens": getattr(usage_meta, "prompt_token_count", None),
                "output_tokens": getattr(usage_meta, "candidates_token_count", None),
                "total_tokens": getattr(usage_meta, "total_token_count", None),
            }

        # --- validate against partial BoreholeLog schema ---
        if not validate_borehole_log_dict(parsed):
            print(f"[WARN] Schema validation failed for inline_responses[{i}], marking as invalid.")
            results.append(None)
            continue

        results.append(parsed)

    return results



# ---------------------------------------------------------------------
# Function 2: collect both waves and build final JSON OR wave-2 plan
# ---------------------------------------------------------------------
def collect_boring_batches_results(
    batch_plan: Dict[str, Any],
    max_output_tokens_g3_retry: int = 4096,
    max_output_tokens_flash: int = 2048,
) -> Dict[str, Any]:
    """
    2-WAVE COLLECTION via a single function.

    Supports both:
      - Old plan format: region-only (header/soil/spt) with single images.
      - New plan format: mixed region + multi-image ("full_borelog") entries.

    BEHAVIOR BY WAVE:

    WAVE 1 (default, if batch_plan["wave"] is missing or == 1):
      - Check the Gemini-3 batch in `batch_plan["gem3_batch_name"]`.
      - Parse each inline response with `extract_jsons_from_batch`.
      - Fill per-borelog results where JSON is valid.
        * kind == "region": only header/soil/spt for that part.
        * kind == "multi" or label == "full_borelog": fill all fields from one JSON.
      - Collect remaining invalid regions.
      - If NO invalid regions:
          -> return final JSON:
              {"borelog": [ {header_info, soil_layers, ...}, ... ]}
      - If there ARE invalid regions:
          -> create wave-2 batches:
                * Gemini-3 RETRY batch (single or multi-image)
                * Gemini-2.5-flash fallback batch (single or multi-image)
             and return a WAVE-2 plan dict:
              {
                "wave": 2,
                "num_images": ...,
                "wave1_results": [...],  # partial JSON from wave 1
                "invalid_regions": [...],# minimal mapping for wave 2
                "gem3_retry_batch_name": "...",
                "flash_batch_name": "...",
              }

    WAVE 2 (batch_plan["wave"] == 2):
      - Check wave-2 Gemini-3 RETRY and Flash batches.
      - For each invalid region:
          * try Gemini-3 RETRY
          * if unusable, try Flash and mark *_fall_back = True
        (for multi-image, mark all three fallback flags if Flash is used)
      - Merge into stored wave1_results and return final JSON:
          {"borelog": [...]}.
    """
    client = genai.Client(api_key=API_KEY)

    wave = batch_plan.get("wave", 1)

    # -------------------------------------------------
    # ===============  WAVE 1 BRANCH  =================
    # -------------------------------------------------
    if wave == 1:
        gem3_batch_name: Optional[str] = batch_plan.get("gem3_batch_name")
        region_infos: List[Dict[str, Any]] = batch_plan.get("region_infos", [])
        num_images: int = batch_plan.get("num_images", 0)

        # --- Prepare result containers (one per group/borelog) ---
        results: List[Dict[str, Any]] = [
            {
                "header_info": None,
                "soil_layers": None,
                "spt_tests": None,
                "header_fall_back": False,
                "soil_fall_back": False,
                "spt_fall_back": False,
            }
            for _ in range(num_images)
        ]

        # ========== WAVE 1: Gemini-3 batch ==========
        gem3_results: List[Optional[Dict[str, Any]]] = []

        if gem3_batch_name:
            print(f"\n[WAVE 1] Checking Gemini-3 batch: {gem3_batch_name}")
            gem3_batch = _wait_for_batch_completion(client, gem3_batch_name)
            if gem3_batch is None:
                # Batch not yet succeeded; caller should call again later
                return {"borelog": []}
            gem3_results = extract_jsons_from_batch(gem3_batch)
        else:
            print("[WARN] No Gemini-3 batch name in batch_plan for wave 1.")

        invalid_regions: List[Dict[str, Any]] = []

        # Fill what we can from wave 1, record invalid
        for region in region_infos:
            img_idx = region["img_idx"]
            label = region.get("label")
            kind = region.get("kind", "region")  # default to "region" for old plans
            g3_index = region["g3_index"]

            result_obj: Optional[Dict[str, Any]] = None

            if 0 <= g3_index < len(gem3_results):
                result_obj = gem3_results[g3_index]

            if result_obj is None:
                print(f"[INFO] Wave-1 invalid/None for img {img_idx+1}, region {label}")
                # keep the full region info here; we need bytes & instruction path for wave 2
                invalid_regions.append(region)
                continue

            payload = result_obj.get("borehole_log", result_obj)

            # --- MULTI-IMAGE or full-borelog entry ---
            if kind == "multi" or label == "full_borelog":
                results[img_idx]["header_info"] = payload.get("header_info")
                results[img_idx]["soil_layers"] = payload.get("soil_layers")
                results[img_idx]["spt_tests"] = payload.get("spt_tests")

            # --- REGION-style entry (header/soil/spt separately) ---
            else:
                if label == "header_region":
                    results[img_idx]["header_info"] = payload.get("header_info")
                elif label == "soil_region":
                    results[img_idx]["soil_layers"] = payload.get("soil_layers")
                elif label == "spt_region":
                    results[img_idx]["spt_tests"] = payload.get("spt_tests")

        # If nothing is invalid, we're done with final JSON
        if not invalid_regions:
            print("[WAVE 1] All regions/groups succeeded with Gemini-3. No second wave needed.")
            print("All batch jobs processed.")
            return {"borelog": results}

        # ======================
        # WAVE 2 PREP: build retry batches only for invalid regions
        # ======================
        print(f"\n[WAVE 2] Building retry batches for {len(invalid_regions)} invalid regions...")

        gem3_retry_inline: List[Dict[str, Any]] = []
        flash_inline: List[Dict[str, Any]] = []
        # We return a *minimal* invalid_regions list for wave 2 (no raw bytes).
        invalid_regions_wave2: List[Dict[str, Any]] = []

        for r in invalid_regions:
            img_idx = r["img_idx"]
            label = r.get("label")
            kind = r.get("kind", "region")

            if kind == "multi" or label == "full_borelog":
                # MULTI-IMAGE retry
                image_bytes_list = r["multi_image_bytes"]
                instruction_path = r["instruction_path"]
                instruction_text = load_instruction_text(instruction_path).strip()

                # Multi-image retry max tokens = 3x
                g3_retry_req = _build_inline_request_for_multi_images(
                    image_bytes_list=image_bytes_list,
                    instruction_text=instruction_text,
                    max_output_tokens=max_output_tokens_g3_retry * 3,
                )
                g3_retry_index = len(gem3_retry_inline)
                gem3_retry_inline.append(g3_retry_req)

                flash_req = _build_inline_request_for_multi_images(
                    image_bytes_list=image_bytes_list,
                    instruction_text=instruction_text,
                    max_output_tokens=max_output_tokens_flash * 3,
                )
                flash_index = len(flash_inline)
                flash_inline.append(flash_req)

            else:
                # REGION-style retry (single image)
                image_bytes = r["image_bytes"]
                instruction_path = r["instruction_path"]
                instruction_text = load_instruction_text(instruction_path).strip()

                g3_retry_req = _build_inline_request_for_region(
                    image_bytes=image_bytes,
                    instruction_text=instruction_text,
                    max_output_tokens=max_output_tokens_g3_retry,
                )
                g3_retry_index = len(gem3_retry_inline)
                gem3_retry_inline.append(g3_retry_req)

                flash_req = _build_inline_request_for_region(
                    image_bytes=image_bytes,
                    instruction_text=instruction_text,
                    max_output_tokens=max_output_tokens_flash,
                )
                flash_index = len(flash_inline)
                flash_inline.append(flash_req)

            # Store ONLY the mapping info for wave 2 (no bytes)
            invalid_regions_wave2.append(
                {
                    "img_idx": img_idx,
                    "label": label,
                    "kind": kind,
                    "g3_retry_index": g3_retry_index,
                    "flash_index": flash_index,
                }
            )

        gem3_retry_batch_name: Optional[str] = None
        flash_batch_name: Optional[str] = None

        if gem3_retry_inline:
            gem3_retry_source = {"inlined_requests": gem3_retry_inline}
            gem3_retry_batch = client.batches.create(
                model="models/gemini-3-pro-preview",
                src=gem3_retry_source,
            )
            gem3_retry_batch_name = gem3_retry_batch.name
            print(f"Created Gemini-3 RETRY batch: {gem3_retry_batch_name}")

        if flash_inline:
            flash_source = {"inlined_requests": flash_inline}
            flash_batch = client.batches.create(
                model="models/gemini-2.5-flash",
                src=flash_source,
            )
            flash_batch_name = flash_batch.name
            print(f"Created Flash fallback batch: {flash_batch_name}")

        # Return a WAVE-2 PLAN instead of final JSON
        wave2_plan: Dict[str, Any] = {
            "wave": 2,
            "num_images": num_images,
            # partial results from wave 1; will be completed in wave 2
            "wave1_results": results,
            # minimal mapping of which regions are still invalid
            "invalid_regions": invalid_regions_wave2,
            # new batch names for wave 2
            "gem3_retry_batch_name": gem3_retry_batch_name,
            "flash_batch_name": flash_batch_name,
        }
        print("[WAVE 2] Wave-2 plan created and returned.")
        return wave2_plan

    # -------------------------------------------------
    # ===============  WAVE 2 BRANCH  =================
    # -------------------------------------------------
    elif wave == 2:
        print("\n[WAVE 2] Collecting results and merging with wave-1 JSON...")

        num_images: int = batch_plan.get("num_images", 0)
        wave1_results: List[Dict[str, Any]] = batch_plan.get("wave1_results", [])
        invalid_regions: List[Dict[str, Any]] = batch_plan.get("invalid_regions", [])

        gem3_retry_batch_name: Optional[str] = batch_plan.get("gem3_retry_batch_name")
        flash_batch_name: Optional[str] = batch_plan.get("flash_batch_name")

        # If wave1_results is missing/short, rebuild empty shells
        if not wave1_results or len(wave1_results) != num_images:
            wave1_results = [
                {
                    "header_info": None,
                    "soil_layers": None,
                    "spt_tests": None,
                    "header_fall_back": False,
                    "soil_fall_back": False,
                    "spt_fall_back": False,
                }
                for _ in range(num_images)
            ]

        results = wave1_results  # we will modify in-place

        # --- Check WAVE 2 batches ---
        gem3_retry_results: List[Optional[Dict[str, Any]]] = []
        flash_results: List[Optional[Dict[str, Any]]] = []

        gem3_retry_batch = None
        flash_batch = None

        if gem3_retry_batch_name:
            print(f"[WAVE 2] Checking Gemini-3 RETRY batch: {gem3_retry_batch_name}")
            gem3_retry_batch = _wait_for_batch_completion(client, gem3_retry_batch_name)

        if flash_batch_name:
            print(f"[WAVE 2] Checking Flash fallback batch: {flash_batch_name}")
            flash_batch = _wait_for_batch_completion(client, flash_batch_name)

        if gem3_retry_batch is None:
            return {"borelog": []}
        gem3_retry_results = extract_jsons_from_batch(gem3_retry_batch)

        if flash_batch is None:
            return {"borelog": []}
        flash_results = extract_jsons_from_batch(flash_batch)

        # --- Apply WAVE 2 results ---
        for r in invalid_regions:
            img_idx = r["img_idx"]
            label = r.get("label")
            kind = r.get("kind", "region")
            g3_retry_index = r.get("g3_retry_index", -1)
            flash_index = r.get("flash_index", -1)

            result_obj: Optional[Dict[str, Any]] = None
            used_fallback = False

            # Try Gemini-3 retry first
            if 0 <= g3_retry_index < len(gem3_retry_results):
                result_obj = gem3_retry_results[g3_retry_index]

            # If retry failed, try Flash
            if result_obj is None and 0 <= flash_index < len(flash_results):
                result_obj = flash_results[flash_index]
                if result_obj is not None:
                    used_fallback = True

            if result_obj is None:
                print(f"[WARN] Wave-2 retry+flash still unusable for img {img_idx+1}, region {label}")
                continue

            payload = result_obj.get("borehole_log", result_obj)

            # MULTI-IMAGE / full-borelog
            if kind == "multi" or label == "full_borelog":
                results[img_idx]["header_info"] = payload.get("header_info")
                results[img_idx]["soil_layers"] = payload.get("soil_layers")
                results[img_idx]["spt_tests"] = payload.get("spt_tests")
                if used_fallback:
                    results[img_idx]["header_fall_back"] = True
                    results[img_idx]["soil_fall_back"] = True
                    results[img_idx]["spt_fall_back"] = True

            # REGION-style
            else:
                if label == "header_region":
                    results[img_idx]["header_info"] = payload.get("header_info")
                    if used_fallback:
                        results[img_idx]["header_fall_back"] = True
                elif label == "soil_region":
                    results[img_idx]["soil_layers"] = payload.get("soil_layers")
                    if used_fallback:
                        results[img_idx]["soil_fall_back"] = True
                elif label == "spt_region":
                    results[img_idx]["spt_tests"] = payload.get("spt_tests")
                    if used_fallback:
                        results[img_idx]["spt_fall_back"] = True

        print("All batch jobs processed (wave 1 + wave 2).")
        return {"borelog": results}

    else:
        # Unknown wave value
        print(f"[ERROR] Unknown wave value in batch_plan: {wave}")
        return {"borelog": []}

# ---------------------------------------------------------------------
# Optional: debug helper for inspecting raw batch structure
# ---------------------------------------------------------------------
def debug_print_batch_structure(batch: Any) -> None:
    """
    One-time helper to inspect the batch object so we can see where
    responses and text actually live.
    """
    print("=== BATCH RAW OBJECT ===")
    print(batch)
    dest = getattr(batch, "dest", None) or getattr(batch, "output", None)
    print("\n=== dest ===")
    print("type(dest):", type(dest))
    print("dir(dest):", dir(dest))

    # Try common field names for inline responses
    inlined = (
        getattr(dest, "inlined_responses", None)
        or getattr(dest, "inline_responses", None)
        or getattr(dest, "responses", None)
    )

    if not inlined:
        print("\n[DEBUG] No inlined responses found on dest")
        return

    print(f"\nlen(inlined_responses): {len(inlined)}")
    first = inlined[0]
    print("\n=== first inline response ===")
    print("type(first):", type(first))
    print("dir(first):", dir(first))

    resp = getattr(first, "response", None) or first
    print("\n=== first.response ===")
    print("type(response):", type(resp))
    print("dir(response):", dir(resp))

    # Try to print some text
    txt = _extract_text_from_gemini3_response(resp)
    print("\n=== first.response TEXT (first 500 chars) ===")
    print(txt[:500])
