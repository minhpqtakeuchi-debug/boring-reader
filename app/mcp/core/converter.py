# pip install -U google-genai ultralytics opencv-python numpy pydantic

import json
import re
import time
import threading
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import cv2
import numpy as np
from google import genai
from google.genai import types
from ultralytics import YOLO
from math import atan2, degrees
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pydantic import ValidationError

# ----------------------------
# YOUR APP SETTINGS
# ----------------------------
from core.config import settings  # adjust if needed
from core.dto import BoreholeLogEnvelope

API_KEY = settings.api_key

# ----------------------------
# CONFIG
# ----------------------------
TEMPERATURE = 1.0

# Models
GEM3_MODEL = "gemini-3-pro-preview"
FLASH_MODEL = "gemini-2.5-flash"

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

# IMPORTANT: do NOT put a trailing comma here (or it becomes a tuple)
MULTI_INSTRUCTION_PATH = "yolo/instructions/boring_instructions.txt"

# Load YOLO model (adjust to your weights)
model = YOLO("yolo/models/boring_regions.pt")


# ----------------------------
# OPTIONAL: VALIDATOR HOOK
# ----------------------------
# If you already have validate_borehole_log_dict(obj)->bool, pass it into the pipeline.
# If not, you can keep validator=None (basic sanity checks only).
ValidatorFn = Optional[Callable[[Dict[str, Any]], bool]]


# ----------------------------
# UTIL: load instruction text (cached)
# ----------------------------
@lru_cache(maxsize=64)
def load_instruction_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ----------------------------
# UTIL: strip code fences and parse JSON
# ----------------------------
def extract_json(text: str) -> Dict[str, Any]:
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
def crop_image_array(img_bgr: np.ndarray, box_xyxy: List[int], margin: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    h, w = img_bgr.shape[:2]

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return img_bgr[y1:y2, x1:x2]


def image_array_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return buf.tobytes()


def crop_img_to_png_bytes(img: np.ndarray, box_xyxy: List[int]) -> bytes:
    img_bit = crop_image_array(img, box_xyxy, margin=0)
    return image_array_to_png_bytes(img_bit)


# ----------------------------
# YOLO REGION DETECTION (single page -> 3 crops)
# ----------------------------
def detect_regions_for_image(img_bgr: np.ndarray) -> List[Tuple[str, bytes, str]]:
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
            best_conf_per_class[cls_id] = float(conf)

    missing = desired_classes - set(best_idx_per_class.keys())
    if missing:
        raise RuntimeError(f"YOLO did not detect all required regions. Missing classes: {sorted(missing)}")

    jobs: List[Tuple[str, bytes, str]] = []
    for cls_id in sorted(desired_classes):
        idx = best_idx_per_class[cls_id]
        box = bboxes_xyxy[idx].astype(int).tolist()
        label = label_map[cls_id]
        instruction_path = instructions_map[cls_id]
        image_bytes = crop_img_to_png_bytes(img, box)
        jobs.append((label, image_bytes, instruction_path))

    return jobs


# ----------------------------
# DESKEW HELPERS (same as your batch version)
# ----------------------------
def estimate_skew_from_vertical_lines(
    bw: np.ndarray,
    max_vertical_deviation: float = 10.0,
    min_frac_of_height: float = 0.4,
) -> float:
    if len(bw.shape) == 3:
        bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)

    h, _w = bw.shape[:2]
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
            vertical_deviation = (angle - 90.0) if (dev_pos < dev_neg) else (angle + 90.0)
            vertical_deviations.append(vertical_deviation)

    if not vertical_deviations:
        return 0.0

    return float(np.median(vertical_deviations))


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_image_by_vertical(
    img: np.ndarray,
    max_vertical_deviation: float = 10.0,
    min_frac_of_height: float = 0.4,
) -> Tuple[np.ndarray, float]:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skew_deg = estimate_skew_from_vertical_lines(
        bw,
        max_vertical_deviation=max_vertical_deviation,
        min_frac_of_height=min_frac_of_height,
    )

    # Use same behavior as your batch code: rotate by skew*0.5
    rotated = rotate_image(img, skew_deg * 0.5)
    return rotated, skew_deg


# ----------------------------
# SLIDING WINDOW RATE LIMITER
# ----------------------------
class SlidingWindowRateLimiter:
    def __init__(self, limit_per_minute: int, window_sec: float = 60.0):
        self.limit = limit_per_minute
        self.window = window_sec
        self._times = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                while self._times and (now - self._times[0]) > self.window:
                    self._times.popleft()

                if len(self._times) < self.limit:
                    self._times.append(now)
                    return

                oldest = self._times[0]
                sleep_for = self.window - (now - oldest)

            if sleep_for > 0:
                time.sleep(sleep_for)


# ----------------------------
# THREAD-LOCAL GENAI CLIENT (NO TIMEOUT USED)
# ----------------------------
_thread_local = threading.local()

def _get_client() -> genai.Client:
    c = getattr(_thread_local, "client", None)
    if c is None:
        c = genai.Client(api_key=API_KEY)  # <-- no HttpOptions(timeout=...)
        _thread_local.client = c
    return c

# ----------------------------
# SPINNER (console)
# ----------------------------
def _start_spinner(prefix: str = "Working") -> Tuple[threading.Event, threading.Thread]:
    stop_evt = threading.Event()

    def _spin():
        symbols = ["|", "/", "-", "\\"]
        i = 0
        while not stop_evt.is_set():
            sym = symbols[i % len(symbols)]
            print(f"\r{prefix}... {sym}", end="", flush=True)
            time.sleep(0.1)
            i += 1
        # clear line
        print(f"\r{prefix}... done.   ")

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    return stop_evt, t

# ----------------------------
# JOB MODEL
# ----------------------------
@dataclass
class LLMJob:
    kind: str  # "region" | "multi"
    img_idx: int
    label: str  # header_region | soil_region | spt_region | full_borelog
    instruction_path: str
    image_bytes_list: List[bytes]  # 1 for region, N for multi
    max_output_tokens: int
    retry_max_output_tokens: int


# ----------------------------
# MODEL CALL (ONE REQUEST)
# ----------------------------
def _call_model_once(
    model_name: str,
    image_bytes_list: Sequence[bytes],
    instruction_path: str,
    *,
    max_output_tokens: int,
    thinking_low: bool,
    image_mime_type: str = "image/png",
) -> Optional[Dict[str, Any]]:
    if not image_bytes_list:
        return None

    parts = [
        types.Part.from_bytes(data=bytes(b), mime_type=image_mime_type)
        for b in image_bytes_list
    ]

    cfg_kwargs: Dict[str, Any] = dict(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        system_instruction=load_instruction_text(instruction_path),
        max_output_tokens=max_output_tokens,  # <-- use max tokens like batch
    )

    # ThinkingConfig is Gemini-3 specific; safe-guard
    if thinking_low and model_name.startswith("gemini-3"):
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    config = types.GenerateContentConfig(**cfg_kwargs)

    client = _get_client()
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": parts}],
            config=config,
        )
    except Exception:
        return None

    raw_text = _extract_text_from_gemini3_response(resp)
    if not raw_text:
        return None

    try:
        parsed = extract_json(raw_text)
    except Exception:
        return None

    usage_meta = getattr(resp, "usage_metadata", None)
    parsed["_usage"] = {
        "input_tokens": getattr(usage_meta, "prompt_token_count", None) if usage_meta else None,
        "output_tokens": getattr(usage_meta, "candidates_token_count", None) if usage_meta else None,
        "total_tokens": getattr(usage_meta, "total_token_count", None) if usage_meta else None,
    }
    return parsed


def _is_valid_result(obj: Optional[Dict[str, Any]], validator: ValidatorFn) -> bool:
    if obj is None or not isinstance(obj, dict):
        return False

    # If you provide a validator, use it (recommended)
    if validator is not None:
        try:
            return bool(validator(obj))
        except Exception:
            return False

    # Minimal sanity fallback if no validator given
    if "borehole_log" in obj:
        return True
    # region-style partials might be returned as top-level
    if any(k in obj for k in ("header_info", "soil_layers", "spt_tests")):
        return True
    return False


# ----------------------------
# WAVE WORKERS
# ----------------------------
def _wave1_worker(
    job: LLMJob,
    gem3_limiter: SlidingWindowRateLimiter,
    validator: ValidatorFn,
) -> Tuple[LLMJob, Optional[Dict[str, Any]]]:
    gem3_limiter.wait_for_slot()
    res = _call_model_once(
        model_name=GEM3_MODEL,
        image_bytes_list=job.image_bytes_list,
        instruction_path=job.instruction_path,
        max_output_tokens=job.max_output_tokens,
        thinking_low=True,
    )
    if not _is_valid_result(res, validator):
        return job, None
    return job, res


def _wave2_worker_dual(
    job: LLMJob,
    gem3_limiter: SlidingWindowRateLimiter,
    flash_limiter: Optional[SlidingWindowRateLimiter],
    validator: ValidatorFn,
) -> Tuple[LLMJob, Optional[Dict[str, Any]], bool]:
    """
    For ONE invalid job:
      - run Gemini-3 retry and Flash in parallel (no multi-race)
      - prefer Gemini-3 if valid; else Flash if valid
    Returns: (job, result_or_none, used_fallback)
    """
    def _g3_retry():
        gem3_limiter.wait_for_slot()
        return _call_model_once(
            model_name=GEM3_MODEL,
            image_bytes_list=job.image_bytes_list,
            instruction_path=job.instruction_path,
            max_output_tokens=job.retry_max_output_tokens,
            thinking_low=True,
        )

    def _flash():
        if flash_limiter is not None:
            flash_limiter.wait_for_slot()
        return _call_model_once(
            model_name=FLASH_MODEL,
            image_bytes_list=job.image_bytes_list,
            instruction_path=job.instruction_path,
            max_output_tokens=max(1024, job.retry_max_output_tokens // 2),
            thinking_low=False,
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_g3 = ex.submit(_g3_retry)
        f_fl = ex.submit(_flash)
        g3_res = f_g3.result()
        fl_res = f_fl.result()

    if _is_valid_result(g3_res, validator):
        return job, g3_res, False
    if _is_valid_result(fl_res, validator):
        return job, fl_res, True
    return job, None, False


# ----------------------------
# MAIN: 2-WAVE PIPELINE (GROUPED IMAGES LIKE BATCH)
# ----------------------------
def process_boring_groups_two_wave(
    image_groups: Sequence[Sequence[np.ndarray]],
    *,
    gemini3_rate_limit_per_minute: int = 50,
    flash_rate_limit_per_minute: int = 120,
    image_rate_limit_per_minute: int = 5,  # YOLO start rate limit
    max_workers_wave1: int = 20,
    max_workers_wave2: int = 20,
    max_output_tokens_g3: int = 4096,
    max_output_tokens_g3_retry: int = 4096,
    max_output_tokens_flash: int = 2048,
    deskew: bool = True,
    validator: ValidatorFn = None,
    verbose: bool = True,
    use_spinner: bool = True,
) -> Dict[str, Any]:
    gem3_limiter = SlidingWindowRateLimiter(gemini3_rate_limit_per_minute)
    flash_limiter = SlidingWindowRateLimiter(flash_rate_limit_per_minute) if flash_rate_limit_per_minute > 0 else None
    image_limiter = SlidingWindowRateLimiter(image_rate_limit_per_minute)

    num_groups = len(image_groups)

    results: List[Dict[str, Any]] = [
        {
            "header_info": None,
            "soil_layers": None,
            "spt_tests": None,
            "header_fall_back": False,
            "soil_fall_back": False,
            "spt_fall_back": False,
            "_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        for _ in range(num_groups)
    ]

    # -------- build jobs --------
    jobs: List[LLMJob] = []

    if verbose:
        print(f"\n[PIPELINE] Building jobs for {num_groups} borelog group(s)...")
        print("[PIPELINE] Rule: len(group)==1 -> YOLO regions; len(group)>1 -> full multi-image")

    for img_idx, group in enumerate(image_groups):
        if not group:
            if verbose:
                print(f"[WARN] Group {img_idx+1} is empty; skipping.")
            continue

        # deskew pages
        pages: List[np.ndarray] = []
        for p_i, img in enumerate(group):
            if deskew:
                img2, sk = deskew_image_by_vertical(img, max_vertical_deviation=10.0, min_frac_of_height=0.4)
                pages.append(img2)
                if verbose:
                    print(f"[DESKEW] Group {img_idx+1} page {p_i+1}: skew={sk:.3f} deg")
            else:
                pages.append(img)

        if len(pages) == 1:
            image_limiter.wait_for_slot()
            if verbose:
                print(f"\n[YOLO] Group {img_idx+1}/{num_groups}: running region detection...")
            try:
                region_jobs = detect_regions_for_image(pages[0])
            except Exception as e:
                if verbose:
                    print(f"[ERROR] YOLO failed for Group {img_idx+1}: {e}")
                region_jobs = []

            for (label, b, instr) in region_jobs:
                jobs.append(
                    LLMJob(
                        kind="region",
                        img_idx=img_idx,
                        label=label,
                        instruction_path=instr,
                        image_bytes_list=[b],
                        max_output_tokens=max_output_tokens_g3,
                        retry_max_output_tokens=max_output_tokens_g3_retry,
                    )
                )
                if verbose:
                    print(f"[JOB] + region job: group={img_idx+1} label={label} tokens={max_output_tokens_g3}")

        else:
            multi_bytes = [image_array_to_png_bytes(p) for p in pages]
            jobs.append(
                LLMJob(
                    kind="multi",
                    img_idx=img_idx,
                    label="full_borelog",
                    instruction_path=MULTI_INSTRUCTION_PATH,
                    image_bytes_list=multi_bytes,
                    max_output_tokens=max_output_tokens_g3 * 3,
                    retry_max_output_tokens=max_output_tokens_g3_retry * 3,
                )
            )
            if verbose:
                print(
                    f"[JOB] + multi job: group={img_idx+1} pages={len(pages)} "
                    f"tokens={max_output_tokens_g3*3}"
                )

    if not jobs:
        if verbose:
            print("[PIPELINE] No jobs built. Returning empty shells.")
        return {"borelog": results}

    if verbose:
        print(f"\n[PIPELINE] Total LLM jobs: {len(jobs)}")

    # -------- WAVE 1 --------
    if verbose:
        print("\n====================")
        print("[WAVE 1] Gemini-3 initial pass")
        print("====================")

    wave1_invalid: List[LLMJob] = []

    stop_evt = None
    spin_thread = None
    if use_spinner:
        stop_evt, spin_thread = _start_spinner(prefix="[WAVE 1] Processing")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(max_workers_wave1, len(jobs))) as ex:
        futs: List[Future] = [ex.submit(_wave1_worker, job, gem3_limiter, validator) for job in jobs]

        completed = 0
        total = len(futs)

        for f in as_completed(futs):
            job, res = f.result()
            completed += 1

            if res is None:
                wave1_invalid.append(job)
                if verbose:
                    print(f"\n[WAVE 1][INVALID] group={job.img_idx+1} kind={job.kind} label={job.label}")
                continue

            usage = res.get("_usage") or {}
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                results[job.img_idx]["_usage"][k] += usage.get(k) or 0

            payload = res.get("borehole_log", res)

            if job.kind == "multi" or job.label == "full_borelog":
                results[job.img_idx]["header_info"] = payload.get("header_info")
                results[job.img_idx]["soil_layers"] = payload.get("soil_layers")
                results[job.img_idx]["spt_tests"] = payload.get("spt_tests")
            else:
                if job.label == "header_region":
                    results[job.img_idx]["header_info"] = payload.get("header_info")
                elif job.label == "soil_region":
                    results[job.img_idx]["soil_layers"] = payload.get("soil_layers")
                elif job.label == "spt_region":
                    results[job.img_idx]["spt_tests"] = payload.get("spt_tests")

            if verbose:
                print(
                    f"\n[WAVE 1][OK] ({completed}/{total}) "
                    f"group={job.img_idx+1} kind={job.kind} label={job.label}"
                )

    if use_spinner and stop_evt is not None and spin_thread is not None:
        stop_evt.set()
        spin_thread.join()

    if verbose:
        dt = time.time() - t0
        print(f"[WAVE 1] Done in {dt:.1f}s. Invalid jobs: {len(wave1_invalid)}")

    if not wave1_invalid:
        if verbose:
            print("[PIPELINE] Wave 1 succeeded for all jobs. Returning final JSON.")
        return {"borelog": results}

    # -------- WAVE 2 --------
    if verbose:
        print("\n====================")
        print("[WAVE 2] Retry Gemini-3 + Flash fallback (parallel per job)")
        print("====================")

    stop_evt2 = None
    spin_thread2 = None
    if use_spinner:
        stop_evt2, spin_thread2 = _start_spinner(prefix="[WAVE 2] Retrying")

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=min(max_workers_wave2, len(wave1_invalid))) as ex:
        futs2: List[Future] = [
            ex.submit(_wave2_worker_dual, job, gem3_limiter, flash_limiter, validator)
            for job in wave1_invalid
        ]

        completed2 = 0
        total2 = len(futs2)

        for f in as_completed(futs2):
            job, res, used_fallback = f.result()
            completed2 += 1

            if res is None:
                if verbose:
                    print(
                        f"\n[WAVE 2][STILL INVALID] ({completed2}/{total2}) "
                        f"group={job.img_idx+1} kind={job.kind} label={job.label}"
                    )
                continue

            usage = res.get("_usage") or {}
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                results[job.img_idx]["_usage"][k] += usage.get(k) or 0

            payload = res.get("borehole_log", res)

            if job.kind == "multi" or job.label == "full_borelog":
                results[job.img_idx]["header_info"] = payload.get("header_info")
                results[job.img_idx]["soil_layers"] = payload.get("soil_layers")
                results[job.img_idx]["spt_tests"] = payload.get("spt_tests")
                if used_fallback:
                    results[job.img_idx]["header_fall_back"] = True
                    results[job.img_idx]["soil_fall_back"] = True
                    results[job.img_idx]["spt_fall_back"] = True
            else:
                if job.label == "header_region":
                    results[job.img_idx]["header_info"] = payload.get("header_info")
                    if used_fallback:
                        results[job.img_idx]["header_fall_back"] = True
                elif job.label == "soil_region":
                    results[job.img_idx]["soil_layers"] = payload.get("soil_layers")
                    if used_fallback:
                        results[job.img_idx]["soil_fall_back"] = True
                elif job.label == "spt_region":
                    results[job.img_idx]["spt_tests"] = payload.get("spt_tests")
                    if used_fallback:
                        results[job.img_idx]["spt_fall_back"] = True

            if verbose:
                print(
                    f"\n[WAVE 2][OK] ({completed2}/{total2}) "
                    f"group={job.img_idx+1} kind={job.kind} label={job.label} "
                    f"fallback={used_fallback}"
                )

    if use_spinner and stop_evt2 is not None and spin_thread2 is not None:
        stop_evt2.set()
        spin_thread2.join()

    if verbose:
        dt2 = time.time() - t1
        print(f"[WAVE 2] Done in {dt2:.1f}s.")
        print("[PIPELINE] Returning merged results.")

    return {"borelog": results}


# ----------------------------
# CONVENIENCE WRAPPER FOR SIMPLE LIST OF IMAGES (SINGLE PAGE EACH)
# ----------------------------
def process_boring_images_two_wave(
    images: Sequence[np.ndarray],
    *,
    validator: ValidatorFn = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    image_groups = [[img] for img in images]
    return process_boring_groups_two_wave(image_groups, validator=validator, **kwargs)


# Validate
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

