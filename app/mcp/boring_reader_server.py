from __future__ import annotations

import io
import sys
import contextlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mcp.server.fastmcp import FastMCP, Context

from core.batch_converter import (
    create_boring_batches_for_images,
    collect_boring_batches_results,
)
from core.classifier import get_boring_pdf
from core.config import settings

# -------------------------------------------------------------------
# MCP server
# -------------------------------------------------------------------
mcp = FastMCP("boring-batch-mcp")

# Where we persist batch plans + results between tool calls
JOBS_DIR = Path(settings.boring_path)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _job_pickle_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.pkl"


def _job_json_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


# -------------------------------------------------------------------
# Tool 1: create first-wave batch plan from a PDF
# -------------------------------------------------------------------
@mcp.tool()
def create_boring_batch(
    context: Context,
    pdf_file: str,
    max_output_tokens_g3: int = 10_000,
) -> Dict[str, Any]:
    """
    Create a *first-wave* Gemini-3 batch plan for a borehole PDF.
    """

    log_buffer = io.StringIO()

    # Capture EVERYTHING printed during PDF grouping + batch planning
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        # 1) Group pages from PDF into borehole logs
        groups = get_boring_pdf(pdf_file)

        # 2) Build first-wave batch plan
        batch_plan = create_boring_batches_for_images(
            image_groups=groups,
            max_output_tokens_g3=max_output_tokens_g3
        )

    # Now we’re OUTSIDE the redirect context, so MCP stdout is clean again
    logs = log_buffer.getvalue()

    # Make sure wave is set
    batch_plan.setdefault("wave", 1)

    job_id = batch_plan.get("gem3_batch_name") or f"job-{uuid4().hex}"
    batch_plan["_job_id"] = job_id

    pkl_path = _job_pickle_path(job_id)
    with open(pkl_path, "wb") as f:
        pickle.dump(batch_plan, f)

    # Optionally truncate logs if they can get huge
    MAX_LOG_CHARS = 8000
    if len(logs) > MAX_LOG_CHARS:
        logs = logs[-MAX_LOG_CHARS:]
        logs = "[truncated]\n" + logs

    return {
        "job_id": job_id,
        "wave": batch_plan["wave"],
        "pickle_path": str(pkl_path),
        "gem3_batch_name": batch_plan.get("gem3_batch_name"),
        "num_images": batch_plan.get("num_images"),
        "logs": logs,  # 👈 agent can read this
    }



# -------------------------------------------------------------------
# Tool 2: collect (wave 1 or wave 2) results for a job_id
# -------------------------------------------------------------------
@mcp.tool()
def collect_boring_batch_results(
    context: Context,
    job_id: str,
    max_output_tokens_g3_retry: int = 15_000,
    max_output_tokens_flash: int = 10_000,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    Collect batch results for a previously created job.

    Behavior:
      * If the job is still running (wave 1 or 2), returns a status explaining
        that more time is needed.
      * If wave 1 finished but some regions failed, this creates wave-2
        retry/fallback batches and saves a new batch_plan (wave=2).
      * If everything is done, saves a merged JSON and returns its path.
    """

    pkl_path = _job_pickle_path(job_id)
    if not pkl_path.exists():
        return {
            "status": "error",
            "message": f"Batch plan pickle not found for job_id={job_id}",
            "logs": f"[collect_boring_batch_results] No pickle at {pkl_path}\n",
        }

    # 1) Load stored batch_plan (could be wave 1 or wave 2)
    with open(pkl_path, "rb") as f:
        batch_plan = pickle.load(f)

    # 2) Call your existing collector (handles wave 1 and wave 2 logic)
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        result = collect_boring_batches_results(
            batch_plan,
            max_output_tokens_g3_retry=max_output_tokens_g3_retry,
            max_output_tokens_flash=max_output_tokens_flash,
        )

    logs = log_buffer.getvalue()

    # Optional: truncate very long logs
    MAX_LOG_CHARS = 8000
    if len(logs) > MAX_LOG_CHARS:
        logs = "[truncated]\n" + logs[-MAX_LOG_CHARS:]

    # If collector uses the "batch not ready" sentinel ({"borelog": []})
    # you can treat that as "pending"
    if result is None or (
        isinstance(result, dict)
        and result.get("borelog") == []
        and result.get("wave") is None
    ):
        return {
            "status": "pending",
            "message": "Batch job(s) not yet in SUCCEEDED state. Call again later.",
            "logs": logs,
        }

    # Case A: final merged JSON (wave 1 or wave 2 fully completed)
    if (
        isinstance(result, dict)
        and "borelog" in result
        and isinstance(result["borelog"], list)
        and len(result["borelog"]) > 0
    ):
        json_path = _job_json_path(job_id)
        if save_json:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return {
            "status": "done",
            "job_id": job_id,
            "json_path": str(json_path),
            "num_borelogs": len(result["borelog"]),
            "logs": logs,
        }

    # Case B: collector returned a wave-2 plan (some regions invalid after wave 1)
    if isinstance(result, dict) and result.get("wave") == 2:
        # Persist updated wave-2 plan (includes retry batch names etc.)
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        return {
            "status": "wave2_pending",
            "job_id": job_id,
            "wave": 2,
            "gem3_retry_batch_name": result.get("gem3_retry_batch_name"),
            "flash_batch_name": result.get("flash_batch_name"),
            "message": (
                "Wave 1 completed but some regions were invalid; "
                "wave-2 retry/fallback batches have been created. "
                "Call this tool again after those batches succeed."
            ),
            "logs": logs,
        }

    # Case C: unexpected structure
    return {
        "status": "error",
        "message": "Unexpected result structure from collect_boring_batches_results",
        "raw_result": result,
        "logs": logs,
    }



if __name__ == "__main__":
    mcp.run(transport="stdio")
