import json
import pickle
from app.core.batch_converter import collect_boring_batches_results

# Input from wave 1 (or wave 2) – whatever you saved previously
INFO_PATH = "boring_batch_info.pkl"

# Where to save final merged JSON
OUTPUT_PATH = "boring_batch.json"



if __name__ == "__main__":
    # 1) Load saved batch plan (pickle from create_boring_batches_for_images or wave-2 plan)
    with open(INFO_PATH, "rb") as f:
        batch_plan = pickle.load(f)

    # 2) Collect results (handles wave 1 or wave 2 depending on batch_plan["wave"])
    result = collect_boring_batches_results(batch_plan, max_output_tokens_g3_retry=15000, max_output_tokens_flash=10000)
    if result is None:
        pass
    else:
        # Case A: final result (either wave 1 fully succeeded, or wave 2 completed)
        if (
            isinstance(result, dict)
            and "borelog" in result
            and isinstance(result["borelog"], list)
            and len(result["borelog"]) > 0
        ):
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Collected batch results saved to: {OUTPUT_PATH}")

        # Case B: wave-2 plan returned (some regions invalid after wave 1, retry batches created)
        elif isinstance(result, dict) and result.get("wave") == 2:
            # Save the wave-2 plan so you can run this script again later for wave 2
            with open(INFO_PATH, "wb") as f:
                pickle.dump(result, f)
            print(
                "⚠️ Wave 1 completed but some regions were invalid.\n"
                f"   Wave-2 batch plan saved to: {INFO_PATH}\n"
                "   Run collect_boring_batches_results again with this file after wave-2 "
                "batches have been created (or immediately; the helper will wait for them)."
            )

        # Case C: unexpected structure
        else:
            print("❌ Result is neither a final JSON nor a valid wave-2 plan; no output written.")
