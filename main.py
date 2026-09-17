from app.core.batch_converter import (
    create_boring_batches_for_images,
)

from app.core.classifier import get_boring_pdf
import pickle


if __name__ == "__main__":
    # 1) Load pdf
    pdf_path = "root/boring (29).pdf"
    groups = get_boring_pdf(pdf_path)

    # 2) Create batch plan (first wave: Gemini-3 only, or whatever your function does now)
    batch_plan = create_boring_batches_for_images(
        image_groups=groups,
        max_output_tokens_g3=10000,
        multi_instruction_path="boring_instructions.txt"
    )

    # 3) Save batch_plan as pickle (so it can contain bytes safely)
    with open("boring_batch_info.pkl", "wb") as f:
        pickle.dump(batch_plan, f)

    print("First wave complete. Saved batch plan to boring_batch_info.pkl")
