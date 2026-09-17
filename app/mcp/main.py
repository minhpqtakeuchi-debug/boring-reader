import json
from core.classifier import get_boring_pdf
from core.converter import process_boring_groups_two_wave, validate_borehole_log_dict

if __name__ == "__main__":
    pdf_path = "D:\\Project\\boring_reader\\root\\20251112150947.pdf"
    groups = get_boring_pdf(pdf_path)

    out = process_boring_groups_two_wave(groups, 
                                         validator=validate_borehole_log_dict, 
                                         gemini3_rate_limit_per_minute=24, 
                                         flash_rate_limit_per_minute=999, 
                                         max_output_tokens_g3=10000,
                                         max_output_tokens_g3_retry=15000,
                                         max_output_tokens_flash=10000)
    
    with open("boring_jobs/boring_json.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
