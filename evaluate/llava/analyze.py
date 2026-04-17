# The analysis script will perform a post-evaluation on the output results, encompassing an analysis of the original scores and an assessment of the EM scores, where:
# The evaluation criteria for the original score are: accuracy and generalization, whether the editing goal is present in the model's output. For localization, whether the similarity between the model's output before and after editing is greater than 0.6 (this localization is for reference only);
# The evaluation criteria for the post-evaluation are: accuracy and generalization, whether the model output is consistent with the editing goal. Locality, the average similarity of the output before and after model editing.

import json
import os
from collections import defaultdict

class StatisticsConfig:
    INPUT_JSON_PATH = "./em/DECODE_llava_top40_post.json" 
    OUTPUT_JSON_PATH = "./em/DECODE_llava_top40_post_report.json"

    ANALYSIS_PATHS = {
        "joint": (
            ["joint"], 
            "re_evaluated_details_metrics", 
            "raw_token_scores_details"
        ),
        "sequential_text_first_stage1": (
            ["sequential_text_first"], 
            "re_evaluated_stage1_text_details_metrics", 
            "raw_token_scores_stage1_text_details"
        ),
        "sequential_text_first_stage2_final": (
            ["sequential_text_first"], 
            "re_evaluated_stage2_image_details_final_metrics", 
            "raw_token_scores_stage2_image_details_final"
        ),
        "sequential_image_first_stage1": (
            ["sequential_image_first"], 
            "re_evaluated_stage1_image_details_metrics", 
            "raw_token_scores_stage1_image_details"
        ),
        "sequential_image_first_stage2_final": (
            ["sequential_image_first"], 
            "re_evaluated_stage2_text_details_final_metrics", 
            "raw_token_scores_stage2_text_details_final"
        ),
    }
    
    METRIC_NAMES = [
        "T_Reliability", "M_Reliability",
        "T_Generality", "M_Generality",
        "T_locality", "M_locality"
    ]

def get_nested_dict(data, path):
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def analyze_block(blocks_to_analyze, metric_names, metrics_key, raw_score_key):
    original_success_rates = defaultdict(float)
    valid_original_cases = 0

    reeval_raw_score_sums = defaultdict(float)
    valid_reeval_cases = 0

    for block in blocks_to_analyze:
        if metrics_key in block:
            valid_original_cases += 1
            metrics = block[metrics_key]
            for name in metric_names:
                rate_key = f"{name}_success_rate"
                if rate_key in metrics:
                    original_success_rates[name] += metrics.get(rate_key, 0.0)
        
        if raw_score_key in block:
            valid_reeval_cases += 1
            raw_scores_dict = block[raw_score_key]
            
            for name in metric_names:
                score_list = raw_scores_dict.get(name, [0.0])
                raw_score = score_list[0] if score_list else 0.0
                reeval_raw_score_sums[name] += raw_score

    final_analysis = {
        "success_rate_metrics": { 
            "valid_cases": valid_original_cases, 
            "average_success_rates_percent": {} 
        },
        "raw_token_scores_analysis": { 
            "valid_cases": valid_reeval_cases, 
            "average_raw_scores_percent": {},
        }
    }

    if valid_original_cases > 0:
        for name, total_rate in original_success_rates.items():
            final_analysis["success_rate_metrics"]["average_success_rates_percent"][name] = round(total_rate / valid_original_cases, 2)
            
    if valid_reeval_cases > 0:
        for name in metric_names:
            if name in reeval_raw_score_sums:
                avg_raw_score = reeval_raw_score_sums[name] / valid_reeval_cases
                final_analysis["raw_token_scores_analysis"]["average_raw_scores_percent"][name] = round(avg_raw_score * 100, 2)

    return final_analysis

def print_report(title, analysis_data):
    print("\n" + "#"*70)
    print(f"### STATISTICAL REPORT FOR: '{title}' ###")
    print("#"*70)

    metrics_results = analysis_data.get("success_rate_metrics", {})
    if metrics_results and metrics_results.get("valid_cases", 0) > 0:
        print("\n" + "="*60)
        print("Analysis of Success Rates (Strict Metric)")
        print("="*60)
        print(f"Valid cases: {metrics_results['valid_cases']}")
        print("\nAverage Success Rates (%):")
        for name, rate in sorted(metrics_results.get("average_success_rates_percent", {}).items()):
            print(f"{name:<16}: {rate:.2f}%")
    
    raw_results = analysis_data.get("raw_token_scores_analysis", {})
    if raw_results and raw_results.get("valid_cases", 0) > 0:
        print("\n" + "="*60)
        print("Analysis of Raw Token Scores (No-Clean Accuracy)")
        print("="*60)
        print(f"Valid cases: {raw_results['valid_cases']}")
        print("\nAverage Raw Scores (%):")
        print("-"*60)
        for name, rate in sorted(raw_results.get("average_raw_scores_percent", {}).items()):
            print(f"{name:<16}: {rate:.2f}%")
            
    print("\n" + "#"*70 + "\n")

def main():
    config = StatisticsConfig()

    if not os.path.exists(config.INPUT_JSON_PATH):
        print(f"Error: Input file '{config.INPUT_JSON_PATH}' not found.")
        return

    try:
        with open(config.INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                all_cases = content
            else:
                all_cases = [content]
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    final_report = { "overall_summary": { "total_cases_in_file": len(all_cases) } }
    print(f"Found {len(all_cases)} total cases to process.")
    
    for report_name, (path, metrics_key, raw_score_key) in config.ANALYSIS_PATHS.items():
        blocks_to_analyze = []
        for case in all_cases:
            block = get_nested_dict(case, path)
            if block:
                blocks_to_analyze.append(block)
        
        if not blocks_to_analyze:
            print(f"\nWarning: No data found for path '{' -> '.join(path)}'. Skipping '{report_name}'.")
            continue
            
        analysis_data = analyze_block(
            blocks_to_analyze, 
            config.METRIC_NAMES, 
            metrics_key,
            raw_score_key
        )
        final_report[report_name] = analysis_data
        print_report(report_name, analysis_data)
        
    output_dir = os.path.dirname(config.OUTPUT_JSON_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(config.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\nFinal analysis report successfully saved to '{config.OUTPUT_JSON_PATH}'")

if __name__ == "__main__":
    main()