import os
import json
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch

class PostEvalConfigToken:
    TOKENIZER_ID = "your_path/llava-1.5-7b-hf"
    SIMILARITY_MODEL_ID = 'your_path/all-MiniLM-L6-v2' 
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    SIMILARITY_THRESHOLD = 0.6 

    INPUT_RESULTS_JSON = "your_path/DECODE_llava_top40.json" 
    OUTPUT_RESULTS_JSON = "./em/DECODE_llava_top40_post.json" 

def calculate_token_accuracy(response_text, target_text, tokenizer):
    if not response_text or not target_text: return 0.0
    try:
        response_text = response_text.strip()
        target_text = target_text.strip()
        
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
        target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_tokens: return 0.0
        
        len_to_compare = len(target_tokens)
        response_prefix_tokens = response_tokens[:len_to_compare]
        
        matches = sum(1 for i in range(len(response_prefix_tokens)) if i < len(target_tokens) and response_prefix_tokens[i] == target_tokens[i])
        return matches / len(target_tokens)
    except Exception: return 0.0

def check_semantic_similarity(response, ground_truth, model):
    if not response or not ground_truth: return 0.0
    try:
        e1 = model.encode(response, convert_to_tensor=True)
        e2 = model.encode(ground_truth, convert_to_tensor=True)
        return util.pytorch_cos_sim(e1, e2).item()
    except Exception: return 0.0

def my_avg(a):
    if not a: return 0.0
    return round(sum(a) * 100 / len(a), 2)

def process_evaluation_block(details_dict, similarity_model, tokenizer, config):
    re_eval_metrics = {}
    raw_scores_log = {}
    
    all_test_types = details_dict.keys()
    
    for t_type in all_test_types:
        type_success_flags = []
        raw_scores_log[t_type] = []
        
        items = details_dict.get(t_type, [])
        if not items:
            re_eval_metrics[f"{t_type}_success_rate"] = 0.0
            continue

        for item in items:
            is_success = 0
            score = 0.0
            
            raw_post_resp = item.get("response_post", "")
            
            if "locality" in t_type.lower():
                raw_pre_resp = item.get("response_pre", "")
                
                sim = check_semantic_similarity(raw_post_resp, raw_pre_resp, similarity_model)
                score = sim
                if sim >= config.SIMILARITY_THRESHOLD:
                    is_success = 1

            else: 
                target_text = item.get("target", "")
                
                accuracy = calculate_token_accuracy(raw_post_resp, target_text, tokenizer)
                score = accuracy
                if accuracy == 1.0:
                    is_success = 1
            
            type_success_flags.append(is_success)
            raw_scores_log[t_type].append(round(score, 4))
        
        re_eval_metrics[f"{t_type}_success_rate"] = my_avg(type_success_flags)
        
    return re_eval_metrics, raw_scores_log

def main():
    config = PostEvalConfigToken()
    
    print(f" Loading tokenizer: {config.TOKENIZER_ID}")
    try: 
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID, use_fast=False)
    except Exception as e: 
        print(f" Error: Could not load tokenizer: {e}")
        return
    
    print(f" Loading semantic similarity model: {config.SIMILARITY_MODEL_ID}")
    try: 
        similarity_model = SentenceTransformer(config.SIMILARITY_MODEL_ID, device=config.DEVICE)
    except Exception as e: 
        print(f" Error: Could not load similarity model: {e}")
        return

    print(f" Reading original results file: {config.INPUT_RESULTS_JSON}")
    if not os.path.exists(config.INPUT_RESULTS_JSON):
        print(f" File does not exist, please check the path: {config.INPUT_RESULTS_JSON}")
        return

    with open(config.INPUT_RESULTS_JSON, 'r', encoding='utf-8') as f:
        content = json.load(f)
        original_results = content if isinstance(content, list) else [content]
    
    print(f" Starting evaluation of {len(original_results)} cases...")
    re_evaluated_results = []

    for case in tqdm(original_results, desc="Re-evaluating"):
        new_case_entry = json.loads(json.dumps(case)) 
        
        targets_to_evaluate = [
            ("joint", "details"),
            ("sequential_text_first", "stage1_text_details"),
            ("sequential_text_first", "stage2_image_details_final"),
            ("sequential_image_first", "stage1_image_details"),
            ("sequential_image_first", "stage2_text_details_final")
        ]
        
        for parent_key, details_key in targets_to_evaluate:
            if parent_key in new_case_entry:
                parent_block = new_case_entry[parent_key]
                
                if details_key in parent_block:
                    details_data = parent_block[details_key]
                    
                    metrics, scores = process_evaluation_block(
                        details_data, 
                        similarity_model, 
                        tokenizer, 
                        config
                    )
                    
                    parent_block[f"re_evaluated_{details_key}_metrics"] = metrics
                    parent_block[f"raw_token_scores_{details_key}"] = scores

        re_evaluated_results.append(new_case_entry)

    output_dir = os.path.dirname(config.OUTPUT_RESULTS_JSON)
    if output_dir and not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        
    with open(config.OUTPUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(re_evaluated_results, f, indent=2, ensure_ascii=False)
        
    print("-" * 50)
    print(f" Evaluation complete! Results saved to: {config.OUTPUT_RESULTS_JSON}")
    print("-" * 50)

if __name__ == "__main__":
    main()