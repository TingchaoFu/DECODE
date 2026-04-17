import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

import torch
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from torch.nn import functional as F
import gc
from sentence_transformers import SentenceTransformer, util
import string
import traceback
from torch.amp import autocast 
import copy

MODEL_ID = "your_path/llava-1.5-7b-hf"
IMAGE_BENCH_ROOT = "your_path/data_image"
DECODE_EDIT_JSONS = [ "your_path/dataset/vqa.json" ]
CANDIDATES_PATH = "your_path/neurons_llava.json"
RESULTS_DIR = "your_path/results"
MAX_NEURONS_PER_TYPE = 40
FINAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "DECODE_llava_top{MAX_NEURONS_PER_TYPE}.json")
EDIT_STRATEGIES = [ 'sequential_text_first']

class DECODEHyperParams:
    def __init__(self):
        self.EPOCHS = 200
        self.LEARNING_RATE = 2e-3
        self.ALPHA_KL = 0.6
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.SIMILARITY_MODEL_ID = 'your_path/all-MiniLM-L6-v2'
        self.SIMILARITY_THRESHOLD = 0.6
        self.LOG_EVERY = 10
        self.REPETITION_PENALTY = 1.2
        self.EARLY_STOP_THRESHOLD_JOINT = 0.5
        self.EARLY_STOP_THRESHOLD_TEXT = 0.3
        self.EARLY_STOP_THRESHOLD_IMAGE = 0.7

SIMILARITY_MODEL = None

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_full_image_path(relative_path):
    if not relative_path: return None
    return os.path.join(IMAGE_BENCH_ROOT, relative_path)

def check_semantic_similarity(response, ground_truth):
    global SIMILARITY_MODEL
    if SIMILARITY_MODEL is None: raise ValueError("Sim model not initialized.")
    if not response or not ground_truth: return 0.0
    try:
        e1 = SIMILARITY_MODEL.encode(response, convert_to_tensor=True)
        e2 = SIMILARITY_MODEL.encode(ground_truth, convert_to_tensor=True)
        return util.pytorch_cos_sim(e1, e2).item()
    except Exception: return 0.0

def is_exact_match(response, ground_truth):
    if not response or not ground_truth: return False
    def normalize(s): return ' '.join(s.lower().translate(str.maketrans('', '', string.punctuation)).split())
    return normalize(ground_truth) in normalize(response)

def my_avg(a):
    if not a: return 0.0
    return round(sum(a) * 100 / len(a), 2)

def get_model_response(model, processor, prompt_text, image=None, max_new_tokens=128, repetition_penalty_val=1.2):
    model_dtype = torch.bfloat16
    try:
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        if image:
            img = image.convert('RGB') if image.mode != 'RGB' else image
        else:
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True).to(model.device)
        inputs = {k: v.to(model_dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad(), autocast(device_type='cuda', dtype=model_dtype):
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=repetition_penalty_val)

        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parts = output.split("ASSISTANT:")
        return parts[-1].strip() if len(parts) > 1 else output
    except Exception:
        traceback.print_exc()
        return ""

def generate_pre_edit_responses(model_pristine, processor, task, hparams):
    pre_edit_details = {}
    print("    > Generating pre-edit baseline responses...")
    for t_type, tests in task['tests'].items():
        if not tests: continue
        pre_edit_details[t_type] = []
        for test in tqdm(tests, desc=f"      Pre-Run ({t_type})", leave=False):
            img = None
            try:
                if test.get('image'):
                    img = Image.open(_get_full_image_path(test.get('image')))
            except FileNotFoundError:
                print(f"[Warning] Image not found: {test.get('image')}")
                pre_edit_details[t_type].append({"question": test['text'], "response": "[Image not found]"})
                continue
            
            response = get_model_response(model_pristine, processor, test['text'], img, repetition_penalty_val=hparams.REPETITION_PENALTY)
            pre_edit_details[t_type].append({"question": test['text'], "response": response})
    return pre_edit_details

def evaluate_model_on_task(model_edited, processor, task, hparams, pre_edit_details, verbose=False):
    scores = {k: [] for k in ["T_Reliability", "M_Reliability", "T_Generality", "M_Generality", "T_locality", "M_locality"]}
    detailed_outputs = {k: [] for k in scores.keys()}
    tests = task.get('tests', {})
    
    for t_type in ["T_Reliability", "M_Reliability", "T_Generality", "M_Generality"]:
        if t_type not in tests: continue
        for idx, test in enumerate(tests.get(t_type, [])):
            try:
                img = Image.open(_get_full_image_path(test['image']))
            except FileNotFoundError:
                scores[t_type].append(0)
                detailed_outputs[t_type].append({"question": test['text'], "target": test['target'], "response": "[Image not found]", "success": False})
                continue
            resp = get_model_response(model_edited, processor, test['text'], img, repetition_penalty_val=hparams.REPETITION_PENALTY)
            succ = 1 if is_exact_match(resp, test['target']) else 0
            scores[t_type].append(succ)
            
            pre_resp = pre_edit_details[t_type][idx]['response'] if t_type in pre_edit_details and idx < len(pre_edit_details[t_type]) else "N/A"
            detailed_outputs[t_type].append({"question": test['text'], "target": test['target'], "response_pre": pre_resp, "response_post": resp, "success": bool(succ)})
            if verbose: print(f"    [Eval-{t_type}] Resp: '{resp[:50]}...' vs GT: '{test['target']}' | {'Success' if succ else 'Fail'}")
    
    for t_type in ["T_locality", "M_locality"]:
        if t_type not in tests: continue
        for i, test in enumerate(tests.get(t_type, [])):
            img = None
            try:
                if test.get('image'):
                    img = Image.open(_get_full_image_path(test.get('image')))
            except FileNotFoundError:
                scores[t_type].append(0)
                detailed_outputs[t_type].append({"question": test['text'], "response_pre": "N/A", "response_post": "[Image not found]", "similarity": 0.0, "success": False})
                continue

            resp_after = get_model_response(model_edited, processor, test['text'], img, repetition_penalty_val=hparams.REPETITION_PENALTY)
            resp_before = pre_edit_details.get(t_type, [])[i]['response'] if pre_edit_details.get(t_type) else ""
            sim = check_semantic_similarity(resp_after, resp_before)
            succ = 1 if sim >= hparams.SIMILARITY_THRESHOLD else 0
            scores[t_type].append(succ)
            detailed_outputs[t_type].append({"question": test['text'], "response_pre": resp_before, "response_post": resp_after, "similarity": sim, "success": bool(succ)})
            if verbose: print(f"    [Eval-{t_type}] Sim: {sim:.2f} | {'Kept' if succ else 'Changed'}")
            
    metrics = {f"{k}_success_rate": my_avg(v) for k, v in scores.items()}
    return metrics, detailed_outputs

class DECODEEditor:
    def __init__(self, processor, hparams):
        self.processor, self.hparams = processor, hparams
        self.added_deltas = []
        self.edited_neurons = []

    def restore_model(self, model):
        if not self.added_deltas or not self.edited_neurons:
            return
        with torch.no_grad():
            for i, (l_idx, n_idx) in enumerate(self.edited_neurons):
                target_weight = model.language_model.layers[l_idx].mlp.down_proj.weight
                delta_to_subtract = self.added_deltas[i].to(target_weight.device, dtype=target_weight.dtype)
                target_weight.data[:, n_idx] -= delta_to_subtract
        
        self.added_deltas = []
        self.edited_neurons = []

    def _create_forward_hook(self, neurons_dict, deltas):
        def fn(layer_idx):
            def hook(module, input, output):
                if layer_idx in neurons_dict:
                    locs, idxs = neurons_dict[layer_idx]['loc'], neurons_dict[layer_idx]['idx']
                    acts = input[0][:, :, locs]
                    deltas_dev = deltas[idxs].to(acts.device, dtype=acts.dtype)
                    output.add_((acts @ deltas_dev).to(output.dtype))
                    return output
                return output
            return hook
        return fn

    def _neurons_to_dict(self, neuron_list):
        neurons_dict = {}
        for i, (l_idx, n_idx) in enumerate(neuron_list):
            if l_idx not in neurons_dict: neurons_dict[l_idx] = {"loc": [], "idx": []}
            neurons_dict[l_idx]["loc"].append(n_idx); neurons_dict[l_idx]["idx"].append(i)
        return neurons_dict

    def edit_stage(self, model_editable, neurons, edit_input_data, logits_pristine, stage_name="", early_stopping_threshold=0.1):
        if not neurons:
            print(f"  [Stage: {stage_name}] No neurons to edit, skipping.")
            return model_editable

        print(f"\n  --- Stage [{stage_name.upper()}]: Editing {len(neurons)} neurons (Threshold: {early_stopping_threshold}) ---")

        neurons_dict = self._neurons_to_dict(neurons)
        unique_layers = set(neurons_dict.keys())

        model_editable.gradient_checkpointing_enable()
        
        deltas = torch.zeros(len(neurons), model_editable.config.text_config.hidden_size, dtype=torch.float32, device=model_editable.device, requires_grad=True)
        optimizer = torch.optim.Adam([deltas], lr=self.hparams.LEARNING_RATE)
        
        inputs = edit_input_data['inputs']
        prompt_len = edit_input_data['prompt_len']
        target_ids = edit_input_data['target_ids']

        original_distribution = None
        if self.hparams.ALPHA_KL > 0 and logits_pristine is not None:
            original_distribution = logits_pristine[0, :prompt_len-1, :].detach().to(model_editable.device, dtype=torch.float32)

        for epoch in range(self.hparams.EPOCHS):
            optimizer.zero_grad()
            hooks = [model_editable.language_model.layers[l_idx].mlp.down_proj.register_forward_hook(self._create_forward_hook(neurons_dict, deltas)(l_idx)) for l_idx in unique_layers]

            try:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model_editable(**inputs)
                    logits = outputs.logits
                    logits_target = logits[0, prompt_len-1:-1]
                    
                    if logits_target.device != target_ids.device:
                        logits_target = logits_target.to(target_ids.device)
                    
                    loss_nll = F.cross_entropy(logits_target, target_ids)

                    loss_kl = torch.tensor(0.0, device=loss_nll.device)
                    if self.hparams.ALPHA_KL > 0 and original_distribution is not None:
                        current_prompt_logits = logits[0, :prompt_len-1, :]
                        kl_log_probs = F.log_softmax(current_prompt_logits.to(torch.float32), dim=-1)
                        kl_probs_pristine = F.softmax(original_distribution, dim=-1)
                        loss_kl = F.kl_div(kl_log_probs, kl_probs_pristine, reduction='batchmean')

                    total_loss = loss_nll + self.hparams.ALPHA_KL * loss_kl

                total_loss.backward()
                optimizer.step()

            finally:
                for h in hooks: h.remove()

            if (epoch + 1) % self.hparams.LOG_EVERY == 0:
                print(f"      [{stage_name}] Epoch {epoch+1}/{self.hparams.EPOCHS} | Total={total_loss.item():.4f} (NLL={loss_nll.item():.4f}, KL={loss_kl.item()*self.hparams.ALPHA_KL:.4f})")

            if loss_nll.item() < early_stopping_threshold:
                print(f"      [{stage_name}] NLL loss ({loss_nll.item():.4f}) < {early_stopping_threshold} at epoch {epoch+1}, early stopping.")
                break

        print(f"  [{stage_name}] Training done, applying weights permanently...")
        with torch.no_grad():
            for i, (l_idx, n_idx) in enumerate(neurons):
                target_weight = model_editable.language_model.layers[l_idx].mlp.down_proj.weight
                delta_to_add = deltas[i].to(target_weight.device, dtype=torch.bfloat16)
                target_weight.data[:, n_idx] += delta_to_add
        
        self.added_deltas.extend([d.detach().cpu() for d in deltas])
        self.edited_neurons.extend(neurons)

        del deltas, optimizer, original_distribution 
        torch.cuda.empty_cache()

        if hasattr(model_editable, "gradient_checkpointing_disable"):
            model_editable.gradient_checkpointing_disable()

        return model_editable.eval()
    
def load_model(model_id, device):
    print(f"    > Loading model (BF16) to {device}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    model.config.vision_feature_layer = -1
    model.config.vision_feature_select_strategy = "default"
    print("    > Model loaded.")
    return model

def main():
    set_seed(42)
    global SIMILARITY_MODEL; hparams = DECODEEditor(); os.makedirs(RESULTS_DIR, exist_ok=True)
    DEVICE = hparams.DEVICE
    
    processor = LlavaProcessor.from_pretrained(MODEL_ID)
    SIMILARITY_MODEL = SentenceTransformer(hparams.SIMILARITY_MODEL_ID, device=hparams.DEVICE)
    candidate_neurons = json.load(open(CANDIDATES_PATH, 'r', encoding='utf-8'))

    editor = DECODEEditor(processor, hparams)

    all_results, completed_case_ids = [], set()
    if os.path.exists(FINAL_RESULTS_PATH):
        try:
            with open(FINAL_RESULTS_PATH, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            completed_case_ids = {res['case_id'] for res in all_results}
            print(f"Loaded {len(all_results)} existing results.")
        except (json.JSONDecodeError, IOError):
            print(f"Failed to load or parse existing results file: {FINAL_RESULTS_PATH}. Starting from scratch.")
            all_results = []
            completed_case_ids = set()

    print("\n--- Loading Base Model (BF16) ---")
    model_pristine = load_model(MODEL_ID, device=DEVICE)
    print("--- Base Model Loaded ---\n")

    for json_path in DECODE_EDIT_JSONS:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n--- Processing Dataset: {os.path.basename(json_path)} ---")
        for i,task_entry in enumerate(tqdm(data, desc="Processing Cases")):
            case_id = f"{i}"
            if case_id in completed_case_ids: continue

            required_image_paths = [
                task_entry.get('image'),
                task_entry.get('image_rephrase'),
                task_entry.get('m_loc')
            ]
            
            image_missing = False
            for relative_path in required_image_paths:
                if relative_path:
                    full_path = _get_full_image_path(relative_path)
                    if not os.path.exists(full_path):
                        print(f"\n[Warning] Image not found for case_id {case_id}: {full_path}. Skipping this case.")
                        image_missing = True
                        break
            
            if image_missing:
                continue

            original_entry = task_entry
            tests = {k: [] for k in ["T_Reliability", "M_Reliability", "T_Generality", "M_Generality", "T_locality", "M_locality"]}
            if 't_rel' in original_entry: tests['T_Reliability'].append({'image': original_entry.get('image'), 'text': original_entry.get('t_rel'), 'target': original_entry.get('alt')})
            if 'i_rel' in original_entry: tests['M_Reliability'].append({'image': original_entry.get('image'), 'text': original_entry.get('i_rel'), 'target': original_entry.get('alt')})
            if 'rephrase' in original_entry: tests['T_Generality'].append({'image': original_entry.get('image'), 'text': original_entry.get('rephrase'), 'target': original_entry.get('alt')})
            if 'image_rephrase' in original_entry: tests['M_Generality'].append({'image': original_entry.get('image_rephrase'), 'text': original_entry.get('i_rel'), 'target': original_entry.get('alt')})
            if 'loc' in original_entry: tests['T_locality'].append({'image': None, 'text': original_entry.get('loc'), 'ground_truth': original_entry.get('loc_ans')})
            if 'm_loc' in original_entry: tests['M_locality'].append({'image': original_entry.get('m_loc'), 'text': original_entry.get('m_loc_q'), 'ground_truth': original_entry.get('m_loc_a')})
            
            formatted_task = {'case_id': case_id, 'original_entry': original_entry, 'edit_prompt': {'image': original_entry.get('image'), 'text': original_entry.get('rephrase'), 'target_new': original_entry.get('alt')}, 'tests': tests}
            task_results = {"case_id": case_id}
            
            print(f"\n--- Task: Editing Case '{case_id}' ---")
            
            try:
                pre_responses = generate_pre_edit_responses(model_pristine, processor, formatted_task, hparams)
                task_results["pre_edit_responses"] = pre_responses
            except torch.OutOfMemoryError:
                print(f"  [OOM] Pre-edit evaluation")
                torch.cuda.empty_cache(); gc.collect()
                continue
            
            case_neurons = candidate_neurons.get(case_id, {})
            text_neurons = case_neurons.get('text_specific', [])[:MAX_NEURONS_PER_TYPE]
            image_neurons = case_neurons.get('image_specific', [])[:MAX_NEURONS_PER_TYPE]
            if len(image_neurons) == 0: continue

            edit_prompt = formatted_task['edit_prompt']
            target_raw = edit_prompt.get('target_new')
            target_for_training = f"{target_raw}{processor.tokenizer.eos_token}" if target_raw else processor.tokenizer.eos_token
            
            prompt_text_only_str = f"USER: {original_entry.get('t_rel')}\nASSISTANT:"
            inputs_text = processor(text=[f"{prompt_text_only_str} {target_for_training}"], images=None, return_tensors="pt", padding=True)
            prompt_len_text = processor(text=[prompt_text_only_str], images=None, return_tensors="pt").input_ids.shape[1]
            text_edit_data = {
                'inputs': {k: v.to(hparams.DEVICE) if not torch.is_floating_point(v) else v.to(hparams.DEVICE, dtype=torch.bfloat16) for k,v in inputs_text.items()}, 
                'prompt_len': prompt_len_text, 
                'target_ids': inputs_text.input_ids[0, prompt_len_text:].to(hparams.DEVICE)
            }
            
            image_obj = Image.open(_get_full_image_path(edit_prompt['image'])).convert("RGB")
            prompt_image_only = f"USER: <image>\n{original_entry.get('i_rel')}\nASSISTANT:"
            inputs_image = processor(text=[f"{prompt_image_only} {target_for_training}"], images=[image_obj], return_tensors="pt", padding=True)
            prompt_len_image = processor(text=[prompt_image_only], images=[image_obj], return_tensors="pt").input_ids.shape[1]
            image_edit_data = {
                'inputs': {k: v.to(hparams.DEVICE) if not torch.is_floating_point(v) else v.to(hparams.DEVICE, dtype=torch.bfloat16) for k,v in inputs_image.items()}, 
                'prompt_len': prompt_len_image, 
                'target_ids': inputs_image.input_ids[0, prompt_len_image:].to(hparams.DEVICE)
            }

            prompt_joint_str = f"USER: <image>\n{original_entry.get('t_rel')}\nASSISTANT:"
            inputs_joint = processor(text=[f"{prompt_joint_str} {target_for_training}"], images=[image_obj], return_tensors="pt", padding=True)
            prompt_len_joint = processor(text=[prompt_joint_str], images=[image_obj], return_tensors="pt").input_ids.shape[1]
            joint_edit_data = {
                'inputs': {k: v.to(hparams.DEVICE) if not torch.is_floating_point(v) else v.to(hparams.DEVICE, dtype=torch.bfloat16) for k,v in inputs_joint.items()}, 
                'prompt_len': prompt_len_joint, 
                'target_ids': inputs_joint.input_ids[0, prompt_len_joint:].to(hparams.DEVICE)
            }
            
            logits_pristine_text = None
            logits_pristine_image = None
            logits_pristine_joint = None
            
            try:
                if hparams.ALPHA_KL > 0:
                    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits_pristine_text = model_pristine(**text_edit_data['inputs']).logits.detach().cpu()
                        logits_pristine_image = model_pristine(**image_edit_data['inputs']).logits.detach().cpu()
                        logits_pristine_joint = model_pristine(**joint_edit_data['inputs']).logits.detach().cpu()
            except torch.OutOfMemoryError:
                print(f"  [OOM] Logits calculation")
                torch.cuda.empty_cache(); gc.collect()
                task_results['error'] = 'OOM during logits'
                continue

            model_to_edit = model_pristine

            for strategy in EDIT_STRATEGIES:
                print(f"\n  --> Testing Strategy: {strategy}")
                
                try:
                    if strategy == 'joint':
                        text_neurons_set = {tuple(n) for n in text_neurons}
                        image_neurons_set = {tuple(n) for n in image_neurons}
                        union_neurons = [list(n) for n in text_neurons_set.union(image_neurons_set)]
                        
                        if any(union_neurons):
                            edited_model = editor.edit_stage(
                                model_to_edit, 
                                union_neurons, 
                                joint_edit_data, 
                                logits_pristine_joint, 
                                "joint", 
                                early_stopping_threshold=hparams.EARLY_STOP_THRESHOLD_JOINT
                            )
                            
                            metrics, details = evaluate_model_on_task(edited_model, processor, formatted_task, hparams, pre_responses, verbose=True)
                            task_results[strategy] = {"metrics": metrics, "details": details}
                        else:
                             task_results[strategy] = {"message": "No neurons to edit."}
                    
                    elif strategy == 'sequential_text_first':
                        task_results[strategy] = {}
                        edited_model_s1 = editor.edit_stage(model_to_edit, text_neurons, text_edit_data, logits_pristine_text, "Text", early_stopping_threshold=hparams.EARLY_STOP_THRESHOLD_TEXT)
                        metrics_stage1, details_stage1 = evaluate_model_on_task(edited_model_s1, processor, formatted_task, hparams, pre_responses, verbose=True)
                        task_results[strategy]["stage1_text_metrics"] = metrics_stage1
                        task_results[strategy]["stage1_text_details"] = details_stage1
                        
                        edited_model_s2 = editor.edit_stage(edited_model_s1, image_neurons, image_edit_data, logits_pristine_image, "Image", early_stopping_threshold=hparams.EARLY_STOP_THRESHOLD_IMAGE)
                        metrics_stage2, details_stage2 = evaluate_model_on_task(edited_model_s2, processor, formatted_task, hparams, pre_responses, verbose=True)
                        task_results[strategy]["stage2_image_metrics_final"] = metrics_stage2
                        task_results[strategy]["stage2_image_details_final"] = details_stage2
                    
                    elif strategy == 'sequential_image_first':
                        task_results[strategy] = {}
                        edited_model_s1 = editor.edit_stage(model_to_edit, image_neurons, image_edit_data, logits_pristine_image, "Image", early_stopping_threshold=hparams.EARLY_STOP_THRESHOLD_IMAGE)
                        metrics_stage1, details_stage1 = evaluate_model_on_task(edited_model_s1, processor, formatted_task, hparams, pre_responses, verbose=True)
                        task_results[strategy]["stage1_image_metrics"] = metrics_stage1
                        task_results[strategy]["stage1_image_details"] = details_stage1
                        
                        edited_model_s2 = editor.edit_stage(edited_model_s1, text_neurons, text_edit_data, logits_pristine_text, "Text", early_stopping_threshold=hparams.EARLY_STOP_THRESHOLD_TEXT)
                        metrics_stage2, details_stage2 = evaluate_model_on_task(edited_model_s2, processor, formatted_task, hparams, pre_responses, verbose=True)
                        task_results[strategy]["stage2_text_metrics_final"] = metrics_stage2
                        task_results[strategy]["stage2_text_details_final"] = details_stage2

                except torch.OutOfMemoryError:
                    print(f"  [OOM] Strategy {strategy} failed due to Out of Memory.")
                    task_results[strategy] = {'error': 'CUDA OOM'}
                    torch.cuda.empty_cache(); gc.collect()
                except Exception as e:
                    print(f"  [Critical Error] Strategy {strategy} failed: {e}"); traceback.print_exc()
                    task_results[strategy] = {'error': str(e)}
                finally:
                    editor.restore_model(model_to_edit)
                    gc.collect(); torch.cuda.empty_cache()
            
            all_results.append(task_results)
            with open(FINAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
                
    print(f"\n--- Experiment Complete! Results saved to: {FINAL_RESULTS_PATH} ---")

if __name__ == "__main__":
    main()