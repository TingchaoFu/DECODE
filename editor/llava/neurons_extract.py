
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import os
from tqdm import tqdm
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
import traceback
from collections import defaultdict

MODEL_ID = "your_path/llava-1.5-7b-hf"
DATASET_PATH = "your_path/vqa.json"
IMAGE_BASE_DIR = "your_path/data_image"

RESULTS_DIR = "your_path/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, f"neurons_llava.json") 
NUM_LAYERS_TO_EXCLUDE = 3
TOP_K_NEURONS_PER_MODALITY = 40

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--- Step 1/5: Loading LLaVA model... ---")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
).to(device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
main_device, main_dtype = model.device, model.dtype
print(f"Model loaded. Main device: {main_device}, Main dtype: {main_dtype}")

print("\n--- Step 2/5: Manually loading Unembedding Matrix... ---")
unembedding_matrix, unembedding_key = None, None; target_weight_tensor = model.get_output_embeddings().weight
for name, param in model.named_parameters():
    if param is target_weight_tensor: unembedding_key = name; break
if not unembedding_key: raise RuntimeError("Could not dynamically find Unembedding Matrix name.")
model_path = Path(MODEL_ID); weight_files_to_check = []
index_path = model_path / "model.safetensors.index.json"
if not index_path.exists(): index_path = model_path / "pytorch_model.bin.index.json"
if index_path.exists():
    with open(index_path, 'r') as f: index = json.load(f)
    alternative_key = f"language_model.{unembedding_key}"
    unembedding_filename = index["weight_map"].get(unembedding_key) or index["weight_map"].get(alternative_key)
    if unembedding_filename: weight_files_to_check.append(model_path / unembedding_filename)
    else: weight_files_to_check.extend(list({model_path / f for f in index["weight_map"].values()}))
else:
    for f in [model_path / "model.safetensors", model_path / "pytorch_model.bin"]:
        if f.exists(): weight_files_to_check.append(f); break
if not weight_files_to_check: raise FileNotFoundError(f"No weight files found in directory '{MODEL_ID}'.")
for weight_path in weight_files_to_check:
    if not weight_path.exists(): continue
    try:
        weights = load_safetensors(weight_path) if weight_path.suffix == ".safetensors" else torch.load(weight_path, map_location="cpu", weights_only=True)
        if unembedding_key in weights: unembedding_matrix = weights[unembedding_key]; break
        elif alternative_key in weights: unembedding_matrix = weights[alternative_key]; break
    except Exception as e: print(f"  Error loading file '{weight_path.name}': {e}"); continue
if unembedding_matrix is None: raise RuntimeError(f"Could not load data for '{unembedding_key}'.")
unembedding_matrix = unembedding_matrix.to(device=main_device, dtype=main_dtype)
print("Unembedding Matrix loaded successfully.")

total_layers = len(model.language_model.model.layers)
LAYERS_TO_ANALYZE = list(range(total_layers - NUM_LAYERS_TO_EXCLUDE))
print(f"\n--- Step 3/5: Model has {total_layers} layers. Analyzing first {len(LAYERS_TO_ANALYZE)} layers. ---")

activations_storage = {}
def get_activation_hook(layer_name):
    def hook(module, input, output):
        activations_storage[layer_name] = input[0].detach().cpu()
    return hook

hooks = []
for layer_idx in LAYERS_TO_ANALYZE:
    module = model.language_model.model.layers[layer_idx].mlp.down_proj
    hooks.append(module.register_forward_hook(get_activation_hook(f"layer_{layer_idx}")))
print(f"Successfully registered hooks on {len(hooks)} MLP down_proj layers to capture inputs.")

def calculate_top_k_for_all_tokens(
    case_info, modality, layers_to_analyze: list, unembed_matrix: torch.Tensor
):
    target_text = case_info['alt']
    question, image_input = None, None

    if modality == 'text':
        image_input = None
        question = case_info.get('t_rel')
    elif modality in ['image', 'multimodal']:
        image_path = Path(IMAGE_BASE_DIR) / case_info.get('image', '')
        if not image_path.exists(): return {}
        image_input = Image.open(image_path).convert("RGB")
        question = case_info.get('i_rel') if modality == 'image' else case_info.get('t_rel')
    if not question: return {}

    prompt_text = f"USER: <image>\n{question}\nASSISTANT:" if image_input is not None else f"USER: {question}\nASSISTANT:"

    try:
        full_text = f"{prompt_text} {target_text}"
        prompt_inputs = processor(text=prompt_text, images=image_input, return_tensors="pt") if image_input is not None else processor(text=prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        activations_storage.clear()
        full_inputs = processor(text=full_text, images=image_input, return_tensors="pt").to(main_device) if image_input is not None else processor(text=full_text, return_tensors="pt").to(main_device)
        with torch.no_grad():
            model(**full_inputs)

        target_ids = full_inputs.input_ids[0, prompt_len:]
        if len(target_ids) == 0: return {}

        results_per_token = defaultdict(list)
        
        for t_idx, current_target_token in enumerate(target_ids):
            all_neuron_scores_for_token = []
            
            with torch.no_grad():
                for l_idx in layers_to_analyze:
                    l_name = f"layer_{l_idx}"
                    act_tensor = activations_storage.get(l_name)
                    if act_tensor is None: continue
                    act_tensor = act_tensor.to(unembed_matrix.device)

                    down_proj = model.language_model.model.layers[l_idx].mlp.down_proj
                    down_proj_matrix = down_proj.weight
                    
                    proj_matrix = unembed_matrix @ down_proj_matrix

                    activation_position = (prompt_len - 1) + t_idx
                    if activation_position >= act_tensor.shape[1]: continue 

                    acts_at_pos = act_tensor[0, activation_position, :]
                    
                    q = acts_at_pos.to(dtype=main_dtype)
                    projection_for_token = proj_matrix[current_target_token.item(), :]
                    neuron_contributions = (q * projection_for_token).to(torch.float32)

                    for n_idx, score in enumerate(neuron_contributions.cpu().numpy()):
                        all_neuron_scores_for_token.append((float(score), l_idx, n_idx))
            
            all_neuron_scores_for_token.sort(key=lambda x: x[0], reverse=True)
            results_per_token[current_target_token.item()] = all_neuron_scores_for_token[:TOP_K_NEURONS_PER_MODALITY]
        
        return results_per_token

    except Exception as e:
        print(f"  [Warning] Error in case {case_info.get('id', 'N/A')} modality {modality}: {e}")
        traceback.print_exc()
        return {}


print("\n--- Step 4/5: Starting integrated dataset processing... ---")
final_results, dataset_len = {}, 0
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        print(f"Successfully loaded {len(final_results)} existing results. Resuming.")
    except json.JSONDecodeError:
        print(f"Warning: Result file {OUTPUT_PATH} corrupted or empty. Restarting.")
        final_results = {}
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)
    dataset_len = len(dataset)
for i, case_data in tqdm(enumerate(dataset), desc="Analyzing cases", total=dataset_len):
    case_id = str(i)
    if case_id in final_results: continue
    image_path = Path(IMAGE_BASE_DIR) / case_data.get('image', '')
    if not image_path.exists(): continue

    case_results = {}
    for modality in ["text", "image", "multimodal"]:
        try:
            case_results[modality] = calculate_top_k_for_all_tokens(
                case_info=case_data, modality=modality, layers_to_analyze=LAYERS_TO_ANALYZE, unembed_matrix=unembedding_matrix
            )
        except Exception as e:
            print(f"  [Warning] Critical error processing case {case_id} modality {modality}: {e}"); traceback.print_exc()
            case_results[modality] = {}

    def aggregate_neurons_from_tokens(token_results: dict):
        aggregated_set = set()
        for token_id, neurons_with_scores in token_results.items():
            neuron_tuples = {tuple(n[1:]) for n in neurons_with_scores}
            aggregated_set.update(neuron_tuples)
        return sorted([list(n) for n in aggregated_set])

    text_neurons = aggregate_neurons_from_tokens(case_results.get('text', {}))
    image_neurons = aggregate_neurons_from_tokens(case_results.get('image', {}))
    multimodal_neurons = aggregate_neurons_from_tokens(case_results.get('multimodal', {}))

    text_set, image_set = {tuple(n) for n in text_neurons}, {tuple(n) for n in image_neurons}
    agnostic_set = text_set.intersection(image_set)
    joint_set = text_set.union(image_set)

    def get_flat_list_with_scores(token_results: dict):
        flat_list = []
        for _, neurons in token_results.items():
            flat_list.extend(neurons)
        return flat_list
    
    agnostic_sorted = sorted([n for n in get_flat_list_with_scores(case_results.get('text', {})) if tuple(n[1:]) in agnostic_set], key=lambda x: x[0], reverse=True)
    text_specific_sorted = sorted([n for n in get_flat_list_with_scores(case_results.get('text', {})) if tuple(n[1:]) not in image_set], key=lambda x: x[0], reverse=True)
    image_specific_sorted = sorted([n for n in get_flat_list_with_scores(case_results.get('image', {})) if tuple(n[1:]) not in text_set], key=lambda x: x[0], reverse=True)
    
    final_results[case_id] = {
        "text_specific": text_neurons,
        "image_specific": image_neurons,
        "multimodal": multimodal_neurons,
        "agnostic": sorted(list(map(list, agnostic_set))),
        "joint": sorted(list(map(list, joint_set))),
        "raw_results_per_token": case_results
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

print("\n--- Step 5/5: Cleaning up hooks... ---")
for h in hooks:
    h.remove()

print(f"\nIntegrated analysis complete!")
print(f"All neuron location results saved to: {OUTPUT_PATH}")
print("Process aligned with FiNE multi-token analysis.")