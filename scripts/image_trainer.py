#!/usr/bin/env python3

"""
Jordansky
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        # Filter for safetensors files
        files = [f for f in os.listdir(path) if f.endswith(".safetensors") and os.path.isfile(os.path.join(path, f))]
        
        if len(files) > 0:
            # If multiple safetensors, pick the largest one (likely the model)
            largest_file = max(files, key=lambda f: os.path.getsize(os.path.join(path, f)))
            return os.path.join(path, largest_file)
            
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        for k, v in default_config.items():
            if v is not None:
                merged[k] = v

    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if v is not None:
                merged[k] = v

    return merged if merged else None

def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count

def load_size_based_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "autoepoch") # Point to autoepoch dir
    
    if model_type == "flux":
        config_file = os.path.join(config_dir, "a-epochflux.json")
    elif model_type == "qwen-image":
        config_file = os.path.join(config_dir, "a-epochqwen.json")
    elif model_type == "z-image":
        config_file = os.path.join(config_dir, "a-epochz.json")
    elif is_style:
        config_file = os.path.join(config_dir, "a-epochstyle.json")
    else:
        config_file = os.path.join(config_dir, "a-epochperson.json")
    
    try:
        if not os.path.exists(config_file):
            print(f"Warning: Autoepoch config file not found: {config_file}", flush=True)
            return None
            
        with open(config_file, 'r') as f:
            size_config = json.load(f)
        
        size_ranges = size_config.get("size_ranges", [])
        for size_range in size_ranges:
            min_size = size_range.get("min", 0)
            max_size = size_range.get("max", float('inf'))
            
            if min_size <= dataset_size <= max_size:
                print(f"Using size-based config for {dataset_size} images (range: {min_size}-{max_size})", flush=True)
                return size_range.get("config", {})
        
        default_config = size_config.get("default", {})
        if default_config:
            print(f"Using default size-based config for {dataset_size} images", flush=True)
        return default_config
        
    except Exception as e:
        print(f"Warning: Could not load autoepoch config from {config_file}: {e}", flush=True)
        return None

def get_dataset_size_category(dataset_size: int) -> str:
    if dataset_size <= 15:
        return "small"
    elif dataset_size <= 35:
        return "medium"
    else:
        return "large"

def get_config_for_model(lrs_config: dict, model_name: str, dataset_size: int = None) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        model_config = data.get(model_name)
        
        # If dataset_size provided and model_config has size categories, merge them
        if dataset_size is not None and isinstance(model_config, dict):
            size_category = get_dataset_size_category(dataset_size)
            
            # Check if model_config has size-specific settings
            if size_category in model_config:
                size_specific_config = model_config.get(size_category, {})
                # Merge: default → model_config (non-size keys) → size_specific
                base_model_config = {k: v for k, v in model_config.items() if k not in ["small", "medium", "large"]}
                merged = merge_model_config(default_config, base_model_config)
                return merge_model_config(merged, size_specific_config)
        
        return merge_model_config(default_config, model_config)

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == ImageModelType.FLUX.value:
        config_file = os.path.join(config_dir, "flux.json")
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        config_file = os.path.join(config_dir, "qwen.json")
    elif model_type == ImageModelType.Z_IMAGE.value:
        config_file = os.path.join(config_dir, "zimage.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word):
    is_style = "style" in model_name.lower() or "style" in task_id.lower()
    train_data_dir = os.path.join(train_cst.IMAGE_CONTAINER_IMAGES_PATH, task_id)
    
    # Try specific template (e.g., base_diffusion_flux_person.toml)
    config_template_path = os.path.join(script_dir, "core", "config", f"base_diffusion_{model_type}_{'style' if is_style else 'person'}.toml")
    
    # Fallback to base template (e.g., base_diffusion_flux.toml)
    if not os.path.exists(config_template_path):
        config_template_path = os.path.join(script_dir, "core", "config", f"base_diffusion_{model_type}.toml")

    if model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]:
        config_template_path = os.path.join(script_dir, "core", "config", f"base_{model_type}.yaml")

    # --- LAYER 2: DICTIONARY & MAPPING (CHAMPION LOGIC) ---
    network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 467,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 467,
        "dataautogpt3/TempestV0.1": 500,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 467,
        "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228,
        "recoilme/colorfulxl": 228,
        "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
        "zenless-lab/sdxl-anything-xl": 228,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 228,
        "bghira/terminus-xl-velocity-v2": 235,
        "ifmain/UltraReal_Fine-Tune": 467
    }

    network_config_style = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 235,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 235,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 235,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 235,
        "dataautogpt3/TempestV0.1": 500,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 235,
        "fluently/Fluently-XL-Final": 235,
        "mann-e/Mann-E_Dreams": 235,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 235,
        "recoilme/colorfulxl": 235,
        "zenless-lab/sdxl-aam-xl-anime-mix": 235,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
        "zenless-lab/sdxl-anything-xl": 235,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
        "Corcelio/mobius": 235,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 235,
        "bghira/terminus-xl-velocity-v2": 235,
        "ifmain/UltraReal_Fine-Tune": 235
    }

    network_config_flux = {
        "dataautogpt3/FLUX-MonochromeManga": 350,
        "mikeyandfriends/PixelWave_FLUX.1-dev_03": 350,
        "rayonlabs/FLUX.1-dev": 350,
        "mhnakif/fluxunchained-dev": 350
    }

    network_config_qwen = {
        "gradients-io-tournaments/Qwen-Image": 888,
        "gradients-io-tournaments/Qwen-Image-Jib-Mix": 888
    }

    config_mapping = {
        228: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=8", "conv_alpha=8", "algo=locon"]},
        235: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=8", "conv_alpha=8", "algo=locon"]},
        456: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=16", "conv_alpha=16", "algo=locon"]},
        467: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=16", "conv_alpha=16", "algo=locon"]},
        699: {"network_dim": 96, "network_alpha": 96, "network_args": ["conv_dim=32", "conv_alpha=32", "algo=locon"]},
        900: {"network_dim": 128, "network_alpha": 128, "network_args": ["conv_dim=32", "conv_alpha=32", "algo=locon"]},
        500: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=0"]},
        350: {"network_dim": 32, "network_alpha": 32, "network_args": []},
        888: {"network_dim": 96, "network_alpha": 96, "network_args": []},
        999: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=32", "conv_alpha=32"]}
    }

    # Determine Model Config ID
    if model_type == "z-image":
        config_id = 999
    elif model_type == "flux":
        config_id = network_config_flux.get(model_name, 350)
    elif model_type == "qwen-image":
        config_id = network_config_qwen.get(model_name, 888)
    else:
        target_dict = network_config_style if is_style else network_config_person
        config_id = target_dict.get(model_name, 235)

    model_params = config_mapping.get(config_id, config_mapping[235])
    net_dim = model_params["network_dim"]
    net_alpha = model_params["network_alpha"]
    net_args = model_params["network_args"]

    print(f"⚡ JORDANSKY LAYER 2: Model '{model_name}' Rank {net_dim}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    # --- PREPARE OVERRIDES (LRS & AUTOEPOCH) ---
    lrs_settings = None
    size_config = None
    
    lrs_config = load_lrs_config(model_type, is_style)
    if lrs_config:
        model_hash = hash_model(model_name)
        dataset_size = count_images_in_directory(train_data_dir)
        lrs_settings = get_config_for_model(lrs_config, model_hash, dataset_size)

    dataset_size = count_images_in_directory(train_data_dir)
    if dataset_size > 0:
        size_config = load_size_based_config(model_type, is_style, dataset_size)

    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        
        process = config['config']['process'][0]
        process['model']['name_or_path'] = model_path
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        process['training_folder'] = output_dir
        
        for dataset in process['datasets']:
            dataset['folder_path'] = train_data_dir

        if 'adapter' in process:
            process['adapter']['rank'] = net_dim
            process['adapter']['alpha'] = net_alpha
        
        # Apply Overrides to YAML
        if 'train' not in process: process['train'] = {}
        
        # Helper to calculate steps based on epochs for AI-Toolkit
        def calculate_steps(epochs):
            batch_size = process.get('train', {}).get('batch_size', 1)
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size == 0: return epochs # Fallback
            return int(epochs * (dataset_size / batch_size))

        # Priority 1: Autoepoch
        if size_config:
            for key, value in size_config.items():
                if key == "max_train_epochs": 
                    process['train']['steps'] = calculate_steps(value)
                elif key == "optimizer_type": 
                    process['train']['optimizer'] = value
                else: 
                    process['train'][key] = value

        # Priority 2: LRS (Superior)
        if lrs_settings:
            for key, value in lrs_settings.items():
                if key in ["unet_lr", "text_encoder_lr", "learning_rate"]: 
                    process['train']['lr'] = value
                elif key in ["optimizer_type", "optimizer"]: 
                    process['train']['optimizer'] = value
                elif key in ["max_train_epochs", "steps"]: 
                    calculated_val = calculate_steps(value) if key == "max_train_epochs" else value
                    process['train']['steps'] = calculated_val
                elif key == "optimizer_args" and isinstance(value, list):
                    # CONVERSION FIX: List["k=v"] -> Dict{"k": v} for AI-Toolkit
                    opt_params = {}
                    for item in value:
                        if "=" in item:
                            k, v = item.split("=", 1)
                            # Simple type inference
                            if v.lower() == "true": v = True
                            elif v.lower() == "false": v = False
                            else:
                                try: v = float(v) if "." in v else int(v)
                                except ValueError: pass
                            opt_params[k.strip()] = v
                    process['train']['optimizer_params'] = opt_params
                else: 
                    process['train'][key] = value

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        config['model_arguments']['pretrained_model_name_or_path'] = model_path
        
        # FLUX Component Auto-Pathing (Battle-Hardened Autodiscovery)
        if model_type == "flux":
            def find_best_file(base_dir, pattern, min_gb=0, max_gb=100, preferred_dir=None, avoid=None):
                candidates = []
                for root, dirs, files in os.walk(base_dir):
                    for f in files:
                        if f.endswith(".safetensors") and pattern in f.lower():
                            if avoid and avoid in f.lower(): continue
                            
                            path = os.path.join(root, f)
                            size_gb = os.path.getsize(path) / (1024**3)
                            if min_gb <= size_gb <= max_gb:
                                score = 0
                                if preferred_dir and preferred_dir in root.lower(): score += 10
                                if pattern in f.lower(): score += 5
                                candidates.append((score, path))
                
                if candidates:
                    # Sort by score descending, then by path
                    candidates.sort(key=lambda x: (-x[0], x[1]))
                    return candidates[0][1]
                return None

            base_models = "/cache/models"
            
            # 1. Find AE (AutoEncoder) - (~300MB)
            ae_path = find_best_file(os.path.dirname(model_path), "ae", max_gb=0.5) or find_best_file(base_models, "ae", max_gb=0.5)
            if ae_path: config['model_arguments']['ae'] = ae_path
            
            # 2. Find CLIP-L - (STRICT: 100MB - 400MB)
            # This is critical to avoid SDXL's 650MB-1.3GB text encoders.
            clip_l_path = find_best_file(base_models, "model", min_gb=0.1, max_gb=0.42, preferred_dir="clip-vit-large-patch14")
            if not clip_l_path:
                clip_l_path = find_best_file(base_models, "clip", min_gb=0.1, max_gb=0.42)
                
            if clip_l_path: 
                config['model_arguments']['clip_l'] = clip_l_path
            
            # 3. Find T5XXL - BIG file (>4GB), AVOID shards ("of")
            t5_path = find_best_file(base_models, "t5", min_gb=4.0, avoid="of") or find_best_file(base_models, "model", min_gb=4.0, avoid="of")
            # Fallback if no unified T5, try to pick the largest one anyway (riskier)
            if not t5_path:
                t5_path = find_best_file(base_models, "t5", min_gb=4.0)

            if t5_path: 
                config['model_arguments']['t5xxl'] = t5_path

            # Log discovered paths for debugging
            print(f"🔍 FLUX TARGETING SYSTEM:\n   AE    : {config['model_arguments'].get('ae')}\n   CLIP-L: {config['model_arguments'].get('clip_l')}\n   T5-XXL: {config['model_arguments'].get('t5xxl')}", flush=True)

        config['train_data_dir'] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir

        # Apply Overrides to TOML
        section_map = {
            "unet_lr": ("optimizer_arguments", "learning_rate"),
            "text_encoder_lr": ("optimizer_arguments", "text_encoder_lr"),
            "min_snr_gamma": ("optimizer_arguments", "min_snr_gamma"),
            "noise_offset": ("training_arguments", "noise_offset"),
            "optimizer_type": ("optimizer_arguments", "optimizer_type"),
            "optimizer_args": ("optimizer_arguments", "optimizer_args"),
        }

        # Apply Overrides (Priority: Autoepoch < LRS)
        configs_to_apply = [size_config, lrs_settings]
        
        for cfg in configs_to_apply:
            if not cfg:
                continue
                
            for key, value in cfg.items():
                # Prodigy Fix: If text_encoder_lr is the same as unet_lr, don't set it separately.
                if key == "text_encoder_lr" and str(cfg.get("unet_lr")) == str(value):
                    continue

                if key in section_map:
                    sec, target = section_map[key]
                    if sec not in config:
                        config[sec] = {}
                    config[sec][target] = value
                else:
                    # Direct injection for root keys (max_train_epochs, train_batch_size, etc.)
                    config[key] = value

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"Created config at {config_path}", flush=True)
        return config_path
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)
    with open(config_path, "r") as f:
        print(f"--- CONFIG CONTENT ---\n{f.read()}\n--- END CONFIG ---", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-script/{model_type}_train_network.py",
                "--config_file", config_path
            ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)
        
        # --- FIX: MOVE FILE IF SAVED IN WRONG LOCATION ---
        try:
            # Parse config to get intended output directory
            intended_output_dir = None
            if config_path.endswith(".toml"):
                import toml
                with open(config_path, "r") as f:
                    c = toml.load(f)
                    intended_output_dir = c.get("output_dir")
            
            if intended_output_dir:
                default_loc = "/app/checkpoints/last.safetensors"
                if os.path.exists(default_loc):
                    print(f"[FIX] Moving checkpoint from {default_loc} to {intended_output_dir}", flush=True)
                    os.makedirs(intended_output_dir, exist_ok=True)
                    import shutil
                    shutil.move(default_loc, os.path.join(intended_output_dir, "last.safetensors"))
                    print(f"[FIX] Successfully moved to {intended_output_dir}/last.safetensors", flush=True)
        except Exception as e:
            print(f"[FIX] Error moving checkpoint: {e}", flush=True)
        # ------------------------------------------------

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = get_model_path(train_paths.get_image_base_model_path(args.model))

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
