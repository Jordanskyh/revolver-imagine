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

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        # Inject (safetensors)
        config['model_arguments']['pretrained_model_name_or_path'] = model_path

        lrs_config = load_lrs_config(model_type, is_style)

        if lrs_config:
            model_hash = hash_model(model_name)
            
            # Count images in training directory for size-specific LR tuning
            dataset_size = count_images_in_directory(train_data_dir)
            
            lrs_settings = get_config_for_model(lrs_config, model_hash, dataset_size)

            if lrs_settings:
                print(f"Applying LRS overrides: {list(lrs_settings.keys())}", flush=True)
                
                # Section mapping for standard keys
                section_map = {
                    "unet_lr": ("optimizer_arguments", "learning_rate"),
                    "text_encoder_lr": ("optimizer_arguments", "text_encoder_lr"),
                    "min_snr_gamma": ("optimizer_arguments", "min_snr_gamma"),
                    "noise_offset": ("training_arguments", "noise_offset"),
                    "max_grad_norm": ("optimizer_arguments", "max_grad_norm"),
                    "lr_scheduler": ("optimizer_arguments", "lr_scheduler"),
                    "lr_warmup_steps": ("optimizer_arguments", "lr_warmup_steps"),
                    "optimizer_type": ("optimizer_arguments", "optimizer_type"),
                    "optimizer_args": ("optimizer_arguments", "optimizer_args"),
                    "network_dim": ("additional_network_arguments", "network_dim"),
                    "network_alpha": ("additional_network_arguments", "network_alpha"),
                    "network_dropout": ("additional_network_arguments", "network_dropout"),
                }

                for key, value in lrs_settings.items():
                    if value is None:
                        continue
                        
                    # Prodigy Fix: If text_encoder_lr is the same as unet_lr, don't set it separately.
                    # This avoids creating separate parameter groups that crash Prodigy.
                    if key == "text_encoder_lr" and str(lrs_settings.get("unet_lr")) == str(value):
                         continue

                    if key in section_map:
                        section, target_key = section_map[key]
                        if section not in config:
                            config[section] = {}
                        config[section][target_key] = value
                    else:
                        # Direct injection for root keys (max_train_epochs, etc.)
                        config[key] = value
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

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
            "dataautogpt3/TempestV0.1": 456,
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
            888: {"network_dim": 128, "network_alpha": 128, "network_args": []}
        }

        config["pretrained_model_name_or_path"] = model_path
    
    # Determine which dictionary to use based on task type
    if model_type == "flux":
        target_dict = network_config_flux
    elif model_type == "qwen-image":
        target_dict = network_config_qwen
    elif model_type == "z-image":
        target_dict = {} # Z-Image logic is handled by Autoepoch/LRS mostly
    else:
        target_dict = network_config_style if is_style else network_config_person
    
    # Lookup Model Config ID (Default to 235/Rank 32 if not found)
    config_id = target_dict.get(model_name, 235)
    
    # Retrieve Network Parameters from Mapping
    model_params = config_mapping.get(config_id, config_mapping[235])
    
    net_dim = model_params["network_dim"]
    net_alpha = model_params["network_alpha"]
    net_args = model_params["network_args"]
    
    print(f"⚡ CHAMPION LOGIC: Model '{model_name}' mapped to ID {config_id} (Rank {net_dim})", flush=True)
    # --------------------------------------------------

    if model_type in ["sdxl", "flux"]:
        if "additional_network_arguments" not in config:
            config["additional_network_arguments"] = {}
        if "dataset_arguments" not in config:
            config["dataset_arguments"] = {}
            
        config["dataset_arguments"]["debug_dataset"] = False
        config["additional_network_arguments"]["network_dim"] = net_dim
        config["additional_network_arguments"]["network_alpha"] = net_alpha
        config["additional_network_arguments"]["network_args"] = net_args

        # LAYER 3: Autoepoch (Size-Ranges Config)
        dataset_size = count_images_in_directory(train_data_dir)
        if dataset_size > 0:
            size_config = load_size_based_config(model_type, is_style, dataset_size)
            if size_config:
                print(f"Applying size-based config from Autoepoch for {model_type} ({dataset_size} images)", flush=True)
                for key, value in size_config.items():
                    config[key] = value

        # Output Dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if "training_arguments" not in config:
             config["training_arguments"] = {}
        config["training_arguments"]["output_dir"] = output_dir
        
        config["dataset_arguments"]["train_data_dir"] = train_data_dir
        config["train_data_dir"] = train_data_dir
        
        if "model_arguments" not in config:
             config["model_arguments"] = {}
        config["model_arguments"]["pretrained_model_name_or_path"] = model_path
        config["pretrained_model_name_or_path"] = model_path
        
        config["dataset_arguments"]["enable_bucket"] = True
        config.setdefault("enable_bucket", True)

    elif is_ai_toolkit:
        # AI-Toolkit Handle (Qwen/Z-Image)
        for process in config['config']['process']:
            if 'model' in process:
                process['model']['name_or_path'] = model_path
                output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                process['training_folder'] = output_dir
            
            if 'datasets' in process:
                for dataset in process['datasets']:
                    dataset['folder_path'] = train_data_dir

            if 'adapter' in process:
                process['adapter']['rank'] = net_dim
                process['adapter']['alpha'] = net_alpha

            if trigger_word:
                process['trigger_word'] = trigger_word

        # LAYER 3: Autoepoch for AI-Toolkit
        dataset_size = count_images_in_directory(train_data_dir)
        if dataset_size > 0:
            size_config = load_size_based_config(model_type, is_style, dataset_size)
            if size_config:
                print(f"Applying size-based config from Autoepoch for {model_type} ({dataset_size} images)", flush=True)
                # Map YAML keys if needed, but for now assuming direct keys work in AI-Toolkit config structure
                # Note: This might need more specific mapping for YAML
                for key, value in size_config.items():
                    # For AI-Toolkit, we usually want these in the process[0] level or globally
                    config['config']['process'][0][key] = value

    dataset_size = 0
    if os.path.exists(train_data_dir):
        dataset_size = count_images_in_directory(train_data_dir)
        if dataset_size > 0:
            print(f"Counted {dataset_size} images in training directory", flush=True)


    
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"config is {config}", flush=True)
    print(f"Created config at {config_path}", flush=True)
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
