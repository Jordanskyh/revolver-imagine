#!/usr/bin/env python3

"""
everything u are
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
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

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


def load_autoepoch_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "autoepoch")
    
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
            return None
        with open(config_file, 'r') as f:
            epoch_config = json.load(f)
        
        size_ranges = epoch_config.get("size_ranges", [])
        for size_range in size_ranges:
            if size_range.get("min", 0) <= dataset_size <= size_range.get("max", float('inf')):
                print(f"Applying Autoepoch settings from {os.path.basename(config_file)} for {dataset_size} images", flush=True)
                return size_range.get("config", {})
        
        return epoch_config.get("default", {})
    except Exception as e:
        print(f"Warning: Could not load Autoepoch config: {e}", flush=True)
        return None

def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif model_type == "qwen-image":
        config_file = os.path.join(config_dir, "qwen.json")
    elif model_type == "z-image":
        config_file = os.path.join(config_dir, "zimage.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        
        # 1. Base Paths
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir
                if trigger_word:
                    process['trigger_word'] = trigger_word

                # 2. Tiered Logic for AI-Toolkit
                lrs_config = load_lrs_config(model_type, is_style)
                dataset_size = count_images_in_directory(train_data_dir)
                ae_config = load_autoepoch_config(model_type, is_style, dataset_size) if dataset_size > 0 else None
                
                # Apply LRS (Tier 2) and Autoepoch (Tier 3)
                for overrides in [load_lrs_config(model_type, is_style), ae_config]:
                    if overrides:
                        settings = get_config_for_model(overrides, hash_model(model_name)) if overrides == lrs_config else overrides
                        if settings:
                            # Map keys to YAML structure
                            train_node = process.get('train', {})
                            network_node = process.get('network', {})
                            for k, v in settings.items():
                                if k == "train_batch_size": train_node['batch_size'] = v
                                elif k == "max_train_steps": train_node['steps'] = v
                                elif k == "unet_lr": train_node['lr'] = v
                                elif k == "network_dim": network_node['linear'] = v
                                elif k == "network_alpha": network_node['linear_alpha'] = v
                                # Add other mappings as needed
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        if model_type == "sdxl":
            # 1. Base Logic: Apply Hardcoded network configs first (can be overridden by LRS/Autoepoch)
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
                "ifmain/UltraReal_Fine-Tune": 467,
                "bghira/terminus-xl-velocity-v2": 235
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
                "dataautogpt3/TempestV0.1": 228,
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
                "ifmain/UltraReal_Fine-Tune": 235,
                "bghira/terminus-xl-velocity-v2": 235
            }

            config_mapping = {
                228: {"network_dim": 32, "network_alpha": 32, "network_args": []},
                235: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]},
                456: {"network_dim": 64, "network_alpha": 64, "network_args": []},
                467: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]},
                699: {"network_dim": 96, "network_alpha": 96, "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]}
            }

            if is_style:
                network_config = config_mapping.get(network_config_style.get(model_name, 235), config_mapping[235])
            else:
                network_config = config_mapping.get(network_config_person.get(model_name, 235), config_mapping[235])

            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]

        # 2. LRS Logic: Hash-based overrides (Tier 2)
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)
            if lrs_settings:
                print(f"Applying LRS hash overrides for {model_name} ({model_hash})", flush=True)
                for key, value in lrs_settings.items():
                    config[key] = value

        # 3. Autoepoch Logic: Size-based math (Tier 3)
        dataset_size = 0
        if os.path.exists(train_data_dir):
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size > 0:
                print(f"Counted {dataset_size} images in training directory", flush=True)

        if dataset_size > 0:
            ae_config = load_autoepoch_config(model_type, is_style, dataset_size)
            if ae_config:
                print(f"Applying Autoepoch efficiency overrides for {dataset_size} images", flush=True)
                for key, value in ae_config.items():
                    config[key] = value
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

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

    model_path = train_paths.get_image_base_model_path(args.model)

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
