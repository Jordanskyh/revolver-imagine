import hashlib
import json

person_models = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "Lykon/dreamshaper-xl-1-0",
    "Lykon/art-diffusion-xl-0.9",
    "SG161222/RealVisXL_V4.0",
    "stablediffusionapi/protovision-xl-v6.6",
    "stablediffusionapi/omnium-sdxl",
    "GraydientPlatformAPI/realism-engine2-xl",
    "GraydientPlatformAPI/albedobase2-xl",
    "KBlueLeaf/Kohaku-XL-Zeta",
    "John6666/hassaku-xl-illustrious-v10style-sdxl",
    "John6666/nova-anime-xl-pony-v5-sdxl",
    "cagliostrolab/animagine-xl-4.0",
    "dataautogpt3/CALAMITY",
    "dataautogpt3/ProteusSigma",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/TempestV0.1",
    "ehristoforu/Visionix-alpha",
    "femboysLover/RealisticStockPhoto-fp16",
    "fluently/Fluently-XL-Final",
    "mann-e/Mann-E_Dreams",
    "misri/leosamsHelloworldXL_helloworldXL70",
    "misri/zavychromaxl_v90",
    "openart-custom/DynaVisionXL",
    "recoilme/colorfulxl",
    "zenless-lab/sdxl-aam-xl-anime-mix",
    "zenless-lab/sdxl-anima-pencil-xl-v5",
    "zenless-lab/sdxl-anything-xl",
    "zenless-lab/sdxl-blue-pencil-xl-v7",
    "Corcelio/mobius",
    "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16",
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "bghira/terminus-xl-velocity-v2",
    "ifmain/UltraReal_Fine-Tune"
]

style_models = person_models # They are the same in image_trainer.py

def build_config(model_list):
    config = {
        "default": {},
        "data": {}
    }
    for m in model_list:
        h = hashlib.sha256(m.encode('utf-8')).hexdigest()
        config["data"][h] = {
            "model_name": m,
            "small": {
                "unet_lr": None,
                "text_encoder_lr": None,
                "noise_offset": None,
                "min_snr_gamma": None
            },
            "medium": {
                "unet_lr": None,
                "text_encoder_lr": None,
                "noise_offset": None,
                "min_snr_gamma": None
            },
            "large": {
                "unet_lr": None,
                "text_encoder_lr": None,
                "noise_offset": None,
                "min_snr_gamma": None
            }
        }
    return config

person_json = build_config(person_models)
# Apply the only optimization we have for now to Animagine Small
animagine_hash = hashlib.sha256("cagliostrolab/animagine-xl-4.0".encode('utf-8')).hexdigest()
person_json["data"][animagine_hash]["small"] = {
    "unet_lr": 0.7,
    "text_encoder_lr": 0.35,
    "noise_offset": 0.045,
    "min_snr_gamma": 5.0
}

style_json = build_config(style_models)

with open('person_config.json', 'w') as f:
    json.dump(person_json, f, indent=4)

with open('style_config.json', 'w') as f:
    json.dump(style_json, f, indent=4)
