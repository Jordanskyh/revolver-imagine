function Get-Hash($m) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($m)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash($bytes)
    return [System.BitConverter]::ToString($hash).Replace('-', '').ToLower()
}

function Create-Registry-File($models, $filePath, $isPerson = $false) {
    $data = @{}
    foreach ($m in $models) {
        $hash = Get-Hash $m
        $entry = @{
            model_name = $m
            small = @{ unet_lr = $null; text_encoder_lr = $null; noise_offset = $null; min_snr_gamma = $null }
            medium = @{ unet_lr = $null; text_encoder_lr = $null; noise_offset = $null; min_snr_gamma = $null }
            large = @{ unet_lr = $null; text_encoder_lr = $null; noise_offset = $null; min_snr_gamma = $null }
        }
        
        # PURE OVERRIDES ONLY: Only include settings that DIFFER from TOML/Autoepoch defaults.
        
        # Custom Optimization for CALAMITY (Person) - WINNING STRATEGY
        # Goal: Widen win margin > 10% (Currently ~0.2%).
        # Strategy: Increase epochs to 50 to lower Text-Guided Loss while maintaining winning Style.
        if ($m -eq "dataautogpt3/CALAMITY") {
             Write-Host ">>> APPLYING CALAMITY WINNER STRATEGY (50 Epochs)" -ForegroundColor Green
             $entry.large.max_train_epochs = 50
             $entry.large.save_every_n_epochs = 10
             
             # Proven Winner Config
             $entry.large.optimizer_type = "prodigy"
             $entry.large.optimizer_args = @('decouple=True', 'd_coef=1', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
             $entry.large.noise_offset = 0.0357
             $entry.large.min_snr_gamma = 5
             $entry.large.unet_lr = 1.0
             $entry.large.text_encoder_lr = 1.0
        }
        
        # Custom Optimization for Animagine-XL (Person) - CHAMPION CONFIG
        # Proven Config: 160 Epochs + Prodigy + Noise Offset 0.0357
        if ($m -eq "cagliostrolab/animagine-xl-4.0") {
             # Note: Animagine by default inherits 160 epochs from 'person' defaults if not overridden.
             # But explicitly setting it ensures consistency.
             $entry.large.max_train_epochs = 160
             $entry.large.save_every_n_epochs = 20
             
             $entry.large.optimizer_type = "prodigy"
             $entry.large.optimizer_args = @('decouple=True', 'd_coef=1', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
             $entry.large.noise_offset = 0.0357
             $entry.large.min_snr_gamma = 5
             $entry.large.unet_lr = 1.0
             $entry.large.text_encoder_lr = 1.0
        }

        # Custom Optimization for Realistic Style - PURE OVERRIDE
        # TOML Style uses AdamW8bit | This model REQUIRES Prodigy + Custom Noise to win.
        if ($m -eq "femboysLover/RealisticStockPhoto-fp16") {
            $entry.large.optimizer_type = "prodigy"
            $entry.large.optimizer_args = @('decouple=True', 'd_coef=1.0', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
            $entry.large.noise_offset = 0.0303
            # Evidence: -10.45% Gap | Hypo: High variance (0.053) needs more steps to stabilize.
            $entry.large.max_train_epochs = 40
            $entry.large.save_every_n_epochs = 10
        }

        # All other models and Person tasks run on Natural Defaults.
        
        $data[$hash] = $entry
    }

    $json = @{
        default = @{
            unet_lr = $null
            text_encoder_lr = $null
            noise_offset = $null
            min_snr_gamma = $null
        }
        data = $data
    } | ConvertTo-Json -Depth 10
    
    $json | Out-File -FilePath $filePath -Encoding utf8
}

$sdxl_models = @(
    'stabilityai/stable-diffusion-xl-base-1.0', 'Lykon/dreamshaper-xl-1-0', 'Lykon/art-diffusion-xl-0.9',
    'SG161222/RealVisXL_V4.0', 'stablediffusionapi/protovision-xl-v6.6', 'stablediffusionapi/omnium-sdxl',
    'GraydientPlatformAPI/realism-engine2-xl', 'GraydientPlatformAPI/albedobase2-xl', 'KBlueLeaf/Kohaku-XL-Zeta',
    'John6666/hassaku-xl-illustrious-v10style-sdxl', 'John6666/nova-anime-xl-pony-v5-sdxl',
    'cagliostrolab/animagine-xl-4.0', 'dataautogpt3/CALAMITY', 'dataautogpt3/ProteusSigma',
    'dataautogpt3/ProteusV0.5', 'dataautogpt3/TempestV0.1', 'ehristoforu/Visionix-alpha',
    'fluently/Fluently-XL-Final', 'GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16', 'zenless-lab/sdxl-blue-pencil-xl-v7',
    'zenless-lab/sdxl-aam-xl-anime-mix', 'misri/leosamsHelloworldXL_helloworldXL70', 'misri/zavychromaxl_v90',
    'femboysLover/RealisticStockPhoto-fp16', 'bghira/terminus-xl-velocity-v2', 'recoilme/colorfulxl',
    'mann-e/Mann-E_Dreams', 'Corcelio/mobius', 'zenless-lab/sdxl-anima-pencil-xl-v5', 
    'OnomaAIResearch/Illustrious-xl-early-release-v0', 'zenless-lab/sdxl-anything-xl',
     'openart-custom/DynaVisionXL', 'ifmain/UltraReal_Fine-Tune'
)

$personRegistry = "C:\56\revolver - imagine\scripts\lrs\person_config.json"
$styleRegistry = "C:\56\revolver - imagine\scripts\lrs\style_config.json"

Create-Registry-File $sdxl_models $personRegistry $true
Create-Registry-File $sdxl_models $styleRegistry $false

Write-Host "Registries reset to TOML Defaults (Safe Mode)."
