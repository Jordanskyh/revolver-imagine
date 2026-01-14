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
        
        # Custom Optimization for Realistic Style - MASTERCLASS PRODIGY
        if ($m -eq "femboysLover/RealisticStockPhoto-fp16") {
            # Target Score: < 0.0356 | Strategy: Full Power Prodigy
            $entry.large.unet_lr = 1.0
            $entry.large.text_encoder_lr = 1.0
            $entry.large.optimizer_type = "prodigy"
            $entry.large.optimizer_args = @('decouple=True', 'd_coef=1.0', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
            $entry.large.lr_scheduler = "constant"
            $entry.large.lr_warmup_steps = 0
            $entry.large.max_grad_norm = 1.0
            $entry.large.min_snr_gamma = 7.0
            $entry.large.noise_offset = 0.0303
            $entry.large.max_train_epochs = 40
            $entry.large.save_every_n_epochs = 10
        }

        # Custom Optimization for Visionix-alpha - TASK SPECIFIC LOGIC
        if ($m -eq "ehristoforu/Visionix-alpha") {
            if ($isPerson) {
                # Target: Recover from -25.01% | Strategy: Prodigy Masterclass (Identity Focus)
                $entry.small.unet_lr = 1.0
                $entry.small.text_encoder_lr = 1.0
                $entry.small.optimizer_type = "prodigy"
                $entry.small.optimizer_args = @('decouple=True', 'd_coef=1.0', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
                $entry.small.lr_scheduler = "constant"
                $entry.small.min_snr_gamma = 5.0
                $entry.small.noise_offset = 0.0357
                $entry.small.max_train_epochs = 100
                $entry.small.save_every_n_epochs = 10
            } else {
                # Winner: Task Style (13.20% Lead) | Strategy: AdamW8bit (Composition Focus)
                $entry.small.unet_lr = 0.0001
                $entry.small.optimizer_type = "adamw8bit"
                $entry.small.optimizer_args = @('weight_decay=0.01', 'betas=(0.9,0.999)', 'eps=1e-08')
                $entry.small.lr_scheduler = "constant"
                $entry.small.min_snr_gamma = 7.0
                $entry.small.noise_offset = 0.0411
                $entry.small.max_train_epochs = 25
                $entry.small.save_every_n_epochs = 5
            }
        }

        # Custom Optimization for Animagine-XL - DOMAIN: PERSON (Fix Underfitting)
        if ($isPerson -and $m -eq "cagliostrolab/animagine-xl-4.0") {
            # Target Score: < 0.0739 | Strategy: Full Power Prodigy
            $entry.small.unet_lr = 1.0
            $entry.small.text_encoder_lr = 1.0
            $entry.small.noise_offset = 0.0357
            $entry.small.min_snr_gamma = 7.0
            $entry.small.optimizer_args = @('decouple=True', 'd_coef=1.0', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
            # Consistent 160 epochs with higher d_coef to ensure convergence
            $entry.small.max_train_epochs = 160
            $entry.small.save_every_n_epochs = 20
        }
        
        $data[$hash] = $entry
    }

    $json = @{
        # Set default LR to null so it inherits from TOML (AdamW8bit friendly)
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
    'John6666/hassaku-xl-illustrious-v10style-sdxl', 'John6666/nova-anime-xl-pony-v5-sdxl', 'cagliostrolab/animagine-xl-4.0',
    'dataautogpt3/CALAMITY', 'dataautogpt3/ProteusSigma', 'dataautogpt3/ProteusV0.5', 'dataautogpt3/TempestV0.1',
    'ehristoforu/Visionix-alpha', 'femboysLover/RealisticStockPhoto-fp16', 'fluently/Fluently-XL-Final',
    'mann-e/Mann-E_Dreams', 'misri/leosamsHelloworldXL_helloworldXL70', 'misri/zavychromaxl_v90',
    'openart-custom/DynaVisionXL', 'recoilme/colorfulxl', 'zenless-lab/sdxl-aam-xl-anime-mix',
    'zenless-lab/sdxl-anima-pencil-xl-v5', 'zenless-lab/sdxl-anything-xl', 'zenless-lab/sdxl-blue-pencil-xl-v7',
    'Corcelio/mobius', 'GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16', 'OnomaAIResearch/Illustrious-xl-early-release-v0',
    'bghira/terminus-xl-velocity-v2', 'ifmain/UltraReal_Fine-Tune'
)

$flux_models = @(
    'mikeyandfriends/PixelWave_FLUX.1-dev_03', 'dataautogpt3/FLUX-MonochromeManga',
    'rayonlabs/FLUX.1-dev', 'mhnakif/fluxunchained-dev'
)

$qwen_models = @(
    'gradients-io-tournaments/Qwen-Image', 'gradients-io-tournaments/Qwen-Image-Jib-Mix'
)

$zimage_models = @(
    'gradients-io-tournaments/Z-Image-Turbo'
)

Create-Registry-File $sdxl_models "scripts/lrs/person_config.json" -isPerson $true
Create-Registry-File $sdxl_models "scripts/lrs/style_config.json"
Create-Registry-File $flux_models "scripts/lrs/flux.json"
Create-Registry-File $qwen_models "scripts/lrs/qwen.json"
Create-Registry-File $zimage_models "scripts/lrs/zimage.json"

Write-Host "Registries reset to TOML Defaults (Safe Mode)."
