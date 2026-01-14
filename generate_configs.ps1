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
        
        # Apply specific optimizations
        if ($isPerson -and $m -eq "cagliostrolab/animagine-xl-4.0") {
            # EXTREME DAMPING for Rank 96 stability
            $entry.small.unet_lr = 0.3
            $entry.small.text_encoder_lr = 0.3
            $entry.small.noise_offset = 0.045
            $entry.small.min_snr_gamma = 5.0
            $entry.small.optimizer_args = @('decouple=True', 'd_coef=0.5', 'weight_decay=0.01', 'use_bias_correction=True', 'safeguard_warmup=True')
        }
        
        $data[$hash] = $entry
    }

    $json = @{
        default = @{
            unet_lr = 1.0
            text_encoder_lr = 1.0
            noise_offset = 0.0357
            min_snr_gamma = 5.0
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

Write-Host "Registries fixed with identical LRs for Prodigy compatibility."
