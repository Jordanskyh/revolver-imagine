$models = @(
    'stabilityai/stable-diffusion-xl-base-1.0',
    'Lykon/dreamshaper-xl-1-0',
    'Lykon/art-diffusion-xl-0.9',
    'SG161222/RealVisXL_V4.0',
    'stablediffusionapi/protovision-xl-v6.6',
    'stablediffusionapi/omnium-sdxl',
    'GraydientPlatformAPI/realism-engine2-xl',
    'GraydientPlatformAPI/albedobase2-xl',
    'KBlueLeaf/Kohaku-XL-Zeta',
    'John6666/hassaku-xl-illustrious-v10style-sdxl',
    'John6666/nova-anime-xl-pony-v5-sdxl',
    'cagliostrolab/animagine-xl-4.0',
    'dataautogpt3/CALAMITY',
    'dataautogpt3/ProteusSigma',
    'dataautogpt3/ProteusV0.5',
    'dataautogpt3/TempestV0.1',
    'ehristoforu/Visionix-alpha',
    'femboysLover/RealisticStockPhoto-fp16',
    'fluently/Fluently-XL-Final',
    'mann-e/Mann-E_Dreams',
    'misri/leosamsHelloworldXL_helloworldXL70',
    'misri/zavychromaxl_v90',
    'openart-custom/DynaVisionXL',
    'recoilme/colorfulxl',
    'zenless-lab/sdxl-aam-xl-anime-mix',
    'zenless-lab/sdxl-anima-pencil-xl-v5',
    'zenless-lab/sdxl-anything-xl',
    'zenless-lab/sdxl-blue-pencil-xl-v7',
    'Corcelio/mobius',
    'GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16',
    'OnomaAIResearch/Illustrious-xl-early-release-v0',
    'bghira/terminus-xl-velocity-v2',
    'ifmain/UltraReal_Fine-Tune'
)

foreach ($m in $models) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($m)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha256.ComputeHash($bytes)
    $hashString = [System.BitConverter]::ToString($hash).Replace('-', '').ToLower()
    Write-Output "$m : $hashString"
}
