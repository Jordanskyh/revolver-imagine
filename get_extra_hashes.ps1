$models = @(
    'mikeyandfriends/PixelWave_FLUX.1-dev_03',
    'dataautogpt3/FLUX-MonochromeManga',
    'rayonlabs/FLUX.1-dev',
    'mhnakif/fluxunchained-dev',
    'gradients-io-tournaments/Qwen-Image',
    'gradients-io-tournaments/Qwen-Image-Jib-Mix',
    'gradients-io-tournaments/Z-Image-Turbo'
)

foreach ($m in $models) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($m)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha256.ComputeHash($bytes)
    $hashString = [System.BitConverter]::ToString($hash).Replace('-', '').ToLower()
    Write-Output "$m : $hashString"
}
