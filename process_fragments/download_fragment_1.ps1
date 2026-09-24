$env:KAGGLE_API_TOKEN = "KGAT_36655f16e43ffa98af5020644d6cf970"
$comp = "vesuvius-challenge-ink-detection"
$frag = "train/1"

Write-Host "Downloading metadata for $frag..."
kaggle competitions download -c $comp -f "$frag/mask.png" -p .
kaggle competitions download -c $comp -f "$frag/ir.png" -p .
kaggle competitions download -c $comp -f "$frag/inklabels.png" -p .

Write-Host "Downloading surface volume slices (65 files)..."
for ($i = 0; $i -lt 65; $i++) {
    $fileNum = $i.ToString("00")
    $filePath = "$frag/surface_volume/$fileNum.tif"
    Write-Host "Downloading $filePath..."
    kaggle competitions download -c $comp -f $filePath -p .
}
