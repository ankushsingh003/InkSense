$baseDir = "d:\ink_detection_fragments"
$targetDir = "d:\ink_detection_fragments\train\1\surface_volume"
mkdir $targetDir -Force | Out-Null

Write-Host "Unzipping surface volume slices..."
$slices = Get-ChildItem "$baseDir\*.tif.zip"
foreach ($zip in $slices) {
    Write-Host "Unzipping $($zip.Name)..."
    Expand-Archive -Path $zip.FullName -DestinationPath $targetDir -Force
    Remove-Item $zip.FullName
}

Write-Host "Unzipping metadata..."
$meta = Get-ChildItem "$baseDir\*.png.zip"
foreach ($zip in $meta) {
    Write-Host "Unzipping $($zip.Name)..."
    Expand-Archive -Path $zip.FullName -DestinationPath "$baseDir\train\1" -Force
    Remove-Item $zip.FullName
}

Write-Host "Cleanup complete."
