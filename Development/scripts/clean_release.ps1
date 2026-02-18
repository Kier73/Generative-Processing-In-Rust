# vGPU Release Sanitization Script (Windows PowerShell)
# Run this script before packaging the repository for public release.

Write-Host "--- vGPU Release Sanitization ---" -ForegroundColor Cyan

# 1. Clean Rust Target
if (Test-Path "vgpu_rust") {
    Write-Host "Cleaning vgpu_rust target..."
    Set-Location vgpu_rust
    cargo clean
    Set-Location ..
}

# 2. Remove Build Directories
$dirs_to_remove = @("build", "__pycache__", ".vscode", ".gemini", "vgpu_rust/target")
foreach ($dir in $dirs_to_remove) {
    if (Test-Path $dir) {
        Write-Host "Removing $dir..."
        Remove-Item -Recurse -Force $dir
    }
}

# 3. Remove Binary Artifacts from Root
$bins_to_remove = @("vgpu_cuda.dll", "vgpu_cuda.lib", "vgpu_cuda.exp", "vgpu_rust.dll")
foreach ($bin in $bins_to_remove) {
    if (Test-Path $bin) {
        Write-Host "Removing binary artifact $bin..."
        Remove-Item -Force $bin
    }
}

Write-Host "Sanitization Complete. Repository is ready for packaging." -ForegroundColor Green
