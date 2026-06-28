# publish.ps1 — Publica ComfyUI-KQube en el Comfy Registry
# Uso: .\publish.ps1 [-Message "changelog opcional"] [-Token "tu-token"]
#
# Requisito: comfy-cli instalado (pip install comfy-cli)
#
# El token (PAT) se genera una vez en https://registry.comfy.org → Settings → Personal Access Tokens
# Opcion A: pasarlo cada vez con -Token
# Opcion B: guardarlo en variable de entorno COMFY_REGISTRY_TOKEN (mas comodo)

param(
    [string]$Message = "",
    [string]$Token = $env:COMFY_REGISTRY_TOKEN
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── Verificar token ──
if (-not $Token) {
    Write-Host "[ERROR] No se encontro COMFY_REGISTRY_TOKEN." -ForegroundColor Red
    Write-Host "  Opcion 1: setx COMFY_REGISTRY_TOKEN ""tu-pat-aqui""" -ForegroundColor Yellow
    Write-Host "  Opcion 2: .\publish.ps1 -Token ""tu-pat-aqui""" -ForegroundColor Yellow
    Write-Host "  Genera tu PAT en: https://registry.comfy.org → Settings → Personal Access Tokens" -ForegroundColor Yellow
    exit 1
}

# ── Leer version actual ──
$tomlPath = Join-Path $PSScriptRoot "pyproject.toml"
$toml = Get-Content $tomlPath -Raw
if ($toml -match 'version\s*=\s*"([^"]+)"') {
    $version = $matches[1]
} else {
    Write-Host "[ERROR] No se pudo leer la version de pyproject.toml" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ComfyUI-KQube v$version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ── Publicar ──
$args = @("node", "publish", "--token", $Token)
if ($Message) {
    $args += "--changelog"
    $args += $Message
}

Write-Host "[*] Publicando..." -ForegroundColor Yellow
& comfy @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Fallo la publicacion (codigo $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[OK] v$version publicada exitosamente." -ForegroundColor Green
