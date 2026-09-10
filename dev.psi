# ============================================================================
# dev.ps1  —  one command to start working on this project
#
# WHAT IT DOES
#   1. Activates the Python virtual environment in this folder
#   2. Prints a short status line so you can see you are in the right place
#
# WHY IT EXISTS
#   Typing ".\.venv\Scripts\Activate.ps1" every time is annoying, and forgetting
#   to do it is the single most common cause of "ModuleNotFoundError: No module
#   named 'torch'" — because the packages live INSIDE the venv, not in your
#   global Python.
#
# HOW TO RUN IT  (from the project folder, in PowerShell)
#   .\dev.ps1
#
# IF POWERSHELL REFUSES TO RUN IT
#   You will see: "...cannot be loaded because running scripts is disabled..."
#   That is Windows' execution policy, not a bug. Fix it for YOUR USER ONLY
#   (this does not need admin rights and does not weaken the whole machine):
#       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#   Then run ".\dev.ps1" again.
#   To undo it later:  Set-ExecutionPolicy -Scope CurrentUser Undefined
#
# WHAT YOU SHOULD SEE
#   (venv) PS C:\...\Question_Answering_System> .\dev.ps1
#   venv ready  |  Python 3.12.x  |  C:\...\Question_Answering_System
#   installed packages: 14
# ============================================================================

# --- 1. Find the venv folder and make sure it exists ------------------------
$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host "ERROR: no virtual environment found at $PSScriptRoot\.venv" -ForegroundColor Red
    Write-Host "Create one first with:  python -m venv .venv" -ForegroundColor Yellow
    return
}

# --- 2. Activate it ---------------------------------------------------------
# "&" is PowerShell's call operator: it runs the script at the path we built.
& $venvActivate

# --- 3. Print a status line so you can SEE that it worked -------------------
# $PSScriptRoot is the folder this .ps1 file lives in — i.e. the project root.
$py   = (& python --version)
$count = (& pip list --format=freeze 2>$null | Measure-Object).Count

Write-Host "venv ready  |  $py  |  $PSScriptRoot" -ForegroundColor Green
Write-Host "installed packages: $count" -ForegroundColor DarkGray
