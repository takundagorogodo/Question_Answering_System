<#
============================================================================
 setup_project.ps1  —  creates the project folder structure + .gitignore

 WHAT IT DOES
   1. Creates every folder this project needs
   2. Puts a ".gitkeep" file in each empty folder (see WHY below)
   3. Writes .gitignore
   4. Shows you the resulting tree

 WHY ".gitkeep"?
   Git does not track empty folders - only files. If a folder is empty, git
   ignores it, and anyone cloning your repo gets no folder structure at all.
   ".gitkeep" is a zero-byte placeholder file (the name is a convention, git
   has no special knowledge of it) whose only job is to give the folder
   something to contain. You can delete them later once real files exist.

 HOW TO RUN  (PowerShell, from the project folder)
   cd C:\Users\Prosperity\Documents\Question_Answering_System
   .\setup_project.ps1

 IF POWERSHELL REFUSES ("running scripts is disabled")
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   (current user only, no admin rights, undo with: ... Undefined)

 SAFE TO RUN TWICE
   -Force and -ItemType Directory overwrite/reuse rather than fail.
============================================================================
#>

# --- 0. Guard: make sure we are in the right place --------------------------
# $PSScriptRoot = the folder this script lives in.
# If .gitignore already exists we assume the structure was already created.
if (Test-Path (Join-Path $PSScriptRoot ".gitignore")) {
    Write-Host ".gitignore already exists - structure looks created already. Continuing anyway." -ForegroundColor Yellow
}

Set-Location $PSScriptRoot

# --- 1. The folder list -----------------------------------------------------
# Each entry is one folder. Read the README.md table to see what goes in each.
$folders = @(
    "src",                  # application code - one file per pipeline stage
    "baseline",             # the TF-IDF comparison baseline (Stage 18)
    "data\raw",             # original source documents
    "data\processed",       # cleaned chunks + metadata
    "data\index",           # the built FAISS index
    "models",               # downloaded model weights (git-ignored)
    "cache",                # Hugging Face cache (git-ignored)
    "evaluation",           # test set + evaluation scripts
    "evaluation\results",   # generated tables and charts
    "logs",                 # query log + feedback log
    "notebooks",            # exploratory work
    "docs\lessons",         # lesson notes
    "docs\adr",             # architecture decision records
    "docs\report"           # the final college report
)

foreach ($f in $folders) {
    New-Item -ItemType Directory -Path $f -Force | Out-Null
    # 2. Placeholder so git will actually commit the empty folder
    New-Item -ItemType File -Path (Join-Path $f ".gitkeep") -Force | Out-Null
    Write-Host "  created  $f" -ForegroundColor DarkGray
}

# --- 3. Write .gitignore ----------------------------------------------------
# '@ ... '@ is a "literal here-string": everything between the markers is taken
# exactly as written, with no variable expansion. That is why the $ signs and
# * wildcards below stay literal instead of being interpreted by PowerShell.
$gitignore = @'
# ---- Virtual environment (derived, rebuildable from requirements.txt) ------
venv/
.venv/

# ---- Python bytecode / build artefacts ------------------------------------
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/

# ---- Model weights and caches (large, re-downloadable) --------------------
models/
cache/
*.bin
*.safetensors
*.onnx

# ---- Built index (rebuilt from data/ by a script) -------------------------
data/index/*.faiss
data/index/*.index
data/index/*.pkl

# ---- Logs (may contain user questions) ------------------------------------
logs/*.jsonl
logs/*.log

# ---- Editors / OS ---------------------------------------------------------
.vscode/
.idea/
.DS_Store
Thumbs.db

# ---- Secrets --------------------------------------------------------------
.env
*.key
*.token
'@

Set-Content -Path ".gitignore" -Value $gitignore -Encoding utf8
Write-Host ""
Write-Host "  wrote    .gitignore" -ForegroundColor Green

# --- 4. Show the result -----------------------------------------------------
Write-Host ""
Write-Host "Project structure:" -ForegroundColor Cyan
Get-ChildItem -Recurse -Directory | ForEach-Object { "  " + $_.FullName.Replace($PSScriptRoot, ".") }

Write-Host ""
Write-Host "Done. Next: create README.md, then 'git init'." -ForegroundColor Green
