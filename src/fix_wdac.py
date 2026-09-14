"""
Fix WDAC blocking: make sklearn not require pyarrow
Patches venv/lib/site-packages/sklearn/utils/fixes.py to make pyarrow optional
"""
import pathlib
import sys

# Find venv
venv_path = pathlib.Path(sys.executable).parent.parent / "lib" / "site-packages" / "sklearn" / "utils" / "fixes.py"
# Windows venv path is different: venv/Lib/site-packages
win_path = pathlib.Path(sys.executable).parent / "Lib" / "site-packages" / "sklearn" / "utils" / "fixes.py"

for p in [venv_path, win_path]:
    if p.exists():
        print(f"Found {p}")
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "import pyarrow" in text and "try:" not in text.split("import pyarrow")[0][-200:]:
            # Patch: wrap pyarrow import in try/except
            text = text.replace("import pyarrow", "try:\n    import pyarrow\nexcept ImportError:\n    pyarrow = None")
            p.write_text(text, encoding="utf-8")
            print(f"Patched {p} to make pyarrow optional")
        else:
            print("Already patched or not found")
        break
else:
    print("fixes.py not found, trying alternative search")
    import sklearn
    print(pathlib.Path(sklearn.__file__).parent / "utils" / "fixes.py")
