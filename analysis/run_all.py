"""Run the analysis scripts 01-10 in order (about three minutes in all). Outputs go to out/."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in sorted(HERE.glob("[01][0-9]_*.py")):
    start = time.time()
    print(f"==== {script.name}", flush=True)
    subprocess.run([sys.executable, script.name], cwd=HERE, check=True)
    print(f"     ({time.time() - start:.0f} s)\n", flush=True)
