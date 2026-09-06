"""Isolate project packages from Colab's preinstalled notebook packages."""
import os
import subprocess
import sys
import venv
from pathlib import Path


def run(command, log_path=None):
    """Forward subprocess output through Python so Colab displays tracebacks."""
    log = None
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("w", encoding="utf-8")
    try:
        with subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if log:
                    log.write(line)
                    log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed (exit {code}). See output above" +
                               (f" and {log_path}." if log_path else "."))
    finally:
        if log:
            log.close()


def setup(root):
    root = Path(root).resolve()
    environment = root / ".venv-colab"
    # Colab may not provide ensurepip; the host pip can manage an empty venv.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([sys.executable, "-m", "pip", "--python", python, "install", "pip",
         "-r", root / "requirements-training.txt"], root / "logs/colab-install.log")
    run([python, "-m", "pip", "check"])
    run([python, "-c", "import sys, torch; from transformers import Trainer; "
         "import peft, bitsandbytes; print('Python:', sys.version); "
         "print('Torch:', torch.__version__); "
         "assert torch.cuda.is_available(), 'Activate a Colab GPU'; "
         "print('GPU:', torch.cuda.get_device_name(0)); print('Training imports OK')"])
    return str(python)
