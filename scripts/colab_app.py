"""Start Streamlit in the existing Colab environment and show its private proxy."""
import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

_process = None


def start(root='/content/medical-assistant', port=8501):
    global _process
    root = Path(root).resolve()
    if not (root/'models/adapter/adapter_model.safetensors').is_file():
        raise FileNotFoundError('Restaure models/adapter do ZIP antes de iniciar.')
    if _process is None or _process.poll() is not None:
        env = os.environ.copy()
        env.update(DEFAULT_MODEL='fine_tuned', LLM_QUANTIZED='1')
        (root/'logs').mkdir(exist_ok=True)
        with (root/'logs/streamlit.log').open('a', encoding='utf-8') as log:
            _process = subprocess.Popen([
                str(root/'.venv-colab/bin/python'), '-m', 'streamlit', 'run', 'app.py',
                '--server.address', '127.0.0.1', '--server.port', str(port),
                '--server.headless', 'true', '--browser.gatherUsageStats', 'false',
            ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        if _process.poll() is not None:
            raise RuntimeError((root/'logs/streamlit.log').read_text(encoding='utf-8')[-4000:])
        try:
            with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise TimeoutError('Streamlit não iniciou. Consulte logs/streamlit.log.')
    from google.colab import output
    output.serve_kernel_port_as_iframe(port, height=900)


def stop():
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        _process.wait(timeout=15)
    _process = None
