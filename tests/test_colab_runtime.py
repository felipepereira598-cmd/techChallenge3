import sys
import pytest
from scripts.colab_runtime import run


def test_subprocess_error_is_visible_and_logged(tmp_path, capsys):
    log = tmp_path / "error.log"
    with pytest.raises(RuntimeError, match="exit 1"):
        run([sys.executable, "-c", "raise ValueError('original failure')"], log)
    assert "original failure" in capsys.readouterr().out
    assert "ValueError: original failure" in log.read_text(encoding="utf-8")


def test_subprocess_success(capsys):
    run([sys.executable, "-c", "print('visible output')"])
    assert "visible output" in capsys.readouterr().out
