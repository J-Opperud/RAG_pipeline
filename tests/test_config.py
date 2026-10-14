# tests/test_config.py
import pytest
from pydantic import ValidationError

from config import Settings

from pathlib import Path

def test_settings_use_defaults():
    settings = Settings(_env_file=None)

    assert settings.MODEL_NAME == "llama3.2:latest"
    assert settings.CHROMA_PATH == Path("chroma_db")
    assert settings.MAX_RESULTS == 3
    assert settings.CONFIDENCE_THRESHOLD == 1.0
    assert settings.DEBUG is False


def test_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "qwen2.5:latest")
    monkeypatch.setenv("MAX_RESULTS", "5")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.MODEL_NAME == "qwen2.5:latest"
    assert settings.MAX_RESULTS == 5
    assert settings.DEBUG is True


@pytest.mark.parametrize("value", ["0", "-1"])
def test_max_results_must_be_positive(monkeypatch, value):
    monkeypatch.setenv("MAX_RESULTS", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_model_name_cannot_be_empty(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "   ")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_ollama_url_requires_http_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "ollama:11434")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)