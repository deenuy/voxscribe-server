"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest

from app.local.config import Config


def test_config_defaults():
    """Test that defaults match .env.example."""
    cfg = Config.from_env()
    assert cfg.database_url == "postgresql://localhost:5432/ideascribe_dev"
    assert cfg.providers == "local"
    assert cfg.privacy_mode == "offline"
    assert cfg.log_level == "info"
    assert cfg.ollama_host == "http://127.0.0.1:11434"
    assert cfg.llm_model == "gemma4:12b"
    assert cfg.llm_temperature == 0.2
    assert cfg.llm_num_ctx == 8192
    assert cfg.llm_max_tokens == 1024
    assert cfg.llm_timeout_s == 120
    assert cfg.llm_keep_alive == "30m"
    assert cfg.live_model == "gemma4:e4b"
    assert cfg.live_temperature == 0.3
    assert cfg.live_max_tokens == 256
    assert cfg.live_timeout_s == 20
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.embed_dim == 768
    assert cfg.asr_backend == "faster_whisper"
    assert cfg.asr_model == "small.en"
    assert cfg.asr_language == "en"
    assert cfg.asr_device == "cpu"
    assert cfg.asr_compute_type == "int8"
    assert cfg.vad_endpoint_ms == 700
    assert cfg.retrieval_top_k == 6


def test_config_env_overrides_default():
    """Test that environment variables override defaults."""
    os.environ["VOXSCRIBE_PROVIDERS"] = "fake"
    os.environ["VOXSCRIBE_LOG_LEVEL"] = "debug"
    os.environ["VOXSCRIBE_LLM_MODEL"] = "custom:model"

    try:
        cfg = Config.from_env()
        assert cfg.providers == "fake"
        assert cfg.log_level == "debug"
        assert cfg.llm_model == "custom:model"
    finally:
        del os.environ["VOXSCRIBE_PROVIDERS"]
        del os.environ["VOXSCRIBE_LOG_LEVEL"]
        del os.environ["VOXSCRIBE_LLM_MODEL"]


def test_config_dotenv_overrides_default():
    """Test that dotenv file overrides defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("VOXSCRIBE_PROVIDERS=fake\n")
        f.write("VOXSCRIBE_LOG_LEVEL=warning\n")
        f.write("VOXSCRIBE_EMBED_DIM=384\n")
        f.flush()
        temp_path = f.name

    try:
        # Clear env so dotenv takes effect
        os.environ.pop("VOXSCRIBE_PROVIDERS", None)
        os.environ.pop("VOXSCRIBE_LOG_LEVEL", None)
        os.environ.pop("VOXSCRIBE_EMBED_DIM", None)

        cfg = Config.from_env(temp_path)
        assert cfg.providers == "fake"
        assert cfg.log_level == "warning"
        assert cfg.embed_dim == 384
    finally:
        Path(temp_path).unlink()


def test_config_env_overrides_dotenv():
    """Test that environment variables take precedence over dotenv."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("VOXSCRIBE_PROVIDERS=local\n")
        f.write("VOXSCRIBE_LOG_LEVEL=error\n")
        f.flush()
        temp_path = f.name

    try:
        os.environ["VOXSCRIBE_PROVIDERS"] = "fake"
        os.environ["VOXSCRIBE_LOG_LEVEL"] = "debug"

        try:
            cfg = Config.from_env(temp_path)
            # Env vars should win
            assert cfg.providers == "fake"
            assert cfg.log_level == "debug"
        finally:
            del os.environ["VOXSCRIBE_PROVIDERS"]
            del os.environ["VOXSCRIBE_LOG_LEVEL"]
    finally:
        Path(temp_path).unlink()


def test_config_voxscribe_env_file_env_var():
    """Test that VOXSCRIBE_ENV_FILE environment variable is used."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("VOXSCRIBE_PROVIDERS=fake\n")
        f.flush()
        temp_path = f.name

    try:
        os.environ["VOXSCRIBE_ENV_FILE"] = temp_path
        os.environ.pop("VOXSCRIBE_PROVIDERS", None)

        try:
            cfg = Config.from_env()
            assert cfg.providers == "fake"
        finally:
            del os.environ["VOXSCRIBE_ENV_FILE"]
    finally:
        Path(temp_path).unlink()


def test_config_google_keys_ignored():
    """Test that GOOGLE_* keys are ignored."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("GOOGLE_OAUTH_CLIENT_ID=secret123\n")
        f.write("GOOGLE_OAUTH_CLIENT_SECRET=anothersecret\n")
        f.write("VOXSCRIBE_PROVIDERS=fake\n")
        f.flush()
        temp_path = f.name

    try:
        cfg = Config.from_env(temp_path)
        # Google keys should be ignored, only VOXSCRIBE keys loaded
        assert cfg.providers == "fake"
        assert not hasattr(cfg, "google_oauth_client_id")
    finally:
        Path(temp_path).unlink()


def test_config_invalid_providers():
    """Test that invalid providers value causes exit."""
    os.environ["VOXSCRIBE_PROVIDERS"] = "invalid"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_PROVIDERS"]


def test_config_invalid_log_level():
    """Test that invalid log level causes exit."""
    os.environ["VOXSCRIBE_LOG_LEVEL"] = "invalid"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_LOG_LEVEL"]


def test_config_invalid_privacy_mode():
    """Test that invalid privacy mode causes exit."""
    os.environ["VOXSCRIBE_PRIVACY_MODE"] = "invalid"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_PRIVACY_MODE"]


def test_config_invalid_asr_backend():
    """Test that invalid ASR backend causes exit."""
    os.environ["VOXSCRIBE_ASR_BACKEND"] = "invalid"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_ASR_BACKEND"]


def test_config_invalid_embed_dim():
    """Test that non-positive embed_dim causes exit."""
    os.environ["VOXSCRIBE_EMBED_DIM"] = "0"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_EMBED_DIM"]


def test_config_invalid_int():
    """Test that non-integer values cause exit."""
    os.environ["VOXSCRIBE_LLM_NUM_CTX"] = "not_an_int"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_LLM_NUM_CTX"]


def test_config_invalid_float():
    """Test that non-float values cause exit."""
    os.environ["VOXSCRIBE_LLM_TEMPERATURE"] = "not_a_float"

    try:
        with pytest.raises(SystemExit) as exc_info:
            Config.from_env()
        assert exc_info.value.code == 2
    finally:
        del os.environ["VOXSCRIBE_LLM_TEMPERATURE"]


def test_config_integer_parsing():
    """Test integer field parsing."""
    os.environ["VOXSCRIBE_LLM_NUM_CTX"] = "4096"
    os.environ["VOXSCRIBE_RETRIEVAL_TOP_K"] = "10"

    try:
        cfg = Config.from_env()
        assert cfg.llm_num_ctx == 4096
        assert cfg.retrieval_top_k == 10
    finally:
        del os.environ["VOXSCRIBE_LLM_NUM_CTX"]
        del os.environ["VOXSCRIBE_RETRIEVAL_TOP_K"]


def test_config_float_parsing():
    """Test float field parsing."""
    os.environ["VOXSCRIBE_LLM_TEMPERATURE"] = "0.7"
    os.environ["VOXSCRIBE_LIVE_TEMPERATURE"] = "0.5"

    try:
        cfg = Config.from_env()
        assert cfg.llm_temperature == 0.7
        assert cfg.live_temperature == 0.5
    finally:
        del os.environ["VOXSCRIBE_LLM_TEMPERATURE"]
        del os.environ["VOXSCRIBE_LIVE_TEMPERATURE"]
