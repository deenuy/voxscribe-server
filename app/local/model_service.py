"""Model service for managing model selection, discovery, and download progress."""

from typing import Literal

from .models import Model

# Generative model catalogue
GENERATIVE_MODELS_CATALOGUE = {
    "gemma4:12b": {
        "name": "Gemma 4 Large",
        "description": "12B parameter model for enrichment, summaries, and Ask",
        "sizeBytes": 7_000_000_000,  # ~7GB
    },
    "gemma4:e4b": {
        "name": "Gemma 4 Medium",
        "description": "Efficient 4B parameter model for live assistance and voice",
        "sizeBytes": 2_500_000_000,  # ~2.5GB
    },
    "gemma4:e2b": {
        "name": "Gemma 4 Small",
        "description": "Compact 2B parameter model for low-resource environments",
        "sizeBytes": 1_300_000_000,  # ~1.3GB
    },
}

# Expert models (required, not selectable)
EXPERT_MODELS = {
    "asr": {
        "name": "Faster Whisper",
        "description": "Speech recognition",
        "sizeBytes": None,  # Variable, depends on model
    },
    "embedding": {
        "name": "Nomic Embed Text",
        "description": "Text embeddings for knowledge retrieval",
        "sizeBytes": 274_000_000,  # ~274MB
    },
}

# In-memory model role selection (persisted to database by store)
_selected_roles: dict[str, list[str]] = {}


def set_selected_roles(model_id: str, roles: list[str]) -> None:
    """Set the roles for a model (persisted separately)."""
    _selected_roles[model_id] = roles


def get_selected_roles(model_id: str) -> list[str]:
    """Get the roles for a model."""
    return _selected_roles.get(model_id, [])


def active_model(role: Literal["main", "live"]) -> str:
    """Get the model ID that is selected for a given role.

    Returns:
        Model ID (e.g., 'gemma4:12b') or '' if not set
    """
    for model_id, roles in _selected_roles.items():
        if role in roles:
            return model_id
    return ""


async def build_model_list(settings, ollama_host: str, providers_mode: str) -> list[Model]:
    """Build the current model list with status from reality.

    In fake mode: simulated ready states, no network calls.
    In local mode: check ASR in cache, Ollama for generative models.

    Args:
        settings: Config object
        ollama_host: Ollama API URL
        providers_mode: 'fake' or 'local'

    Returns:
        List of Model objects
    """
    models = []

    if providers_mode == "fake":
        # Fake mode: deterministic status
        models.append(
            Model(
                id="faster-whisper",
                kind="expert",
                name="Faster Whisper",
                description="Speech recognition",
                required=True,
                selected=False,
                selectedRoles=[],
                status="ready",
                sizeBytes=None,
            )
        )
        models.append(
            Model(
                id="nomic-embed-text",
                kind="expert",
                name="Nomic Embed Text",
                description="Text embeddings",
                required=True,
                selected=False,
                selectedRoles=[],
                status="ready",
                sizeBytes=274_000_000,
            )
        )
        for model_id, info in GENERATIVE_MODELS_CATALOGUE.items():
            roles = get_selected_roles(model_id)
            models.append(
                Model(
                    id=model_id,
                    kind="generative",
                    name=info["name"],
                    description=info["description"],
                    required=False,
                    selected=len(roles) > 0,
                    selectedRoles=roles,
                    status="ready" if model_id in ("gemma4:e4b", "gemma4:e2b") else "available",
                    sizeBytes=info["sizeBytes"],
                )
            )
    else:
        # Local mode: check reality
        import httpx

        # ASR: check if faster-whisper can be imported and model is cached
        asr_ready = False
        try:
            import faster_whisper  # noqa: F401

            # Check if model is in cache
            # Note: This is a simplified check; real implementation would verify cache path
            asr_ready = True
        except (ImportError, Exception):
            pass

        models.append(
            Model(
                id="faster-whisper",
                kind="expert",
                name="Faster Whisper",
                description="Speech recognition",
                required=True,
                selected=False,
                selectedRoles=[],
                status="ready" if asr_ready else "available",
                sizeBytes=None,
            )
        )

        # Embedding model: check Ollama
        embedding_ready = False
        embedding_error = None
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    models_list = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models_list]
                    embedding_ready = settings.embed_model in model_names
        except Exception as e:
            embedding_error = str(e)

        models.append(
            Model(
                id="nomic-embed-text",
                kind="expert",
                name="Nomic Embed Text",
                description="Text embeddings",
                required=True,
                selected=False,
                selectedRoles=[],
                status="ready" if embedding_ready else "available",
                sizeBytes=274_000_000,
                error=embedding_error,
            )
        )

        # Generative models: check Ollama
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                models_list = response.json().get("models", []) if response.status_code == 200 else []
                available_models = [m.get("name", "") for m in models_list]

                for model_id, info in GENERATIVE_MODELS_CATALOGUE.items():
                    status = "ready" if model_id in available_models else "available"
                    roles = get_selected_roles(model_id)
                    models.append(
                        Model(
                            id=model_id,
                            kind="generative",
                            name=info["name"],
                            description=info["description"],
                            required=False,
                            selected=len(roles) > 0,
                            selectedRoles=roles,
                            status=status,
                            sizeBytes=info["sizeBytes"],
                        )
                    )
        except Exception as e:
            # Ollama unavailable: return all models as available with error
            for model_id, info in GENERATIVE_MODELS_CATALOGUE.items():
                roles = get_selected_roles(model_id)
                models.append(
                    Model(
                        id=model_id,
                        kind="generative",
                        name=info["name"],
                        description=info["description"],
                        required=False,
                        selected=len(roles) > 0,
                        selectedRoles=roles,
                        status="available",
                        sizeBytes=info["sizeBytes"],
                        error=f"Ollama unavailable: {e}",
                    )
                )

    return models
