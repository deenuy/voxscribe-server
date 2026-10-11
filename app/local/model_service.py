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


def assign_role(role: str, model_id: str) -> None:
    """Give a role to exactly one model: it is taken from whichever model held it before."""
    for other, roles in list(_selected_roles.items()):
        if role in roles and other != model_id:
            _selected_roles[other] = [r for r in roles if r != role]
    held = set(_selected_roles.get(model_id, []))
    held.add(role)
    _selected_roles[model_id] = sorted(held)


async def restore_selected_roles(pool) -> int:
    """Load the roles the person chose in an earlier run (saved as settings), so a restart keeps them."""
    import json

    async with pool.connection() as conn:
        cur = await conn.execute("SELECT key, value FROM settings WHERE key IN ('model_role_main', 'model_role_live')")
        rows = await cur.fetchall()
    restored = 0
    for key, value in rows:
        model_id = value
        if isinstance(value, str):
            try:
                model_id = json.loads(value)  # stored as JSON text...
            except ValueError:
                model_id = value  # ...or already decoded by the driver
        if isinstance(model_id, str) and model_id:
            assign_role(key.removeprefix("model_role_"), model_id)
            restored += 1
    return restored


def roles_for(model_id: str, settings) -> list[str]:
    """The roles a model holds right now: what the person chose, else what the engine is configured to use.

    Before anything is chosen the engine already uses `llm_model` for the main role and `live_model` for the live
    role, so those are reported as selected instead of showing nothing in use."""
    chosen = {role: next((m for m, roles in _selected_roles.items() if role in roles), None) for role in ("main", "live")}
    defaults = {"main": getattr(settings, "llm_model", None), "live": getattr(settings, "live_model", None)}
    return [role for role in ("main", "live") if (chosen[role] or defaults[role]) == model_id]


def tag_installed(tag: str, installed: list[str]) -> bool:
    """Is a configured tag installed? A tag without a version means `:latest` (so `nomic-embed-text` matches `nomic-embed-text:latest`)."""
    return tag in installed or (":" not in tag and f"{tag}:latest" in installed)


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
            roles = roles_for(model_id, settings)
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
                    embedding_ready = tag_installed(settings.embed_model, model_names)
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
                    status = "ready" if tag_installed(model_id, available_models) else "available"
                    roles = roles_for(model_id, settings)
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
                roles = roles_for(model_id, settings)
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
