from groq import Groq

from .config import get_api_key, get_configured_model


def create_client() -> Groq:
    return Groq(api_key=get_api_key())


def resolve_model_name(client: Groq) -> str:
    candidate_names = []
    configured = get_configured_model()
    if configured:
        candidate_names.append(configured)

    candidate_names.extend(
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "openai/gpt-oss-120b",
        ]
    )

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for name in candidate_names:
        if name and name not in seen:
            ordered_candidates.append(name)
            seen.add(name)

    try:
        model_response = client.models.list()
    except Exception as exc:  # pragma: no cover - surfaced as user-facing runtime issue
        raise RuntimeError(
            "Unable to fetch Groq model list. Check your network connection and API key."
        ) from exc

    available_models = []
    for item in getattr(model_response, "data", []):
        model_id = getattr(item, "id", None) or str(item)
        available_models.append(model_id)

    for candidate in ordered_candidates:
        if candidate in available_models:
            return candidate

    excluded = ("guard", "whisper", "tts", "safeguard", "embed")
    for model_id in available_models:
        lowered = model_id.lower()
        if any(word in lowered for word in excluded):
            continue
        if "llama" in lowered or "gpt-oss" in lowered:
            return model_id

    raise RuntimeError(
        "No compatible Groq model is currently available. "
        "Set GROQ_MODEL to a valid model name or check the Groq model catalog. "
        f"Available models: {', '.join(available_models) if available_models else 'none'}"
    )