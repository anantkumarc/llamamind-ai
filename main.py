from __future__ import annotations

try:
    from llamamind_ai.app import LlamaMindChat
    from llamamind_ai.client import create_client, resolve_model_name
except ModuleNotFoundError as exc:
    if exc.name not in {"groq", "dotenv", "rich"}:
        raise
    print(
        f"Missing dependency: {exc.name}. Install project dependencies with:\n"
        "  python -m pip install -r requirements.txt"
    )
    raise SystemExit(1) from exc


def main() -> None:
    try:
        client = create_client()
        model_name = resolve_model_name(client)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        print("Set GROQ_API_KEY and ensure the model is available in your Groq account.")
        raise SystemExit(1) from exc

    app = LlamaMindChat(client=client, model_name=model_name)
    app.run()


if __name__ == "__main__":
    main()
