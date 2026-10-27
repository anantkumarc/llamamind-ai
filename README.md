# LlamaMind AI

A cloud-powered terminal chatbot that uses Groq-hosted models and a lightweight, ChatGPT-like interface.

## Features

- Terminal-based chat interface with Rich styling
- Cloud inference via Groq SDK
- Model auto-discovery with graceful fallback behavior
- Conversation history for multi-turn chats
- Structured answers rendered with Markdown headings, lists, and code blocks
- Environment-based secret management

## Setup

### Run locally on macOS

Clone the project onto your Mac and install it in a virtual environment:

```bash
git clone https://github.com/anantkumarc/llamamind-ai.git
cd llamamind-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The `GROQ_API_KEY` Codespaces secret does not automatically exist on your Mac.
Create a local `.env` file and add your key there:

```bash
cp .env.example .env
nano .env
```

Replace the placeholder with your Groq API key, save, and exit. Keep `.env`
private; it is excluded from Git. Never paste the key into a command or commit it.

Start the chat from the activated virtual environment:

```bash
Friday
```

You can also run `.venv/bin/Friday` without activating the environment, or use
`.venv/bin/python main.py`.

## Response behavior

LlamaMind applies one system instruction to every conversation. Responses are expected to lead with the direct answer, use Markdown structure only when it improves readability, preserve relevant conversation context, distinguish uncertainty from established facts, and provide concise but complete technical guidance. API and stream interruptions are reported in the terminal and do not get added to conversation history.

## Testing

Run the mocked unit tests without making an API request:

```bash
python -m unittest discover -v
```

## Model handling

The app no longer assumes a fixed Groq model name. It tries the configured `GROQ_MODEL` value first, then checks the current Groq model catalog and picks the first compatible Llama or GPT-OSS option. If no compatible model is available, the app exits with a clear error instead of repeatedly retrying a dead model ID.

## Notes

- Never commit your `.env` file or Groq key.
- `GROQ_API_KEY` should be stored in the environment or a local `.env` file only.
