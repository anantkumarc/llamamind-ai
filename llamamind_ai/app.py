from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

SYSTEM_PROMPT = """You are LlamaMind AI, a helpful assistant.

Present answers in a clear, useful structure:
- Start with the direct answer or main takeaway.
- For explanations or multi-part answers, use brief descriptive Markdown headings and concise bullets or numbered steps.
- Use fenced code blocks with a language label for code, and explain important changes or usage.
- Keep simple answers short; do not add headings or sections when they would be unnecessary.
- Follow any format the user explicitly requests.
- Be accurate, distinguish facts from uncertainty, and ask a focused question when essential context is missing."""


class LlamaMindChat:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self.console = Console()
        self.history: list[dict[str, str]] = []

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in self.history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _stream_response(self, user_message: str) -> str:
        messages = self._build_messages(user_message)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        full_text: list[str] = []
        with Live(
            Markdown(""),
            console=self.console,
            refresh_per_second=12,
            vertical_overflow="visible",
        ) as live:
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    full_text.append(delta)
                    live.update(Markdown("".join(full_text)))
        return "".join(full_text)

    def run(self) -> None:
        self.console.print(
            Panel.fit(
                "[bold cyan]LlamaMind AI[/bold cyan]\n"
                "[bold]Cloud-powered terminal assistant[/bold]\n"
                f"Model: {self.model_name}",
                title="LLAMAMIND AI",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("You")
            except EOFError:
                self.console.print("\nInput stream closed. Goodbye!", style="bold yellow")
                break
            if user_input.strip().lower() in {"/exit", "/quit", "quit", "exit"}:
                self.console.print("Goodbye!", style="bold yellow")
                break
            if user_input.strip().lower() == "/model":
                self.console.print(f"Using model: {self.model_name}", style="bold green")
                continue

            self.console.print("LlamaMind AI > ", end="")
            answer = self._stream_response(user_input)
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": answer})

            if len(self.history) > 12:
                self.history = self.history[-12:]
