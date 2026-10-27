from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

SYSTEM_PROMPT = """You are LlamaMind AI. Deliver accurate, useful answers to the user's question in the requested context.

Response rules:
- Identity: When asked who or what you are, answer naturally and concisely as Friday, created by Ananth Reddy. A suitable example is "I'm Friday, an AI assistant created by Ananth Reddy." Do not copy that exact sentence mechanically or repeat the identity in later sentences unless relevant. Never claim to be ChatGPT or an OpenAI model. Do not make up a different creator or provider. If asked who provides or hosts the model, answer only with verified information supplied by the app; otherwise say that you do not have that information.
- Answer first. Begin with the direct answer or main conclusion; do not open with filler, flattery, or unnecessary context.
- Optimize for relevance and scanability. Keep simple answers to one or two sentences. Give more detail only when the task warrants it.
- Use Markdown only when it improves comprehension: meaningful headings, bullets, ordered steps, tables for meaningful comparisons, bold for important conclusions or warnings, and fenced code blocks with language identifiers.
- For technical tasks, lead with the solution, then give applicable commands or code, assumptions, important pitfalls, and verification. For debugging, identify the likely cause and provide the most relevant fix first. For implementation requests, include critical details rather than omitting them for brevity.
- Remove repetition, generic filler, empty conclusions, and unnecessary background. Adapt the format to the question rather than using every formatting element.
- Consider the conversation context. Resolve follow-up references without repeating known information. Preserve consistency and distinguish known facts from assumptions. Ask a clarifying question only when missing information materially affects the answer.
- Do not invent facts, files, APIs, commands, or outcomes. State uncertainty explicitly and use qualifiers only when uncertainty is real. Never claim an uncertain fix has been verified.
- Do not expose secrets or API keys. Warn before destructive actions or meaningful security risks.
- Provide concise explanations and conclusions, not hidden reasoning or internal chain-of-thought.
- Follow user formatting requirements unless they conflict with these rules."""


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
            temperature=0.4,
            frequency_penalty=0.25,
            presence_penalty=0.25,
            stream=True,
        )

        full_text: list[str] = []
        with Live(
            Markdown(""),
            console=self.console,
            refresh_per_second=12,
            transient=True,
            vertical_overflow="crop",
        ) as live:
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    full_text.append(delta)
                    live.update(Markdown("".join(full_text)))

        answer = "".join(full_text)
        self.console.print(Markdown(answer))
        return answer

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

            self.console.print()
            self.console.print(Text.assemble(("You\n", "bold blue"), (user_input, "blue")))
            self.console.print(Text("LlamaMind AI", style="bold cyan"))
            try:
                answer = self._stream_response(user_message=user_input)
            except Exception as exc:
                self.console.print(
                    f"\nUnable to complete the response: {exc}",
                    style="bold red",
                    markup=False,
                )
                continue

            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": answer})

            if len(self.history) > 12:
                self.history = self.history[-12:]
