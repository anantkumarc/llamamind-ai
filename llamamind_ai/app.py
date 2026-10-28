from __future__ import annotations

import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
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
- Web search: when you use live web results, say so briefly and name the sites you relied on, because the app does not show citations separately. If results are missing or conflicting, say that instead of guessing.
- Follow user formatting requirements unless they conflict with these rules.

Default answer shape (adapt it; skip sections that add nothing):
1. One or two sentences with the direct answer.
2. The details: short bullets or numbered steps, or a fenced code block with a language tag.
3. Only if useful: a short "Notes" line for pitfalls, assumptions, or how to verify.
For simple or conversational questions, reply in plain sentences with no headings or bullets."""


EXIT_COMMANDS = {"/exit", "/quit", "quit", "exit"}
MAX_PANEL_WIDTH = 100

COMMANDS = [
    ("/help", "Show this list of commands"),
    ("/model", "Show the model in use"),
    ("/web", "Turn live web search on or off (/web on, /web off)"),
    ("/clear", "Clear the screen and start a fresh conversation"),
    ("/exit", "Quit (Ctrl+C also works)"),
]


class LlamaMindChat:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self.console = Console()
        self.history: list[dict[str, str]] = []
        self.web_search = True

    # ---- request -----------------------------------------------------

    def _web_supported(self) -> bool:
        return "gpt-oss" in self.model_name.lower()

    def _web_enabled(self) -> bool:
        return self.web_search and self._web_supported()

    def _web_status_text(self) -> str:
        if not self._web_supported():
            return "unavailable (needs a GPT-OSS model)"
        return "on" if self._web_enabled() else "off"

    def _handle_web_command(self, argument: str) -> None:
        if not self._web_supported():
            self.console.print(
                "Web search needs a GPT-OSS model. Set GROQ_MODEL=openai/gpt-oss-120b.",
                style="yellow",
            )
        elif argument in {"on", "off"}:
            self.web_search = argument == "on"
            self.console.print(
                f"Web search: {self._web_status_text()}", style="bold green"
            )
        elif argument == "":
            self.web_search = not self.web_search
            self.console.print(
                f"Web search: {self._web_status_text()}", style="bold green"
            )
        else:
            self.console.print("Usage: /web, /web on, or /web off", style="yellow")
        self.console.print()

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in self.history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _generation_options(self) -> dict:
        options: dict = {"temperature": 0.3, "max_tokens": 2048}
        if self._web_supported():
            if self._web_enabled():
                options["tools"] = [{"type": "browser_search"}]
                options["reasoning_effort"] = "low"
            else:
                options["reasoning_effort"] = "medium"
        return options

    # ---- rendering ---------------------------------------------------

    def _panel_width(self) -> int:
        return min(self.console.width, MAX_PANEL_WIDTH)

    def _answer_panel(self, text: str, elapsed: float | None = None) -> Panel:
        subtitle = Text(self.model_name, style="dim")
        if elapsed is not None:
            subtitle.append(f" \u00b7 {elapsed:.1f}s")
        return Panel(
            Markdown(text),
            title=Text("LlamaMind AI", style="bold cyan"),
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._panel_width(),
        )

    def _print_banner(self) -> None:
        details = Group(
            Text("LlamaMind AI", style="bold cyan"),
            Text("Cloud-powered terminal assistant", style="dim"),
            Text(),
            Text.assemble(("Model  ", "dim"), (self.model_name, "green")),
            Text.assemble(("Web    ", "dim"), (self._web_status_text(), "green")),
            Text.assemble(
                ("Tips   ", "dim"),
                ("/help", "bold"),
                (" for commands \u00b7 ", "dim"),
                ("/exit", "bold"),
                (" to quit", "dim"),
            ),
        )
        self.console.print(
            Panel(
                details,
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
                width=self._panel_width(),
            )
        )
        self.console.print()

    def _print_help(self) -> None:
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="dim")
        for command, description in COMMANDS:
            table.add_row(command, description)
        self.console.print(
            Panel(
                table,
                title="Commands",
                title_align="left",
                border_style="blue",
                box=box.ROUNDED,
                width=self._panel_width(),
            )
        )
        self.console.print()

    # ---- streaming ---------------------------------------------------

    def _stream_response(self, user_message: str) -> str:
        messages = self._build_messages(user_message)
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **self._generation_options(),
        )

        full_text: list[str] = []
        label = "Searching the web..." if self._web_enabled() else "Thinking..."
        status = self.console.status(f"[cyan]{label}[/cyan]", spinner="dots")
        status.start()
        live = None
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0].delta, "content", None)
                if not delta:
                    continue
                if live is None:
                    # Swap the spinner for the live answer on the first token.
                    status.stop()
                    live = Live(
                        self._answer_panel(""),
                        console=self.console,
                        refresh_per_second=12,
                        transient=True,
                        vertical_overflow="crop",
                    )
                    live.start()
                full_text.append(delta)
                live.update(self._answer_panel("".join(full_text)))
        finally:
            status.stop()
            if live is not None:
                live.stop()

        answer = "".join(full_text)
        if not answer.strip():
            raise RuntimeError("The model returned an empty response.")

        elapsed = time.perf_counter() - started
        self.console.print(self._answer_panel(answer, elapsed))
        self.console.print()
        return answer

    # ---- main loop ---------------------------------------------------

    def run(self) -> None:
        self._print_banner()

        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]", console=self.console)
            except EOFError:
                self.console.print("\nInput stream closed. Goodbye!", style="bold yellow")
                break
            except KeyboardInterrupt:
                self.console.print("\nGoodbye!", style="bold yellow")
                break

            text = user_input.strip()
            command = text.lower()
            if not text:
                continue
            if command in EXIT_COMMANDS:
                self.console.print("Goodbye!", style="bold yellow")
                break
            if command == "/help":
                self._print_help()
                continue
            if command == "/model":
                self.console.print(f"Using model: {self.model_name}", style="bold green")
                self.console.print(
                    f"Web search: {self._web_status_text()}", style="green"
                )
                self.console.print()
                continue
            if command == "/web" or command.startswith("/web "):
                self._handle_web_command(command[len("/web"):].strip())
                continue
            if command == "/clear":
                self.history = []
                self.console.clear()
                self._print_banner()
                continue
            if text.startswith("/") and " " not in text:
                self.console.print(
                    f"Unknown command {text}. Type /help to see what is available.",
                    style="yellow",
                    markup=False,
                )
                self.console.print()
                continue

            self.console.print()
            try:
                answer = self._stream_response(user_message=text)
            except KeyboardInterrupt:
                self.console.print("\nResponse cancelled.", style="yellow")
                self.console.print()
                continue
            except Exception as exc:
                self.console.print(
                    f"\nUnable to complete the response: {exc}",
                    style="bold red",
                    markup=False,
                )
                self.console.print()
                continue

            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": answer})

            if len(self.history) > 12:
                self.history = self.history[-12:]