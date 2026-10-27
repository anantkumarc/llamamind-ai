from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from llamamind_ai.app import LlamaMindChat, SYSTEM_PROMPT


class FakeCompletions:
    def __init__(self, chunks=(), error: Exception | None = None):
        self.chunks = chunks
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return iter(self.chunks)


class FakeClient:
    def __init__(self, chunks=(), error: Exception | None = None):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(chunks=chunks, error=error)
        )


def chunk(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


class LlamaMindChatTests(unittest.TestCase):
    def make_chat(self, chunks=(), error: Exception | None = None):
        client = FakeClient(chunks=chunks, error=error)
        chat = LlamaMindChat(client=client, model_name="test-model")
        chat.console = Console(
            file=io.StringIO(), force_terminal=False, width=80
        )
        return chat

    def test_system_prompt_is_one_answer_first_instruction_set(self):
        messages = self.make_chat()._build_messages("Simple question")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIs(messages[0]["content"], SYSTEM_PROMPT)
        self.assertIn("Answer first", SYSTEM_PROMPT)
        self.assertIn("Do not invent facts", SYSTEM_PROMPT)
        self.assertIn("hidden reasoning", SYSTEM_PROMPT)
        self.assertIn("fenced code blocks", SYSTEM_PROMPT)
        self.assertIn("internal chain-of-thought", SYSTEM_PROMPT)
        self.assertIn("answer naturally and concisely", SYSTEM_PROMPT)
        self.assertIn("Do not copy that exact sentence mechanically", SYSTEM_PROMPT)
        self.assertIn("Never claim to be ChatGPT", SYSTEM_PROMPT)

    def test_stream_request_keeps_history_and_uses_markdown_stream(self):
        chat = self.make_chat(chunks=[chunk("## Result\n"), chunk("- direct")])
        chat.history = [
            {"role": "user", "content": "What did we compare?"},
            {"role": "assistant", "content": "PostgreSQL and SQLite."},
        ]

        with patch("llamamind_ai.app.Live") as live_class:
            live_class.return_value.__enter__.return_value = live_class
            answer = chat._stream_response("Which is more portable?")

        self.assertEqual(answer, "## Result\n- direct")
        request = chat.client.chat.completions.calls[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["temperature"], 0.4)
        self.assertEqual(request["frequency_penalty"], 0.25)
        self.assertEqual(request["presence_penalty"], 0.25)
        self.assertTrue(request["stream"])
        live_options = live_class.call_args.kwargs
        self.assertEqual(live_options["vertical_overflow"], "crop")
        self.assertTrue(live_options["transient"])
        self.assertEqual(chat.console.file.getvalue().count("Result"), 1)
        self.assertEqual(request["messages"][1:3], chat.history)
        self.assertEqual(request["messages"][3], {
            "role": "user",
            "content": "Which is more portable?",
        })

    def test_history_is_trimmed_to_last_twelve_messages(self):
        chat = self.make_chat(chunks=[chunk("Next answer")])
        chat.history = [
            {"role": role, "content": f"message {index}"}
            for index in range(12)
            for role in ["user", "assistant"]
        ]
        chat.history = chat.history[-12:]

        with patch("llamamind_ai.app.Prompt.ask", side_effect=["follow-up", "/quit"]):
            chat.run()

        self.assertEqual(len(chat.history), 12)
        self.assertEqual(chat.history[-2:], [
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "Next answer"},
        ])
        output = chat.console.file.getvalue()
        self.assertIn("You\nfollow-up", output)
        self.assertIn("LlamaMind AI", output)

    def test_interrupted_stream_is_reported_and_not_saved(self):
        chat = self.make_chat()
        original_stream = chat._stream_response

        def interrupted_stream(user_message: str):
            del user_message
            raise RuntimeError("connection interrupted")

        chat._stream_response = interrupted_stream
        with patch("llamamind_ai.app.Prompt.ask", side_effect=["hello", "/quit"]):
            chat.run()

        output = chat.console.file.getvalue()
        self.assertIn("Unable to complete the response: connection interrupted", output)
        self.assertEqual(chat.history, [])
        self.assertFalse(original_stream is interrupted_stream)


if __name__ == "__main__":
    unittest.main()
