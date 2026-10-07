from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from legal_ita.config import OPENAI_GPT6_MAX_OUTPUT_TOKENS
from legal_ita.modeling.query import (
    default_openai_response_kwargs,
    model_request_kwargs_for_summary,
    query_openai_with_metrics,
)
from legal_ita.modeling.request_config import openai_completion_kwargs, openai_response_kwargs


class OpenAIRequestConfigTest(unittest.TestCase):
    def test_gpt6_luna_uses_responses_api_with_high_effort(self) -> None:
        kwargs = default_openai_response_kwargs("gpt-6-luna", "quesito")

        self.assertEqual(kwargs["model"], "gpt-6-luna")
        self.assertEqual(kwargs["input"], [{"role": "user", "content": "quesito"}])
        self.assertEqual(
            kwargs["max_output_tokens"],
            OPENAI_GPT6_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(kwargs["reasoning"], {"effort": "high"})

    def test_gpt6_chat_fallback_uses_highest_supported_chat_effort(self) -> None:
        kwargs = openai_completion_kwargs("  GPT-6-LUNA  ", "quesito", 123)

        self.assertEqual(kwargs["max_completion_tokens"], 123)
        self.assertEqual(kwargs["reasoning_effort"], "xhigh")
        self.assertNotIn("max_tokens", kwargs)

    def test_responses_helper_rejects_non_gpt6_models(self) -> None:
        with self.assertRaises(ValueError):
            openai_response_kwargs("gpt-5.5", "quesito", 123)

    def test_existing_reasoning_models_keep_high_effort(self) -> None:
        kwargs = openai_completion_kwargs("gpt-5.5", "quesito", 456)

        self.assertEqual(kwargs["max_completion_tokens"], 456)
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_legacy_chat_models_keep_max_tokens(self) -> None:
        kwargs = openai_completion_kwargs("gpt-4o", "quesito", 789)

        self.assertEqual(kwargs["max_tokens"], 789)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)

    @patch("legal_ita.modeling.query.openai.OpenAI")
    def test_gpt6_query_routes_to_responses_api(self, openai_factory: MagicMock) -> None:
        client = openai_factory.return_value
        client.responses.create.return_value = SimpleNamespace(
            output_text="risposta",
            status="completed",
            incomplete_details=None,
            usage={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 12},
            },
        )

        result = query_openai_with_metrics("gpt-6-luna", "quesito")

        self.assertEqual(result.text, "risposta")
        client.responses.create.assert_called_once_with(
            **default_openai_response_kwargs("gpt-6-luna", "quesito")
        )
        client.chat.completions.create.assert_not_called()
        self.assertEqual(result.metrics["reasoning_tokens"], 12)

    def test_gpt6_summary_reports_responses_parameters(self) -> None:
        kwargs = model_request_kwargs_for_summary("gpt-6-luna")

        self.assertEqual(kwargs["reasoning"], {"effort": "high"})
        self.assertEqual(kwargs["max_output_tokens"], OPENAI_GPT6_MAX_OUTPUT_TOKENS)
        self.assertNotIn("max_completion_tokens", kwargs)


if __name__ == "__main__":
    unittest.main()
