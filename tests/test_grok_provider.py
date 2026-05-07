from types import SimpleNamespace
import unittest

from backend.providers.grok_provider import (
    messages_to_openai,
    response_message_to_blocks,
    usage_cost_usd,
)


class GrokProviderTranslationTests(unittest.TestCase):
    def test_initial_user_image_translates_to_image_url(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "media_type": "image/jpeg",
                            "data": "abc123",
                        },
                    },
                    {"type": "text", "text": "look around"},
                ],
            }
        ]

        translated = messages_to_openai(messages)

        self.assertEqual(translated[0]["role"], "user")
        self.assertEqual(translated[0]["content"][0]["type"], "image_url")
        self.assertEqual(
            translated[0]["content"][0]["image_url"]["url"],
            "data:image/jpeg;base64,abc123",
        )
        self.assertEqual(translated[0]["content"][1]["text"], "look around")

    def test_image_tool_result_splits_tool_text_and_followup_image(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "media_type": "image/jpeg",
                                    "data": "frame",
                                },
                            },
                            {"type": "text", "text": "Current camera view."},
                        ],
                    }
                ],
            }
        ]

        translated = messages_to_openai(messages)

        self.assertEqual(translated[0]["role"], "tool")
        self.assertEqual(translated[0]["tool_call_id"], "call_1")
        self.assertEqual(translated[0]["content"], "Current camera view.")
        self.assertEqual(translated[1]["role"], "user")
        self.assertEqual(translated[1]["content"][0]["type"], "image_url")

    def test_response_message_tool_calls_become_tool_use_blocks(self) -> None:
        message = SimpleNamespace(
            content="I will look.",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="look", arguments="{}"),
                )
            ],
        )

        blocks = response_message_to_blocks(message)

        self.assertEqual(blocks[0], {"type": "text", "text": "I will look."})
        self.assertEqual(
            blocks[1],
            {"type": "tool_use", "id": "call_1", "name": "look", "input": {}},
        )

    def test_usage_cost_ticks_convert_to_usd(self) -> None:
        usage = SimpleNamespace(cost_in_usd_ticks=25_000_000)

        self.assertEqual(usage_cost_usd(usage), 0.0025)


if __name__ == "__main__":
    unittest.main()
