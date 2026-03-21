from __future__ import annotations

import unittest

from backend.config import Settings
from backend.judge_service import LLMJudgeService


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletionsAPI:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error

    def create(self, **_: object) -> _FakeCompletion:
        if self.error is not None:
            raise self.error
        return _FakeCompletion(self.content or "{}")


class _FakeChatAPI:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.completions = _FakeCompletionsAPI(content=content, error=error)


class _FakeOpenAIClient:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.chat = _FakeChatAPI(content=content, error=error)


class JudgeServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LLMJudgeService(Settings(openai_api_key="test-key"))
        self.draft_answer = "Draft answer"

    def _run_with_content(self, content: str) -> str:
        self.service._client = _FakeOpenAIClient(content=content)
        return self.service.review_and_revise_answer(
            question="How does RAG work?",
            draft_answer=self.draft_answer,
            sources=[],
        )

    def test_returns_revised_answer_for_numeric_scores(self) -> None:
        with self.assertNoLogs("backend.judge_service", level="WARNING"):
            answer = self._run_with_content(
                """
                {
                  "groundedness": 5,
                  "citation_quality": 4,
                  "completeness": 5,
                  "insufficiency_handling": 4,
                  "needs_revision": false,
                  "rationale": "Looks good",
                  "final_answer": "Revised answer"
                }
                """
            )

        self.assertEqual(answer, "Revised answer")

    def test_normalizes_qualitative_scores(self) -> None:
        with self.assertNoLogs("backend.judge_service", level="WARNING"):
            answer = self._run_with_content(
                """
                {
                  "groundedness": "high",
                  "citation_quality": "strong",
                  "completeness": "acceptable",
                  "insufficiency_handling": "adequate",
                  "final_answer": "Revised answer"
                }
                """
            )

        self.assertEqual(answer, "Revised answer")

    def test_accepts_partial_json_with_only_final_answer(self) -> None:
        with self.assertNoLogs("backend.judge_service", level="WARNING"):
            answer = self._run_with_content('{"final_answer": "Revised answer"}')

        self.assertEqual(answer, "Revised answer")

    def test_malformed_json_falls_back_to_draft(self) -> None:
        with self.assertLogs("backend.judge_service", level="WARNING") as logs:
            answer = self._run_with_content('{"final_answer": ')

        self.assertEqual(answer, self.draft_answer)
        self.assertTrue(any("JSON parse failure" in entry for entry in logs.output))

    def test_empty_final_answer_falls_back_to_draft(self) -> None:
        with self.assertLogs("backend.judge_service", level="WARNING") as logs:
            answer = self._run_with_content('{"final_answer": ""}')

        self.assertEqual(answer, self.draft_answer)
        self.assertTrue(any("empty final_answer" in entry for entry in logs.output))

    def test_api_failure_falls_back_to_draft(self) -> None:
        self.service._client = _FakeOpenAIClient(error=RuntimeError("boom"))

        with self.assertLogs("backend.judge_service", level="WARNING") as logs:
            answer = self.service.review_and_revise_answer(
                question="How does RAG work?",
                draft_answer=self.draft_answer,
                sources=[],
            )

        self.assertEqual(answer, self.draft_answer)
        self.assertTrue(any("API failure" in entry for entry in logs.output))

    def test_unknown_diagnostic_value_is_ignored(self) -> None:
        with self.assertLogs("backend.judge_service", level="WARNING") as logs:
            answer = self._run_with_content(
                """
                {
                  "groundedness": "unknown-ish",
                  "final_answer": "Revised answer"
                }
                """
            )

        self.assertEqual(answer, "Revised answer")
        self.assertTrue(any("diagnostic normalization fallback" in entry for entry in logs.output))


if __name__ == "__main__":
    unittest.main()
