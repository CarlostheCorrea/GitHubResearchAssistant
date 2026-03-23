from __future__ import annotations

import json
import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.models import SourceSnippet
from backend.utils import looks_like_placeholder_secret

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are the final quality gate for a repository-question-answering system.

You receive:
- the user's repository question
- a draft answer
- the cited repository evidence used to answer it

Your job:
1. Produce the best possible final answer for the user from the provided repository evidence.
2. If the draft answer has problems, rewrite it into a better final answer.
3. If the draft answer is already good, preserve it with only minimal edits.
4. Optional diagnostics are allowed, but they are secondary to returning a usable final answer.

Hard rules for the final answer:
- Use only the provided repository evidence.
- Do not invent files, functions, classes, behavior, or architecture.
- If the evidence is insufficient, say so clearly.
- Return the best final answer for the user. Do not mention judging, review, or internal evaluation.
- Write plain prose only, with no code fences, markdown bullets, numbered lists, or inline code formatting.
- Do not append bracketed file citations or line-range references in the answer body.
- `final_answer` must always be present and non-empty.
- Optional fields may include: `needs_revision`, `rationale`, `groundedness`, `citation_quality`, `completeness`, `insufficiency_handling`.
- Optional diagnostic fields may be numeric scores or qualitative labels such as "high", "adequate", or "weak".

Return strict JSON only.
"""


class JudgeReview(BaseModel):
    groundedness: int | None = Field(default=None, ge=1, le=5)
    citation_quality: int | None = Field(default=None, ge=1, le=5)
    completeness: int | None = Field(default=None, ge=1, le=5)
    insufficiency_handling: int | None = Field(default=None, ge=1, le=5)
    needs_revision: bool = False
    rationale: str = ""
    final_answer: str


class LLMJudgeService:
    _DIAGNOSTIC_LABELS = {
        "high": 5,
        "strong": 5,
        "medium": 3,
        "adequate": 3,
        "acceptable": 3,
        "low": 1,
        "weak": 1,
        "poor": 1,
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def review_and_revise_answer(
        self,
        question: str,
        draft_answer: str,
        sources: list[SourceSnippet],
    ) -> str:
        logger.info("Running LLM-as-a-Judge for question: %s", question)
        prompt = self._build_prompt(question, draft_answer, sources)
        try:
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:  # noqa: BLE001
            logger.warning("LLM judge API failure; falling back to draft answer.", exc_info=True)
            return draft_answer

        content = completion.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM judge JSON parse failure; falling back to draft answer. Payload: %s", content[:800], exc_info=True)
            return draft_answer

        try:
            normalized, used_normalization_fallback = self._normalize_payload(payload)
            review = JudgeReview.model_validate(normalized)
        except Exception:  # noqa: BLE001
            logger.warning("LLM judge payload validation failure; falling back to draft answer. Payload: %s", content[:800], exc_info=True)
            return draft_answer

        if used_normalization_fallback:
            logger.warning("LLM judge diagnostic normalization fallback used for one or more fields.")

        final_answer = review.final_answer.strip()
        if not final_answer:
            logger.warning("LLM judge returned empty final_answer; falling back to draft answer.")
            return draft_answer

        return final_answer

    def _build_prompt(
        self,
        question: str,
        draft_answer: str,
        sources: list[SourceSnippet],
    ) -> str:
        source_blocks = []
        for index, source in enumerate(sources, start=1):
            source_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"File: {source.file_path}",
                        f"Lines: {source.start_line or 'n/a'}-{source.end_line or 'n/a'}",
                        f"Chunk type: {source.chunk_type}",
                        f"Summary: {source.short_summary or 'n/a'}",
                        "Snippet:",
                        source.snippet,
                    ]
                )
            )

        return "\n\n".join(
            [
                f"Question: {question}",
                "Draft answer:",
                draft_answer,
                "",
                "Cited evidence:",
                "\n\n".join(source_blocks) if source_blocks else "No sources were returned.",
                "",
                "Return JSON only.",
                "Required key: final_answer.",
                "Optional keys: needs_revision, rationale, groundedness, citation_quality, completeness, insufficiency_handling.",
                "Optional diagnostic values may be numbers or qualitative labels.",
            ]
        )

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for LLM-as-a-Judge evaluation.")
            if looks_like_placeholder_secret(self.settings.openai_api_key):
                raise RuntimeError(
                    "OPENAI_API_KEY is still set to the placeholder value. Update .env with a real OpenAI API key and restart."
                )
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def _normalize_payload(self, payload: dict) -> tuple[dict, bool]:
        normalized = dict(payload)
        used_normalization_fallback = False
        for key in ("groundedness", "citation_quality", "completeness", "insufficiency_handling"):
            normalized[key], used_fallback = self._normalize_diagnostic_value(normalized.get(key))
            used_normalization_fallback = used_normalization_fallback or used_fallback

        if not normalized.get("rationale"):
            normalized["rationale"] = "Judge response was incomplete."
        if not normalized.get("final_answer"):
            normalized["final_answer"] = ""
        needs_revision = normalized.get("needs_revision")
        if isinstance(needs_revision, str):
            normalized["needs_revision"] = needs_revision.strip().lower() in {"true", "yes", "1"}

        return normalized, used_normalization_fallback

    def _normalize_diagnostic_value(self, value: object) -> tuple[int | None, bool]:
        if value is None or value == "":
            return None, False
        if isinstance(value, bool):
            return None, True
        if isinstance(value, int):
            return value if 1 <= value <= 5 else None, not (1 <= value <= 5)
        if isinstance(value, float):
            integer_value = int(value)
            return (integer_value, False) if 1 <= integer_value <= 5 else (None, True)
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in self._DIAGNOSTIC_LABELS:
                return self._DIAGNOSTIC_LABELS[cleaned], False
            digits = "".join(char for char in cleaned if char.isdigit())
            if digits:
                numeric_value = int(digits)
                return (numeric_value, False) if 1 <= numeric_value <= 5 else (None, True)
            return None, True
        return None, True
