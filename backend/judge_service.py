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
1. Judge whether the draft answer is grounded, cited well, complete, and appropriately cautious.
2. If the draft answer has problems, rewrite it into a better final answer.
3. If the draft answer is already good, preserve it with only minimal edits.

Hard rules for the final answer:
- Use only the provided repository evidence.
- Do not invent files, functions, classes, behavior, or architecture.
- Keep inline citations in the format [path:start-end].
- If the evidence is insufficient, say so clearly.
- Return the best final answer for the user. Do not mention judging, review, or internal evaluation.

Return strict JSON only.
"""


class JudgeReview(BaseModel):
    groundedness: int = Field(..., ge=1, le=5)
    citation_quality: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    insufficiency_handling: int = Field(..., ge=1, le=5)
    needs_revision: bool = False
    rationale: str
    final_answer: str


class LLMJudgeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def review_and_revise_answer(
        self,
        question: str,
        draft_answer: str,
        sources: list[SourceSnippet],
    ) -> str:
        client = self._get_client()
        logger.info("Running LLM-as-a-Judge for question: %s", question)
        prompt = self._build_prompt(question, draft_answer, sources)
        completion = client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
            normalized = self._normalize_payload(payload)
            review = JudgeReview.model_validate(normalized)
            final_answer = review.final_answer.strip()
            return final_answer or draft_answer
        except Exception:  # noqa: BLE001
            logger.warning("LLM judge returned invalid payload: %s", content[:800], exc_info=True)
            return draft_answer

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
                "Return JSON with keys: groundedness, citation_quality, completeness, insufficiency_handling, needs_revision, rationale, final_answer.",
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

    def _normalize_payload(self, payload: dict) -> dict:
        normalized = dict(payload)
        for key in ("groundedness", "citation_quality", "completeness", "insufficiency_handling"):
            value = normalized.get(key)
            if isinstance(value, str):
                digits = "".join(char for char in value if char.isdigit())
                normalized[key] = int(digits) if digits else value

        if not normalized.get("rationale"):
            normalized["rationale"] = "Judge response was incomplete."
        if not normalized.get("final_answer"):
            normalized["final_answer"] = ""
        needs_revision = normalized.get("needs_revision")
        if isinstance(needs_revision, str):
            normalized["needs_revision"] = needs_revision.strip().lower() in {"true", "yes", "1"}

        return normalized
