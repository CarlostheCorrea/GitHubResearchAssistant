from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

from openai import OpenAI

from backend.config import Settings
from backend.judge_service import LLMJudgeService
from backend.models import (
    CommitRecord,
    ContributorProfile,
    CoreConcept,
    OnboardingResponse,
    ReadingStep,
    RepoDescriptor,
    RepoSummary,
)
from backend.utils import looks_like_placeholder_secret

logger = logging.getLogger(__name__)


class OnboardingService:
    def __init__(self, settings: Settings, judge: LLMJudgeService | None = None) -> None:
        self.settings = settings
        self.judge = judge
        self._client: OpenAI | None = None

    def generate(
        self,
        repo: RepoDescriptor,
        summary: RepoSummary,
        commit_history: list[CommitRecord],
    ) -> OnboardingResponse:
        contributors = self._extract_contributors(commit_history)
        reading_order, core_concepts, complexity_note = self._generate_guide(repo, summary, contributors)
        return OnboardingResponse(
            repo_name=repo.repo_name,
            reading_order=reading_order,
            core_concepts=core_concepts,
            contributors=contributors,
            complexity_note=complexity_note,
        )

    def _extract_contributors(self, commit_history: list[CommitRecord]) -> list[ContributorProfile]:
        if not commit_history:
            return []
        author_commits: dict[str, list[CommitRecord]] = defaultdict(list)
        for commit in commit_history:
            author_commits[commit.author_name].append(commit)

        profiles = []
        for author, commits in sorted(author_commits.items(), key=lambda x: -len(x[1])):
            all_files: list[str] = []
            for commit in commits:
                for fc in commit.file_changes:
                    all_files.append(fc["path"])
            file_counts = Counter(all_files)
            recent_files = [f for f, _ in file_counts.most_common(5)]
            focus = self._infer_focus(all_files)
            profiles.append(
                ContributorProfile(
                    name=author,
                    commits=len(commits),
                    focus_area=focus,
                    recent_files=recent_files[:4],
                )
            )
        return profiles[:8]

    def _infer_focus(self, all_files: list[str]) -> str:
        if not all_files:
            return "General contributions"
        dirs = [f.split("/")[0] if "/" in f else "root" for f in all_files]
        dir_counts = Counter(dirs)
        top_dir = dir_counts.most_common(1)[0][0] if dir_counts else "root"
        exts = [f.rsplit(".", 1)[-1].lower() for f in all_files if "." in f]
        ext_counts = Counter(exts)
        md_count = ext_counts.get("md", 0) + ext_counts.get("rst", 0)
        if md_count > len(all_files) * 0.6:
            return "Documentation"
        if top_dir in ("frontend", "ui", "client", "web"):
            return "Frontend development"
        if top_dir in ("backend", "api", "server"):
            return "Backend development"
        if top_dir in ("tests", "test", "__tests__"):
            return "Testing"
        top_ext = ext_counts.most_common(1)[0][0] if ext_counts else ""
        if top_ext == "py":
            return "Python development"
        if top_ext in ("js", "ts", "tsx"):
            return "JavaScript / TypeScript development"
        if top_ext == "go":
            return "Go development"
        if top_dir == "root":
            return "Project configuration and setup"
        return f"{top_dir.capitalize()} development"

    def _generate_guide(
        self,
        repo: RepoDescriptor,
        summary: RepoSummary,
        contributors: list[ContributorProfile],
    ) -> tuple[list[ReadingStep], list[CoreConcept], str]:
        try:
            client = self._get_client()
        except RuntimeError:
            return [], [], ""

        key_files_str = ", ".join(summary.key_files[:12]) or "not available"
        entry_str = ", ".join(summary.probable_entry_points[:6]) or "not available"
        config_str = ", ".join(summary.probable_config_files[:4]) or "not available"

        prompt = (
            f"Repository: {repo.repo_name}\n"
            f"Branch: {repo.branch}\n"
            f"Languages: {', '.join(summary.detected_languages[:6])}\n"
            f"Summary: {summary.high_level_summary}\n"
            f"Key files: {key_files_str}\n"
            f"Entry points: {entry_str}\n"
            f"Config files: {config_str}\n"
            f"Graph context: {summary.global_context[:500] if summary.global_context else 'not available'}\n\n"
            "Generate a new-developer onboarding guide. Respond with ONLY valid JSON:\n\n"
            '{"reading_order":[{"step":1,"file_path":"...","role":"short role","reason":"one sentence why"}],'
            '"core_concepts":[{"name":"...","description":"2-3 sentences","key_files":["..."]}],'
            '"complexity_note":"one sentence about the hardest part"}\n\n'
            "Rules: reading_order has 5-8 files from easiest to hardest. "
            "core_concepts has 3-5 key abstractions specific to THIS repo. "
            "All text is plain prose, no markdown."
        )

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0.2,
                max_tokens=1400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You generate structured JSON developer onboarding guides."},
                    {"role": "user", "content": prompt},
                ],
            )
            data = json.loads(response.choices[0].message.content.strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Onboarding guide generation failed: %s", exc)
            return [], [], ""

        if self.judge is not None:
            data = self.judge.review_structured_json(
                task_name="developer onboarding guide",
                draft_payload=data,
                context_text=self._build_judge_context(repo, summary, contributors),
                output_contract=(
                    "A JSON object with reading_order, core_concepts, and complexity_note. "
                    "reading_order is an array of objects with step, file_path, role, reason. "
                    "core_concepts is an array of objects with name, description, key_files. "
                    "complexity_note is a string."
                ),
            )

        reading_order = [
            ReadingStep(
                step=item.get("step", i + 1),
                file_path=str(item.get("file_path", "")),
                role=str(item.get("role", "general")),
                reason=str(item.get("reason", "")),
            )
            for i, item in enumerate(data.get("reading_order", []))
            if item.get("file_path")
        ]
        core_concepts = [
            CoreConcept(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                key_files=item.get("key_files", []),
            )
            for item in data.get("core_concepts", [])
            if item.get("name")
        ]
        return reading_order, core_concepts, str(data.get("complexity_note", ""))

    def _build_judge_context(
        self,
        repo: RepoDescriptor,
        summary: RepoSummary,
        contributors: list[ContributorProfile],
    ) -> str:
        contributor_lines = [
            f"- {contributor.name}: {contributor.commits} commits, {contributor.focus_area}, files: {', '.join(contributor.recent_files) or 'n/a'}"
            for contributor in contributors[:8]
        ]
        return "\n".join(
            [
                f"Repository: {repo.repo_name}",
                f"Branch: {repo.branch}",
                f"Languages: {', '.join(summary.detected_languages[:8]) or 'n/a'}",
                f"High-level summary: {summary.high_level_summary}",
                f"Key files: {', '.join(summary.key_files[:16]) or 'n/a'}",
                f"Entry points: {', '.join(summary.probable_entry_points[:8]) or 'n/a'}",
                f"Config files: {', '.join(summary.probable_config_files[:8]) or 'n/a'}",
                f"Training files: {', '.join(summary.probable_training_files[:8]) or 'n/a'}",
                f"Inference files: {', '.join(summary.probable_inference_files[:8]) or 'n/a'}",
                f"Data files: {', '.join(summary.probable_data_files[:8]) or 'n/a'}",
                f"Graph context: {summary.global_context[:1200] if summary.global_context else 'n/a'}",
                "Contributor evidence:",
                "\n".join(contributor_lines) if contributor_lines else "n/a",
            ]
        )

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required.")
            if looks_like_placeholder_secret(self.settings.openai_api_key):
                raise RuntimeError("OPENAI_API_KEY is still a placeholder.")
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
