from __future__ import annotations

import unittest

from backend.knowledge_graph import KnowledgeGraphService
from backend.models import ChunkRecord, RepoDescriptor, RepoFile
from backend.models import RepoSummary


class KnowledgeGraphServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = KnowledgeGraphService()
        self.repo = RepoDescriptor(
            owner="octocat",
            repo="demo",
            branch="main",
            default_branch="main",
            normalized_repo_url="https://github.com/octocat/demo",
            repo_id="octocat_demo_main",
        )

    def test_build_snapshot_creates_global_context_and_dependencies(self) -> None:
        files = [
            RepoFile(
                path="app/main.py",
                size=120,
                language="python",
                role="entrypoint",
                content="from app.train import train_model\nfrom config.settings import MODEL_NAME\n",
            ),
            RepoFile(
                path="app/train.py",
                size=220,
                language="python",
                role="training",
                content="def train_model():\n    pass\n",
            ),
            RepoFile(
                path="config/settings.py",
                size=80,
                language="python",
                role="config",
                content="MODEL_NAME = 'demo'\n",
            ),
        ]
        chunks = [
            ChunkRecord(
                id="1",
                repo_id=self.repo.repo_id,
                repo_name=self.repo.repo_name,
                file_path="app/main.py",
                language="python",
                chunk_type="python_function",
                symbol_name="main",
                start_line=1,
                end_line=10,
                short_summary="Main entrypoint",
                file_role="entrypoint",
                text="from app.train import train_model\nfrom config.settings import MODEL_NAME\ndef main():\n    train_model()\n",
            ),
            ChunkRecord(
                id="2",
                repo_id=self.repo.repo_id,
                repo_name=self.repo.repo_name,
                file_path="app/train.py",
                language="python",
                chunk_type="python_function",
                symbol_name="train_model",
                start_line=1,
                end_line=12,
                short_summary="Training function",
                file_role="training",
                text="def train_model():\n    pass\n",
            ),
        ]

        snapshot = self.service.build_snapshot(self.repo, files, chunks)

        self.assertEqual(snapshot.repo_id, self.repo.repo_id)
        self.assertEqual(snapshot.languages, ["python"])
        self.assertEqual(len(snapshot.symbols), 2)
        self.assertIn(
            {"source_path": "app/main.py", "target_path": "app/train.py"},
            snapshot.dependencies,
        )
        self.assertIn(
            {"source_path": "app/main.py", "target_path": "config/settings.py"},
            snapshot.dependencies,
        )
        self.assertIn(["app/main.py", "app/train.py"], snapshot.critical_paths)
        self.assertIn(["app/main.py", "app/train.py"], snapshot.dependency_links)
        self.assertIn("app/main.py", snapshot.graph_hubs)
        self.assertIn("Repository-wide graph context for octocat/demo", snapshot.global_context)
        self.assertIn("Likely entrypoint files: app/main.py.", snapshot.global_context)
        self.assertIn("Training-related files: app/train.py.", snapshot.global_context)

    def test_ensure_summary_global_context_hydrates_cached_summary(self) -> None:
        summary = RepoSummary(
            repo_name="octocat/demo",
            owner="octocat",
            branch="main",
            normalized_repo_url="https://github.com/octocat/demo",
            detected_languages=["python"],
            language_distribution={"python": 3},
            key_files=["app/main.py", "config/settings.py"],
            high_level_summary="Demo repo",
            probable_entry_points=["app/main.py"],
            probable_training_files=["app/train.py"],
            probable_inference_files=[],
            probable_config_files=["config/settings.py"],
            probable_data_files=[],
            files_indexed=3,
            chunks_indexed=12,
        )

        hydrated = self.service.ensure_summary_global_context(summary)

        self.assertEqual(summary.global_context, "")
        self.assertIn("Repository-wide graph context for octocat/demo", hydrated.global_context)
        self.assertIn("Likely entrypoint files: app/main.py.", hydrated.global_context)
        self.assertIn("Configuration files: config/settings.py.", hydrated.global_context)
        self.assertEqual(hydrated.critical_paths, [["app/main.py", "config/settings.py"]])
        self.assertEqual(hydrated.dependency_links, [["app/main.py", "config/settings.py"]])
        self.assertEqual(hydrated.graph_hubs, ["app/main.py", "config/settings.py"])


if __name__ == "__main__":
    unittest.main()
