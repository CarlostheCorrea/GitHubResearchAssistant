from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import indent

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import service


DEFAULT_QUESTIONS = [
    "What are the main components of this repository?",
    "How does this repo train the model?",
    "Where is the inference code?",
    "How is data loaded?",
    "Where is the configuration defined?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small evaluation loop against a public GitHub repository.")
    parser.add_argument("repo_url", help="Public GitHub repository URL")
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to ask. You can pass this flag multiple times.",
    )
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS
    analysis = service.analyze_repo(args.repo_url)

    print("Analysis summary")
    print(f"  Repo: {analysis.repo_summary.repo_name}")
    print(f"  Files indexed: {analysis.files_indexed}")
    print(f"  Chunks created: {analysis.chunks_created}")
    print(f"  Cached: {analysis.cached}")
    print()

    for question in questions:
        response = service.ask(args.repo_url, question)
        print(f"Question: {question}")
        print("Answer:")
        print(indent(response.answer, "  "))
        print("Sources:")
        for source in response.sources:
            lines = (
                f"{source.start_line}-{source.end_line}"
                if source.start_line and source.end_line
                else "n/a"
            )
            print(f"  - {source.file_path} ({lines}) [{source.chunk_type}] score={source.score}")
        print()


if __name__ == "__main__":
    main()
