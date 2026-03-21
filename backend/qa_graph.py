from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from backend.judge_service import LLMJudgeService
from backend.models import ChunkRecord, RepoDescriptor, RepoSummary, SourceSnippet
from backend.qa_service import QAService
from backend.retriever import HybridRetriever


class QAState(TypedDict, total=False):
    repo: RepoDescriptor
    repo_summary: RepoSummary
    question: str
    retrieved_chunks: list[tuple[ChunkRecord, float]]
    sources: list[SourceSnippet]
    draft_answer: str
    answer: str


class RepoQAGraph:
    def __init__(
        self,
        retriever: HybridRetriever,
        qa_service: QAService,
        judge_service: LLMJudgeService,
    ) -> None:
        self.retriever = retriever
        self.qa_service = qa_service
        self.judge_service = judge_service
        self.graph = self._build_graph()

    def run(self, repo: RepoDescriptor, repo_summary: RepoSummary, question: str) -> QAState:
        return self.graph.invoke(
            {
                "repo": repo,
                "repo_summary": repo_summary,
                "question": question,
            }
        )

    def _build_graph(self):
        builder = StateGraph(QAState)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("review_answer", self._review_answer)
        builder.add_edge(START, "retrieve_context")
        builder.add_edge("retrieve_context", "generate_answer")
        builder.add_edge("generate_answer", "review_answer")
        builder.add_edge("review_answer", END)
        return builder.compile()

    def _retrieve_context(self, state: QAState) -> QAState:
        retrieved_chunks = self.retriever.retrieve(
            state["repo"],
            state["question"],
        )
        sources = [
            SourceSnippet.from_chunk(chunk, score)
            for chunk, score in retrieved_chunks
        ]
        return {"retrieved_chunks": retrieved_chunks, "sources": sources}

    def _generate_answer(self, state: QAState) -> QAState:
        return {
            "draft_answer": self.qa_service.answer_question(
                state["repo_summary"],
                state["question"],
                state.get("retrieved_chunks", []),
            )
        }

    def _review_answer(self, state: QAState) -> QAState:
        draft_answer = state.get("draft_answer", "")
        try:
            final_answer = self.judge_service.review_and_revise_answer(
                state["question"],
                draft_answer,
                state.get("sources", []),
            )
        except Exception:  # noqa: BLE001
            final_answer = draft_answer
        return {"answer": final_answer or draft_answer}
