from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.ingestion import HybridCodeRetriever
from src.nodes import build_nodes
from src.state import CodeReviewState


def create_review_graph(retriever: HybridCodeRetriever):
    workflow = StateGraph(CodeReviewState)
    for name, fn in build_nodes(retriever).items():
        workflow.add_node(name, fn)

    def route_after_grading(state: CodeReviewState) -> str:
        if not state.get("grade_passed") and state.get("loop_count", 0) < 2:
            return "rewrite_query"
        return {"security_audit": "security_auditor", "refactor": "refactor_planner"}.get(
            state.get("intent"), "architecture_analyst"
        )

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "grade_context")
    workflow.add_conditional_edges("grade_context", route_after_grading,
                                   {"rewrite_query": "rewrite_query",
                                    "architecture_analyst": "architecture_analyst",
                                    "security_auditor": "security_auditor",
                                    "refactor_planner": "refactor_planner"})
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("architecture_analyst", END)
    workflow.add_edge("security_auditor", END)
    workflow.add_edge("refactor_planner", "human_approval")
    workflow.add_edge("human_approval", END)
    return workflow.compile(checkpointer=InMemorySaver())
