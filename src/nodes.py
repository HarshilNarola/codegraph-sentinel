import time
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.ingestion import HybridCodeRetriever
from src.state import CodeReviewState


class RouteQuery(BaseModel):
    intent: Literal["architecture", "security_audit", "refactor"] = Field(
        description="Classify the request as architecture, security_audit, or refactor."
    )


class GradeDocuments(BaseModel):
    sufficient: bool = Field(description="Whether the retrieved code is sufficient.")
    reasoning: str = Field(description="Why the context is or is not sufficient.")


def _context(state: CodeReviewState) -> str:
    return "\n\n".join(
        f"### File: `{doc['file_path']}`\n```\n{doc['content']}\n```"
        for doc in state.get("documents", [])
    )


def build_nodes(retriever: HybridCodeRetriever):
    fast_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    deep_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

    def timed(state, name, updates):
        timings = dict(state.get("node_timings", {}))
        timings[name] = round((time.perf_counter() - updates.pop("_started")) * 1000, 1)
        updates["node_timings"] = timings
        return updates

    def router_node(state: CodeReviewState):
        started = time.perf_counter()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Route this codebase request to architecture, security_audit, or refactor."),
             ("human", "{question}")]
        )
        result = (prompt | fast_llm.with_structured_output(RouteQuery)).invoke(
            {"question": state["question"]}
        )
        return timed(state, "router", {"intent": result.intent, "optimized_query": state["question"],
                                       "loop_count": 0, "_started": started})

    def retrieve_node(state: CodeReviewState):
        started = time.perf_counter()
        docs = retriever.search(state.get("optimized_query") or state["question"])
        return timed(state, f"retrieve_pass_{state.get('loop_count', 0) + 1}",
                     {"documents": docs, "_started": started})

    def grade_context_node(state: CodeReviewState):
        started = time.perf_counter()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Decide whether the supplied code chunks are sufficient to answer without guessing."),
             ("human", "Query: {question}\n\nRetrieved code:\n{context}")]
        )
        verdict = (prompt | fast_llm.with_structured_output(GradeDocuments)).invoke(
            {"question": state["question"], "context": _context(state)}
        )
        return timed(state, "grade_context", {"grade_passed": verdict.sufficient, "_started": started})

    def rewrite_query_node(state: CodeReviewState):
        started = time.perf_counter()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Rewrite this code search query using likely symbols and technical keywords. Return only the query."),
             ("human", "{question}")]
        )
        rewritten = (prompt | fast_llm).invoke({"question": state["question"]}).content.strip()
        return timed(state, "rewrite_query", {"optimized_query": rewritten,
                                              "loop_count": state.get("loop_count", 0) + 1,
                                              "_started": started})

    def specialist_node(state: CodeReviewState, name: str, system: str):
        started = time.perf_counter()
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "Question: {question}\n\nCode context:\n{context}")]
        )
        answer = (prompt | deep_llm).invoke(
            {"question": state["question"], "context": _context(state)}
        ).content
        return timed(state, name, {"final_response": answer, "_started": started})

    def architecture_analyst_node(state):
        return specialist_node(state, "architecture_analyst",
                               "Explain the architecture and data flow strictly from the code. Cite exact file paths.")

    def security_auditor_node(state):
        return specialist_node(state, "security_auditor",
                               "Audit for vulnerabilities and edge-case bugs. Rate severity and cite exact files.")

    def refactor_planner_node(state):
        return specialist_node(state, "refactor_planner",
                               "Propose a safe unified git diff, then explain its maintainability or complexity benefits.")

    def human_approval_node(state: CodeReviewState):
        decision = interrupt({"message": "Review the proposed refactoring diff before finalizing.",
                              "proposed_diff": state.get("final_response", "")})
        approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else bool(decision)
        if approved:
            message = f"### Approved Refactoring Patch\n\n{state.get('final_response', '')}\n\n> **Status:** Approved by human reviewer."
        else:
            message = "### Refactoring Patch Rejected\n\nThe proposed changes were discarded per reviewer decision."
        return {"human_approved": approved, "final_response": message}

    return {"router": router_node, "retrieve": retrieve_node, "grade_context": grade_context_node,
            "rewrite_query": rewrite_query_node, "architecture_analyst": architecture_analyst_node,
            "security_auditor": security_auditor_node, "refactor_planner": refactor_planner_node,
            "human_approval": human_approval_node}
