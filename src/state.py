from typing import Any, Dict, List, Literal, Optional, TypedDict


class CodeReviewState(TypedDict, total=False):
    question: str
    optimized_query: str
    intent: Literal["architecture", "security_audit", "refactor"]
    documents: List[Dict[str, Any]]
    grade_passed: bool
    loop_count: int
    proposed_diff: Optional[str]
    human_approved: Optional[bool]
    final_response: str
    node_timings: Dict[str, float]
