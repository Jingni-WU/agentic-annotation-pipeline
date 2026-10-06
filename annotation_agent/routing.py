"""Routing functions: read the state and return the name of the next node."""
from annotation_agent import config
from annotation_agent.state import AnnotationState


def route_after_review(state: AnnotationState) -> str:
    if state["approved"]:
        return "annotate_a"
    if state.get("iteration", 0) == 0:
        return "generate_taxonomy"     # before any annotation: redesign from scratch
    return "revise_guideline"          # inside the loop: keep labels, revise rules again


def route_after_analysis(state: AnnotationState) -> str:
    if state["iaa_score"] >= config.IAA_THRESHOLD:
        return "adjudicate"
    if state["iteration"] >= config.MAX_ROUNDS:
        print(f"\n>>> Reached max rounds ({config.MAX_ROUNDS}) without hitting the threshold. Proceeding anyway.")
        return "adjudicate"
    return "revise_guideline"


def route_after_adjudicate(state: AnnotationState) -> str:
    if state.get("gold_size", 0) > 0:
        return "collect_gold"
    return "clean_data"                # skip gold evaluation entirely
