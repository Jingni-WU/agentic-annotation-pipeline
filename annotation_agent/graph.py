"""Build the LangGraph StateGraph."""
from langgraph.graph import END, START, StateGraph

from annotation_agent.nodes import (
    adjudicate,
    analyze_disagreements,
    annotate_a,
    annotate_b,
    calculate_iaa,
    clean_data,
    collect_gold,
    evaluate_gold,
    export_data,
    generate_guideline,
    generate_taxonomy,
    human_review,
    revise_guideline,
)
from annotation_agent.routing import (
    route_after_adjudicate,
    route_after_analysis,
    route_after_review,
)
from annotation_agent.state import AnnotationState


def build_graph(checkpointer):
    """Build and compile the graph. The caller decides which checkpointer to use."""
    builder = StateGraph(AnnotationState)

    # ---------- register nodes ----------
    builder.add_node("generate_taxonomy", generate_taxonomy)
    builder.add_node("generate_guideline", generate_guideline)
    builder.add_node("human_review", human_review)
    builder.add_node("annotate_a", annotate_a)
    builder.add_node("annotate_b", annotate_b)
    builder.add_node("calculate_iaa", calculate_iaa)
    builder.add_node("analyze_disagreements", analyze_disagreements)
    builder.add_node("revise_guideline", revise_guideline)
    builder.add_node("adjudicate", adjudicate)
    builder.add_node("collect_gold", collect_gold)
    builder.add_node("evaluate_gold", evaluate_gold)
    builder.add_node("clean_data", clean_data)
    builder.add_node("export_data", export_data)

    # ---------- edges ----------
    builder.add_edge(START, "generate_taxonomy")
    builder.add_edge("generate_taxonomy", "generate_guideline")
    builder.add_edge("generate_guideline", "human_review")

    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        ["annotate_a", "generate_taxonomy", "revise_guideline"],
    )

    builder.add_edge("annotate_a", "annotate_b")
    builder.add_edge("annotate_b", "calculate_iaa")
    builder.add_edge("calculate_iaa", "analyze_disagreements")

    builder.add_conditional_edges(
        "analyze_disagreements",
        route_after_analysis,
        ["revise_guideline", "adjudicate"],
    )
    builder.add_edge("revise_guideline", "human_review")

    builder.add_conditional_edges(
        "adjudicate",
        route_after_adjudicate,
        ["collect_gold", "clean_data"],
    )
    builder.add_edge("collect_gold", "evaluate_gold")
    builder.add_edge("evaluate_gold", "clean_data")
    builder.add_edge("clean_data", "export_data")
    builder.add_edge("export_data", END)

    return builder.compile(checkpointer=checkpointer)
