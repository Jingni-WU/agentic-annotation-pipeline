"""
Tests for the annotation pipeline.

All tests use MockLLM, so they are free and fast.
Run from the project folder with:
    python -m pytest -v
"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import pipeline


# =====================================================================
# Fixtures: shared setup for many tests
# =====================================================================
@pytest.fixture
def items():
    """A small fake dataset."""
    texts = [
        "Where is my package?",
        "Love it, works perfectly!",
        "Does this come in blue?",
        "The shoes fell apart after two days.",
        "Nobody answered my emails.",
        "I want a refund for the broken cup.",
        "Not what I expected.",
        "Can I change my shipping address?",
    ]
    return [{"id": str(i), "text": t} for i, t in enumerate(texts, start=1)]


@pytest.fixture
def graph(monkeypatch, tmp_path):
    """
    A fresh graph that uses MockLLM and writes files to a temporary folder.
    monkeypatch changes are undone automatically after each test.
    """
    monkeypatch.chdir(tmp_path)                        # output/ goes to a temp folder
    monkeypatch.setattr(pipeline, "MOCK_DELAY", 0)     # no waiting in tests
    monkeypatch.setattr(pipeline, "llm_generator", pipeline.MockLLM("generator"))
    monkeypatch.setattr(pipeline, "llm_a", pipeline.MockLLM("annotator_a", seed=1))
    monkeypatch.setattr(pipeline, "llm_b", pipeline.MockLLM("annotator_b", seed=2))
    monkeypatch.setattr(pipeline, "llm_judge", pipeline.MockLLM("judge", seed=3))

    # In-memory checkpointer, so tests don't touch checkpoints.db
    return pipeline.builder.compile(checkpointer=InMemorySaver())


def run_to_end(graph, initial_state):
    """
    Run the whole graph, answering every pause automatically:
      - guideline review -> "y"
      - gold labeling    -> label every item with the first label
    Returns (final_state, list_of_node_names_in_order).
    """
    config = {"configurable": {"thread_id": "test"}}
    visited = []
    graph_input = initial_state

    while True:
        interrupt_value = None
        for update in graph.stream(graph_input, config, stream_mode="updates"):
            for node_name, output in update.items():
                if node_name == "__interrupt__":
                    interrupt_value = output[0].value
                else:
                    visited.append(node_name)

        if interrupt_value is None:          # finished
            break

        if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "gold_labeling":
            first_label = interrupt_value["labels"][0]
            answer = {item["id"]: first_label for item in interrupt_value["items"]}
        else:
            answer = "y"
        graph_input = Command(resume=answer)

    return graph.get_state(config).values, visited


# =====================================================================
# Unit tests: one node or function at a time, no graph
# =====================================================================
def test_calculate_iaa_finds_disagreements():
    state = {
        "items": [
            {"id": "1", "text": "a"},
            {"id": "2", "text": "b"},
            {"id": "3", "text": "c"},
        ],
        "annotations_a": {
            "1": {"label": "shipping", "reason": ""},
            "2": {"label": "return_refund", "reason": ""},
            "3": {"label": "praise", "reason": ""},
        },
        "annotations_b": {
            "1": {"label": "shipping", "reason": ""},
            "2": {"label": "complaint", "reason": ""},
            "3": {"label": "praise", "reason": ""},
        },
    }
    result = pipeline.calculate_iaa(state)

    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["id"] == "2"
    assert result["iteration"] == 1
    assert len(result["iaa_history"]) == 1


def test_clean_data_drops_invalid_and_duplicates():
    state = {
        "final_data": [
            {"id": "1", "text": "Hello  world", "final_label": "praise"},
            {"id": "2", "text": "hello world", "final_label": "praise"},   # duplicate
            {"id": "3", "text": "Broken item", "final_label": "INVALID"},  # invalid
            {"id": "4", "text": "Where is it?", "final_label": "shipping"},
        ]
    }
    result = pipeline.clean_data(state)
    kept_ids = [row["id"] for row in result["final_data"]]

    assert kept_ids == ["1", "4"]
    assert result["final_data"][0]["text"] == "Hello world"   # extra spaces removed


def test_route_after_analysis():
    t = pipeline.IAA_THRESHOLD
    max_rounds = pipeline.MAX_ROUNDS

    # Kappa is good enough -> adjudicate
    assert pipeline.route_after_analysis({"iaa_score": t + 0.1, "iteration": 1}) == "adjudicate"
    # Kappa too low, rounds left -> revise
    assert pipeline.route_after_analysis({"iaa_score": t - 0.1, "iteration": 1}) == "revise_guideline"
    # Kappa too low, no rounds left -> adjudicate anyway
    assert pipeline.route_after_analysis({"iaa_score": t - 0.1, "iteration": max_rounds}) == "adjudicate"


# =====================================================================
# Graph tests: run the whole pipeline with MockLLM
# =====================================================================
def test_passes_on_first_round(graph, items, monkeypatch):
    monkeypatch.setattr(pipeline, "IAA_THRESHOLD", 0.0)   # any kappa passes
    state, visited = run_to_end(graph, {"request": "test", "items": items, "gold_size": 0})

    assert state["iteration"] == 1
    assert "revise_guideline" not in visited
    assert visited[-1] == "export_data"


def test_stops_at_max_rounds(graph, items, monkeypatch):
    monkeypatch.setattr(pipeline, "IAA_THRESHOLD", 1.01)  # impossible to reach
    state, visited = run_to_end(graph, {"request": "test", "items": items, "gold_size": 0})

    assert state["iteration"] == pipeline.MAX_ROUNDS
    assert visited.count("calculate_iaa") == pipeline.MAX_ROUNDS
    assert len(state["iaa_history"]) == pipeline.MAX_ROUNDS


def test_gold_skipped_when_size_is_zero(graph, items, monkeypatch):
    monkeypatch.setattr(pipeline, "IAA_THRESHOLD", 0.0)
    state, visited = run_to_end(graph, {"request": "test", "items": items, "gold_size": 0})

    assert "collect_gold" not in visited
    assert "evaluate_gold" not in visited


def test_gold_evaluation_runs(graph, items, monkeypatch):
    monkeypatch.setattr(pipeline, "IAA_THRESHOLD", 0.0)
    state, visited = run_to_end(graph, {"request": "test", "items": items, "gold_size": 3})

    assert "collect_gold" in visited
    assert "evaluate_gold" in visited
    assert state["gold_metrics"]["n_gold"] == 3
    assert 0.0 <= state["gold_metrics"]["accuracy_final"] <= 1.0


def test_export_creates_files(graph, items, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "IAA_THRESHOLD", 0.0)
    run_to_end(graph, {"request": "test", "items": items, "gold_size": 0})

    assert (tmp_path / "output" / "annotations.json").exists()
    assert (tmp_path / "output" / "annotations.csv").exists()
    assert (tmp_path / "output" / "run_meta.json").exists()