"""The shared state that every node reads from and writes to."""
from typing import TypedDict


class AnnotationState(TypedDict):
    # --- input ---
    request: str
    items: list[dict]            # [{"id": "1", "text": "..."}]

    # --- taxonomy, guideline and review ---
    taxonomy: dict               # labels only: {"labels": [...]}
    guideline: dict              # rules only:  {"rules": [...]}
    approved: bool
    feedback: str

    # --- annotations ---
    annotations_a: dict          # {"1": {"label": "...", "reason": "..."}}
    annotations_b: dict

    # --- IAA and disagreements ---
    iaa_score: float
    disagreements: list[dict]
    disagreement_report: str

    # --- iteration ---
    iteration: int               # number of annotation rounds completed
    iaa_history: list[dict]      # [{"round": 1, "kappa": 0.52}, ...]

    # --- output ---
    final_data: list[dict]

    # --- gold set evaluation ---
    gold_size: int               # 0 = skip gold evaluation
    gold_labels: dict            # {item_id: label assigned by hand}
    gold_metrics: dict
