"""Graph nodes. Each node takes the state and returns a dict of the fields it updates."""
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from langgraph.types import interrupt
from sklearn.metrics import classification_report, cohen_kappa_score

from annotation_agent import config, llms
from annotation_agent.schemas import Annotation, Guideline, Taxonomy
from annotation_agent.state import AnnotationState


# =====================================================================
# Helper functions
# =====================================================================
# TODO: paste taxonomy_to_text, guideline_to_text, full_instructions
#       and annotate_with from section 5 of pipeline.py
"""convert taxonomy to text which is easier for human to read."""
def taxonomy_to_text(taxonomy: dict) -> str:
    """Labels only."""
    return "\n".join(f"  - {l['name']}: {l['definition']}" for l in taxonomy["labels"])


def guideline_to_text(guideline: dict) -> str:
    """Rules only."""
    return "\n".join(f"  {i}. {rule}" for i, rule in enumerate(guideline["rules"], start=1))


def full_instructions(state: AnnotationState) -> str:
    """Taxonomy + guideline as two clearly separated sections. Used in annotator/judge prompts."""
    return "\n".join([
        "## Taxonomy (labels)",
        taxonomy_to_text(state["taxonomy"]),
        "",
        "## Guideline (rules)",
        guideline_to_text(state["guideline"]),
    ])
 
def annotate_with(llm, state: AnnotationState) -> dict:
    """use designated llms for annotation. annotate_a and annotate_b use the same function."""
    taxonomy = state["taxonomy"]
    valid_labels = [label["name"] for label in taxonomy["labels"]]
    structured_llm = llm.with_structured_output(Annotation)
 
    results = {}
    for item in state["items"]:
        prompt = f"""
        Strictly follow the annotation guideline and annotate the text with a label.
        only use labels from: {valid_labels}
        {full_instructions(state)}
        text: {item['text']}
        """
        try:
            output = structured_llm.invoke(prompt)
            # double check labels that are made up by llms instead of from the taxanomy.
            label = output.label if output.label in valid_labels else "INVALID"
            results[item["id"]] = {"label": label, "reason": output.reason}
        except Exception:
            # some local models might make mistakes when output results. mark the wrong formats as 'invalid' and clean them up later.
            results[item["id"]] = {"label": "INVALID", "reason": "model parse failed."}
        print(f"  [{item['id']}] {results[item['id']]['label']}")
    return results


# =====================================================================
# Nodes
# =====================================================================
# TODO: paste every node function from section 5 of pipeline.py:
#   generate_taxonomy, generate_guideline, human_review, annotate_a, annotate_b,
#   calculate_iaa, analyze_disagreements, revise_guideline, adjudicate,
#   collect_gold, evaluate_gold, clean_data, export_data
#
# Then rename these, everywhere in this file:
#   llm_generator  ->  llms.llm_generator
#   llm_a          ->  llms.llm_a
#   llm_b          ->  llms.llm_b
#   llm_judge      ->  llms.llm_judge
#   IAA_THRESHOLD  ->  config.IAA_THRESHOLD

# ---------- Node 1: generate taxonomy (labels only) ----------
def generate_taxonomy(state: AnnotationState) -> dict:
    print("\n>>> generating taxonomy ...")
    samples = "\n".join(f"- {item['text']}" for item in state["items"][:10])

    prompt = f"""
    You are the lead of an annotation project. Based on the annotation request and sample data, design a label taxonomy.
    Requirements: labels must be mutually exclusive and cover all cases (add an "other" label if needed).
    Only define the labels. Do not write annotation rules.

    Annotation request: 
    {state['request']}

    Sample data:
    {samples}
    """

    if state.get("feedback"):
        prompt += f"""
    Previous taxonomy:
    {taxonomy_to_text(state['taxonomy'])}
    Previous guideline:
    {guideline_to_text(state['guideline'])}
    Reviewer feedback (must be addressed): {state['feedback']}"""
    taxonomy = llms.llm_generator.with_structured_output(Taxonomy).invoke(prompt)
    return {"taxonomy": taxonomy.model_dump()}


# ---------- Node 2: generate guideline (rules based on the taxonomy) ----------
def generate_guideline(state: AnnotationState) -> dict:
    print("\n>>> generating guideline ...")
    samples = "\n".join(f"- {item['text']}" for item in state["items"][:10])

    prompt = f"""You are the lead of an annotation project. The label taxonomy below is fixed.
Write annotation rules that tell annotators how to apply these labels,
especially how to handle edge cases and texts that could fit more than one label.
Do not add, remove, or rename labels.

Annotation request: {state['request']}

Taxonomy:
{taxonomy_to_text(state['taxonomy'])}

Sample data:
{samples}"""

    if state.get("feedback"):
        prompt += f"\n\nReviewer feedback (must be addressed): {state['feedback']}"

    guideline = llms.llm_generator.with_structured_output(Guideline).invoke(prompt)
    return {"guideline": guideline.model_dump()} 
 
# ---------- node 3: human review ----------
def human_review(state: AnnotationState) -> dict:
    header = ""
    if state.get("iteration", 0) > 0:
        header = (f"Round {state['iteration']}: kappa = {state['iaa_score']:.3f} "
                  f"(target {config.IAA_THRESHOLD}). Guideline has been revised:\n\n")

    answer = interrupt(header + full_instructions(state))

    if answer.strip().lower() == "y":
        return {"approved": True, "feedback": ""}
    return {"approved": False, "feedback": answer}
 
 
# ---------- node 4 & 5: annotators and judges ----------
def annotate_a(state: AnnotationState) -> dict:
    print("\n>>> annotator a is annotating ...")
    return {"annotations_a": annotate_with(llms.llm_a, state)}
 
 
def annotate_b(state: AnnotationState) -> dict:
    print("\n>>> annotator b is annotating ...")
    return {"annotations_b": annotate_with(llms.llm_b, state)}
 
 
# ---------- node 6: calculate IAA ----------
def calculate_iaa(state: AnnotationState) -> dict:
    ids = [item["id"] for item in state["items"]]
    labels_a = [state["annotations_a"][i]["label"] for i in ids]
    labels_b = [state["annotations_b"][i]["label"] for i in ids]

    # manually set kappa to 1 if two anntators' answers are the same. 
    kappa = cohen_kappa_score(labels_a, labels_b)
    if math.isnan(kappa):
        kappa = 1.0 if labels_a == labels_b else 0.0

    # get disagreements
    text_by_id = {item["id"]: item["text"] for item in state["items"]}
    disagreements = []
    for i in ids:
        a = state["annotations_a"][i]
        b = state["annotations_b"][i]
        if a["label"] != b["label"]:
            disagreements.append({
                "id": i,
                "text": text_by_id[i],
                "label_a": a["label"], "reason_a": a["reason"],
                "label_b": b["label"], "reason_b": b["reason"],
            })
 
    print(f"\n>>> Cohen's kappa = {kappa:.3f}, disagreement: {len(disagreements)}")
    
    iteration = state.get("iteration", 0) + 1
    history = state.get("iaa_history", []) + [{"round": iteration, "kappa": float(kappa)}]

    print(f"\n>>> Round {iteration}: Cohen's kappa = {kappa:.3f}, disagreements = {len(disagreements)}")
    return {
        "iaa_score": float(kappa),
        "disagreements": disagreements,
        "iteration": iteration,
        "iaa_history": history,
    }
 
 
# ---------- node 7: disagreement analysis ----------
def analyze_disagreements(state: AnnotationState) -> dict:
    disagreements = state["disagreements"]
    if not disagreements:
        return {"disagreement_report": "no disagreement."}
 
    # get the most frequent confusion pairs. 
    pair_counts = Counter(
        tuple(sorted([d["label_a"], d["label_b"]])) for d in disagreements
    )
    pair_text = "\n".join(f"- {a} vs {b}: {n} " for (a, b), n in pair_counts.most_common())
 
    examples = "\n".join(
        f"- 「{d['text']}」 A={d['label_a']}({d['reason_a']})  B={d['label_b']}({d['reason_b']})"
        for d in disagreements[:15]
    )
 
    # LLMs analyze the disagreements and provide advice.
    prompt = f"""
            Below is the disagreements of two annotators. 
            Analyze their disagreements and provide specific advice on modification. 
            Point out any vague definitions or rules in the annotation guideline.

            Current taxonomy and guideline:
            {full_instructions(state)}

            The most confusing labels:
            {pair_text}

            Disagreement samples:
            {examples}
            """
    report = llms.llm_generator.invoke(prompt).content
    print("\n>>> Analyzing disagreement: \n" + report)
    return {"disagreement_report": report}

# ---------- node 8: revise guideline (phase 2) ----------
def revise_guideline(state: AnnotationState) -> dict:
    print(f"\n>>> revising guideline after round {state['iteration']} ...")

    prompt = f"""
    You are the lead of an annotation project. Annotators disagreed too often
    (Cohen's kappa = {state['iaa_score']:.3f}, target = {config.IAA_THRESHOLD}).
    Revise the guideline to clarify the rules that caused the disagreements.
    The taxonomy is fixed: do not add, remove, or rename labels. Only change the rules.

    Annotation request: {state['request']}

    Taxonomy:
    {taxonomy_to_text(state['taxonomy'])}

    Current guideline:
    {guideline_to_text(state['guideline'])}

    Disagreement analysis:
    {state['disagreement_report']}
    """

    # If the reviewer rejected the previous revision, include their feedback
    if state.get("feedback"):
        prompt += f"\n\nReviewer feedback on the previous revision (must be addressed): {state['feedback']}"

    guideline = llms.llm_generator.with_structured_output(Guideline).invoke(prompt)
    return {"guideline": guideline.model_dump()}
 
 
# ---------- node 9: adjudicatation ----------
# ==== adjucation model generate a label when two annotators disagree ====
def adjudicate(state: AnnotationState) -> dict:
    print("\n>>> adjudicating ... ")
    valid_labels = [label["name"] for label in state["taxonomy"]["labels"]]
    structured_llm = llms.llm_judge.with_structured_output(Annotation)
    disagreement_ids = {d["id"] for d in state["disagreements"]}
 
    final_data = []
    for item in state["items"]:
        a = state["annotations_a"][item["id"]]
        b = state["annotations_b"][item["id"]]
        row = {
            "id": item["id"],
            "text": item["text"],
            "annotator_a": a["label"],
            "annotator_b": b["label"],
            "final_label": a["label"],  
            "adj_reason": "",    
            "adjudicated": False,
        }
 
        # Send disagreements to the judge model for adjudication.
        if item["id"] in disagreement_ids:
            prompt = f"""
            You are an adjudicator for a data annotation project.
            Two annotators assigned different labels to the same sample.
            Determine the correct label based on the annotation guidelines and taxonomy.

            Choose only from the following labels:
            {valid_labels}

            Annotation guidelines:
            {full_instructions(state)}

            Text:
            {item['text']}

            Annotator A:
            Label: {a['label']}
            Reason: {a['reason']}

            Annotator B:
            Label: {b['label']}
            Reason: {b['reason']}
            """
            try:
                output = structured_llm.invoke(prompt)
                label = output.label if output.label in valid_labels else "INVALID"
                aadj_reason = output.reason
            except Exception:
                label = "INVALID"
                adj_reason = "Failed to parse model output"
            row["final_label"] = label
            row["adjudicated"] = True
            row["adj_reason"] = output.reason
 
        final_data.append(row)
    return {"final_data": final_data}

# ---------- Node 10: collect gold labels from a human ----------
def collect_gold(state: AnnotationState) -> dict:
    # Fixed seed: this node re-runs from the top when resumed after interrupt(),
    # so the sample must be identical every time
    rng = random.Random(42)
    sample = rng.sample(state["items"], min(state["gold_size"], len(state["items"])))
    labels = [label["name"] for label in state["taxonomy"]["labels"]]

    # Pause and hand a structured payload to the terminal code in section 8
    gold = interrupt({
        "type": "gold_labeling",
        "labels": labels,
        "items": sample,
    })
    return {"gold_labels": gold}


# ---------- Node 11: compare model labels against gold labels ----------
def evaluate_gold(state: AnnotationState) -> dict:
    gold = state["gold_labels"]
    if not gold:
        print("\n>>> No gold labels collected, skipping evaluation.")
        return {"gold_metrics": {}}

    rows_by_id = {row["id"]: row for row in state["final_data"]}
    ids = list(gold.keys())

    def accuracy(predicted: dict) -> float:
        """predicted: {item_id: label}"""
        return sum(predicted[i] == gold[i] for i in ids) / len(ids)

    labels_a = {i: state["annotations_a"][i]["label"] for i in ids}
    labels_b = {i: state["annotations_b"][i]["label"] for i in ids}
    labels_final = {i: rows_by_id[i]["final_label"] for i in ids}

    y_true = [gold[i] for i in ids]
    y_pred = [labels_final[i] for i in ids]

    metrics = {
        "n_gold": len(ids),
        "accuracy_final": accuracy(labels_final),   # after adjudication
        "accuracy_a": accuracy(labels_a),
        "accuracy_b": accuracy(labels_b),
        "per_label": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    print(f"\n>>> Gold set evaluation on {len(ids)} items")
    print(f"    Annotator A accuracy: {metrics['accuracy_a']:.1%}")
    print(f"    Annotator B accuracy: {metrics['accuracy_b']:.1%}")
    print(f"    Final label accuracy: {metrics['accuracy_final']:.1%}")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Add a gold_label column to the exported data (empty for items you didn't label)
    final_data = [{**row, "gold_label": gold.get(row["id"], "")} for row in state["final_data"]]
    return {"gold_metrics": metrics, "final_data": final_data}
 
 
# ---------- node 12: Clean data ----------
def clean_data(state: AnnotationState) -> dict:
    cleaned = []
    seen_texts = set()
    dropped = Counter()

    for row in state["final_data"]:
        text = " ".join(row["text"].split())       # collapse extra whitespace
        if not text:
            dropped["empty_text"] += 1
        elif row["final_label"] == "INVALID":
            dropped["invalid_label"] += 1
        elif text.lower() in seen_texts:
            dropped["duplicate"] += 1
        else:
            seen_texts.add(text.lower())
            cleaned.append({**row, "text": text})

    print(f"\n>>> Kept {len(cleaned)} rows after cleaning, dropped: {dict(dropped)}")
    return {"final_data": cleaned}


# ---------- node 13: Export data ----------
def export_data(state: AnnotationState) -> dict:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    rows = state["final_data"]

    # JSON
    with open(out_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # CSV (utf-8-sig so Excel displays non-ASCII text correctly)
    if rows:
        with open(out_dir / "annotations.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Run metadata, useful for reviewing this run later
    meta = {
        "request": state["request"],
        "taxonomy": state["taxonomy"],
        "iaa_score": state["iaa_score"],
        "disagreement_report": state["disagreement_report"],
        "guideline": state["guideline"],
        "rounds": state["iteration"],
        "iaa_history": state["iaa_history"],
        "reached_threshold": state["iaa_score"] >= config.IAA_THRESHOLD,
        "gold_metrics": state.get("gold_metrics", {}),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n>>> Exported to {out_dir.resolve()}")
    return {}
