"""
Command-line entry point.

Usage:
    python main.py --data data/sample.csv
    python main.py --data data/sample.csv --gold-size 10
    python main.py --resume <run_id>
"""
import argparse
import csv
import json
import sqlite3
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from annotation_agent import config
from annotation_agent.graph import build_graph

DEFAULT_REQUEST = "Classify the intent of e-commerce customer feedback for automatic support ticket routing"


def load_items(path: str) -> list[dict]:
    """Load data from a .csv or .jsonl file. Each row needs a 'text' field; 'id' is optional."""
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("Only .csv and .jsonl files are supported")

    items = []
    for i, row in enumerate(rows, start=1):
        if "text" not in row:
            raise ValueError(f"Row {i} has no 'text' field")
        text = str(row["text"]).strip()
        if not text:
            continue
        item_id = str(row.get("id") or i)
        items.append({"id": item_id, "text": text})

    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids found in the data file")

    print(f">>> Loaded {len(items)} items from {path}")
    return items


def run_and_show(graph, graph_input, run_config):
    """Run the graph until it finishes or pauses. Return the interrupt value if paused, else None."""
    interrupt_value = None
    for update in graph.stream(graph_input, run_config, stream_mode="updates"):
        for node_name, output in update.items():
            if node_name == "__interrupt__":
                interrupt_value = output[0].value
            else:
                keys = list(output.keys()) if output else []
                print(f"[DONE] {node_name} -> updated: {keys}")
    return interrupt_value


def ask_gold_labels(payload: dict) -> dict:
    """Ask the human to label each sampled item in the terminal."""
    labels = payload["labels"]
    print("\n========== Gold labeling ==========")
    print("Label each text yourself, following the approved guideline.")
    for n, name in enumerate(labels, start=1):
        print(f"  {n}. {name}")

    gold = {}
    for item in payload["items"]:
        while True:
            answer = input(f"\n[{item['id']}] {item['text']}\nLabel number (s = skip): ").strip()
            if answer.lower() == "s":
                break
            if answer.isdigit() and 1 <= int(answer) <= len(labels):
                gold[item["id"]] = labels[int(answer) - 1]
                break
            print("Invalid input, try again.")
    return gold


def review_loop(graph, pending, run_config):
    """Handle every pause until the graph finishes."""
    while pending is not None:
        if isinstance(pending, dict) and pending.get("type") == "gold_labeling":
            answer = ask_gold_labels(pending)
        else:
            print("\n========== Please review ==========")
            print(pending)
            answer = input("\nType y to approve, or enter your revision feedback: ")
        pending = run_and_show(graph, Command(resume=answer), run_config)


def main():
    parser = argparse.ArgumentParser(description="LLM annotation pipeline")
    parser.add_argument("--data", help="Path to a .csv or .jsonl file with a 'text' column")
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="What the annotation is for")
    parser.add_argument("--resume", help="Run ID of an interrupted run to continue")
    parser.add_argument("--gold-size", type=int, default=0,
                        help="Number of items to label by hand for evaluation (0 = skip)")
    args = parser.parse_args()

    conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
    graph = build_graph(SqliteSaver(conn))

    # ---------- Resume an existing run ----------
    if args.resume:
        run_config = {"configurable": {"thread_id": args.resume}}
        snapshot = graph.get_state(run_config)

        if not snapshot.values:
            print(f"No run found with ID {args.resume}")
            return
        if not snapshot.next:
            print("This run has already finished.")
            return

        waiting = [i.value for task in snapshot.tasks for i in task.interrupts]
        if waiting:
            pending = waiting[0]
        else:
            print(f">>> Resuming from: {snapshot.next}")
            pending = run_and_show(graph, None, run_config)

    # ---------- Start a new run ----------
    else:
        if not args.data:
            parser.error("--data is required for a new run")
        items = load_items(args.data)
        run_id = uuid.uuid4().hex[:8]
        run_config = {"configurable": {"thread_id": run_id}}
        print(f"\n>>> Run ID: {run_id}")
        print(f">>> If interrupted, continue with: python main.py --resume {run_id}\n")
        pending = run_and_show(
            graph,
            {"request": args.request, "items": items, "gold_size": args.gold_size},
            run_config,
        )

    review_loop(graph, pending, run_config)

    final_state = graph.get_state(run_config).values
    print(f"\nAll done! Rounds: {final_state['iteration']}, final kappa: {final_state['iaa_score']:.3f}")


if __name__ == "__main__":
    main()
