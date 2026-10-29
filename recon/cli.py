"""CLI entry points for reconciliation, human review, and evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from .data import format_cents, read_transactions, write_csv
from .engine import Config, reconcile
from .model import PairModel
from .review import apply_review, load_state, save_state, training_pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Learned financial reconciliation")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("match", "review"):
        p = sub.add_parser(command)
        p.add_argument("--bank", default="data/bank_statements.csv")
        p.add_argument("--register", default="data/check_register.csv")
        p.add_argument("--state", default="runs/state.json")
        if command == "match":
            p.add_argument("--output", default="runs/latest")
            p.add_argument("--mode", choices=["learned", "numeric", "lexical"], default="learned")
            p.add_argument("--amount-tolerance-cents", type=int, default=10)
            p.add_argument("--candidate-window-days", type=int, default=45)
        else:
            p.add_argument("--file", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--bank", default="data/bank_statements.csv")
    evaluate.add_argument("--register", default="data/check_register.csv")
    evaluate.add_argument("--truth", required=True, help="Explicit evaluation reference CSV with bank_id,register_id columns")
    evaluate.add_argument("--output", default="reports")
    args = parser.parse_args(argv)
    try:
        bank, register = read_transactions(args.bank), read_transactions(args.register)
        if args.command == "evaluate":
            from .evaluation import run_evaluation
            run_evaluation(bank, register, args.truth, args.output)
            return 0
        state = load_state(args.state, bank, register)
        if args.command == "review":
            updated = apply_review(state, args.file, bank, register)
            save_state(args.state, updated)
            print(json.dumps({"validated_pairs": len(updated["accepted"]), "revision": updated["revision"],
                              "next_step": "Run match again to refit from validated pairs."}, indent=2))
            return 0
        model = PairModel().fit(training_pairs(state, bank, register))
        config = Config(amount_tolerance_cents=args.amount_tolerance_cents,
                        candidate_window_days=args.candidate_window_days)
        matches = reconcile(bank, register, model, state["accepted"],
                            {tuple(pair) for pair in state["rejected"]}, config, args.mode)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "matches.csv", matches)
        queue = [{**row, "decision": "", "corrected_register_id": "", "comment": "",
                  "dataset_fingerprint": state["dataset_fingerprint"], "state_revision": state["revision"]}
                 for row in matches]  # Validated rows remain available for later correction.
        write_csv(output / "review.csv", queue)
        matched_ids = {row["register_id"] for row in matches if row["register_id"]}
        unmatched = [{"register_id": r.id, "date": r.date.isoformat(), "description": r.description,
                      "amount": format_cents(r.cents)} for r in register if r.id not in matched_ids]
        write_csv(output / "unmatched_register.csv", unmatched, ["register_id", "date", "description", "amount"])
        summary = {"bank_rows": len(bank), "register_rows": len(register),
                   "training_pairs": model.training_pairs, "latent_model_fitted": model.svd is not None,
                   "status_counts": dict(Counter(row["status"] for row in matches)),
                   "method_counts": dict(Counter(row["method"] for row in matches)),
                   "unmatched_register_rows": len(unmatched),
                   "confidence_note": "Scores rank evidence; they are not calibrated probabilities.",
                   "review_file": str(output / "review.csv")}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        save_state(args.state, state)
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0

