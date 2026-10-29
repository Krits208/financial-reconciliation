"""Demonstrate match -> simulated review -> refit using an explicit reference file.

Demo decisions are simulated, not real human approvals. Run human review through
the CLI's review.csv workflow instead. Demo files live under runs/demo by default.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recon.cli import main as cli
from recon.data import read_transactions, write_csv
from recon.evaluation import metrics, read_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", default="runs/demo")
    args = parser.parse_args()
    root = Path(args.output)
    state = root / "state.json"
    if state.exists():
        raise ValueError("Demo state already exists; choose a fresh --output directory")
    bank, reg = read_transactions("data/bank_statements.csv"), read_transactions("data/check_register.csv")
    truth = read_truth(args.truth, bank, reg)
    common = ["--state", str(state)]
    if cli(["match", *common, "--output", str(root / "before")]):
        return 2
    queue = list(csv.DictReader((root / "before/review.csv").open()))
    for row in queue:
        if row["method"] == "unique_amount":
            correct = truth[row["bank_id"]]
            row["decision"] = "accept" if correct == row["register_id"] else "correct"
            row["corrected_register_id"] = correct if row["decision"] == "correct" else ""
            row["comment"] = "SIMULATED review using the explicit provisional reference mapping"
    review = root / "simulated_review.csv"
    write_csv(review, queue)
    if cli(["review", *common, "--file", str(review)]):
        return 2
    if cli(["match", *common, "--output", str(root / "after")]):
        return 2
    after = list(csv.DictReader((root / "after/matches.csv").open()))
    remaining = {row["bank_id"]: truth[row["bank_id"]] for row in queue if not row["decision"]}
    result = {"review_type": "SIMULATED; not actual human validation",
              "reference": args.truth, "remaining_pairs": len(remaining),
              "before": metrics(queue, remaining), "after": metrics(after, remaining)}
    (root / "demo_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

