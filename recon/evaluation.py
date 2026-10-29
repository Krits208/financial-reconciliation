"""Reproducible baselines and held-out learning curves; never imported by matching."""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import random
import statistics

from .data import Transaction, write_csv
from .engine import reconcile
from .model import PairModel


def read_truth(path, bank, register) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"bank_id", "register_id"} <= set(reader.fieldnames or []):
            raise ValueError("Truth CSV requires bank_id,register_id columns")
        rows = list(reader)
    truth = {row["bank_id"]: row["register_id"] for row in rows}
    if len(truth) != len(rows) or len(set(truth.values())) != len(rows):
        raise ValueError("Reference mapping must be one-to-one")
    if set(truth) != {t.id for t in bank} or set(truth.values()) != {t.id for t in register}:
        raise ValueError("This evaluator requires a complete one-to-one reference mapping")
    return truth


def metrics(rows: list[dict], truth: dict[str, str], confident_only: bool = True) -> dict:
    selected = [r for r in rows if r["bank_id"] in truth and r["register_id"]
                and (not confident_only or r["status"] == "proposed")]
    if len({r["bank_id"] for r in selected}) != len(selected) or len({r["register_id"] for r in selected}) != len(selected):
        raise ValueError("Cannot score a non-one-to-one result")
    correct = sum(truth[r["bank_id"]] == r["register_id"] for r in selected)
    precision = correct / len(selected) if selected else 0.0
    recall = correct / len(truth) if truth else 0.0
    return {"correct": correct, "matches": len(selected), "total": len(truth),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "coverage": len(selected) / len(truth) if truth else 0.0}


def collision_fixture(pairs: list[tuple[Transaction, Transaction]], seed: int):
    """Remove amount/date shortcuts in groups of four; retain held-out descriptions.

    Every group has the same direction, amount and date lag. Source IDs are
    independently randomized. This controlled stress test is not natural data.
    Repeated semantic categories are retained, including irreducibly ambiguous cases.
    """
    rng = random.Random(seed)
    grouped = {"in": [], "out": []}
    for pair in pairs:
        grouped[pair[0].direction].append(pair)
    bank, register, truth = [], [], {}
    group_number = 0
    for direction_pairs in grouped.values():
        rng.shuffle(direction_pairs)
        for start in range(0, len(direction_pairs), 4):
            group = direction_pairs[start:start+4]
            anchor = date(2024, 1, 1) + timedelta(days=group_number * 10)
            amount = 10000 + group_number * 10000
            for b, r in group:
                bid, rid = f"b-{rng.getrandbits(96):024x}", f"r-{rng.getrandbits(96):024x}"
                bank.append(replace(b, id=bid, cents=amount, date=anchor + timedelta(days=2)))
                register.append(replace(r, id=rid, cents=amount, date=anchor))
                truth[bid] = rid
            group_number += 1
    rng.shuffle(bank)
    rng.shuffle(register)
    return bank, register, truth


def run_evaluation(bank, register, truth_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    truth = read_truth(truth_path, bank, register)
    bm, rm = {t.id: t for t in bank}, {t.id: t for t in register}
    all_pairs = [(bm[bid], rm[rid]) for bid, rid in sorted(truth.items())]
    bc, rc = Counter(t.cents for t in bank), Counter(t.cents for t in register)
    hard_ids = {t.id for t in bank if bc[t.cents] != 1 or rc[t.cents] != 1}
    audit = {
        "bank_rows": len(bank), "register_rows": len(register),
        "unique_exact_amount_candidate_rows": len(bank) - len(hard_ids),
        "remaining_rows": len(hard_ids), "reference_path": str(truth_path),
        "reference_warning": "If using reference_inferred.csv, labels are inferred from synthetic ID suffixes, not independently verified ground truth.",
        "date_gap_counts": dict(sorted(Counter((b.date-r.date).days for b, r in all_pairs).items())),
        "amount_difference_cents_counts": dict(sorted(Counter(abs(b.cents-r.cents) for b, r in all_pairs).items())),
    }
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    full_rows = []
    numeric = reconcile(bank, register, PairModel(), mode="numeric")
    unique = [r for r in numeric if r["method"] == "unique_amount"]
    cold = reconcile(bank, register, PairModel())
    # Simulate review of correct unique-amount proposals, then score ONLY remaining rows.
    # Seed quality is resolved by the supplied reference in this evaluation, not assumed.
    seed_pairs = [(bm[r["bank_id"]], rm[r["register_id"]]) for r in unique
                  if truth[r["bank_id"]] == r["register_id"]]
    validated = {b.id: r.id for b, r in seed_pairs}
    warm_model = PairModel().fit(seed_pairs)
    warm = reconcile(bank, register, warm_model, validated=validated)
    write_csv(output / "initial_predictions.csv", cold)
    write_csv(output / "after_simulated_review.csv", warm)
    for label, rows, subset in [
        ("unique_amount_only", unique, truth), ("numeric_baseline_all", numeric, truth),
        ("cold_hybrid_all", cold, truth),
        ("numeric_baseline_remaining", numeric, {b: truth[b] for b in hard_ids}),
        ("cold_hybrid_remaining", cold, {b: truth[b] for b in hard_ids}),
        ("after_seed_review_remaining", warm, {b: truth[b] for b in hard_ids}),
    ]:
        full_rows.append({"experiment": label, **metrics(rows, subset)})
    write_csv(output / "supplied_data_metrics.csv", full_rows)

    curves, errors, categories = [], [], []
    for seed in (13, 42, 97):
        shuffled = list(all_pairs)
        random.Random(seed).shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * .30))
        test, train = shuffled[:test_count], shuffled[test_count:]
        test_b, test_r = [b for b, _ in test], [r for _, r in test]
        test_truth = {b.id: r.id for b, r in test}
        stress_b, stress_r, stress_truth = collision_fixture(test, seed)
        for size in sorted({0, min(8, len(train)), min(32, len(train)), min(64, len(train)), min(128, len(train)), len(train)}):
            model = PairModel().fit(train[:size])
            for dataset, bs, rs, reference in (
                ("original_holdout", test_b, test_r, test_truth),
                ("collision_stress", stress_b, stress_r, stress_truth),
            ):
                for mode in ("numeric", "lexical", "learned"):
                    predictions = reconcile(bs, rs, model, mode=mode, unique_stage=False)
                    scores = metrics(predictions, reference)
                    all_scores = metrics(predictions, reference, confident_only=False)
                    curves.append({"seed": seed, "dataset": dataset, "training_pairs": size, "mode": mode,
                                   **scores, "all_proposals_precision": all_scores["precision"],
                                   "all_proposals_recall": all_scores["recall"]})
                    if mode == "learned" and size == len(train):
                        register_lookup = {r.id: r for r in rs}
                        for category in sorted({r.category or "Uncategorized" for r in rs}):
                            category_truth = {bid: rid for bid, rid in reference.items()
                                              if (register_lookup[rid].category or "Uncategorized") == category}
                            categories.append({"seed": seed, "dataset": dataset, "category": category,
                                               **metrics(predictions, category_truth)})
                        for row in predictions:
                            if row["register_id"] != reference[row["bank_id"]] or row["status"] != "proposed":
                                expected = register_lookup[reference[row["bank_id"]]]
                                errors.append({"seed": seed, "dataset": dataset,
                                               "expected_register_id": expected.id,
                                               "expected_register_description": expected.description,
                                               "expected_category": expected.category, **row})
        print(f"Evaluated seed {seed}: {len(train)} training pool / {len(test)} held-out pairs", flush=True)
    write_csv(output / "learning_curve.csv", curves)
    write_csv(output / "category_metrics.csv", categories)
    if errors:
        write_csv(output / "errors_and_review_cases.csv", errors)
    else:
        write_csv(output / "errors_and_review_cases.csv", [], ["seed", "dataset", "expected_register_id", "bank_id", "register_id", "status"])
    summaries = []
    keys = sorted({(r["dataset"], r["training_pairs"], r["mode"]) for r in curves})
    for dataset, size, mode in keys:
        runs = [r for r in curves if (r["dataset"], r["training_pairs"], r["mode"]) == (dataset, size, mode)]
        summary = {"dataset": dataset, "training_pairs": size, "mode": mode, "runs": len(runs)}
        for metric in ("precision", "recall", "f1", "coverage", "all_proposals_precision"):
            summary[metric + "_mean"] = statistics.mean(r[metric] for r in runs)
            summary[metric + "_std"] = statistics.stdev(r[metric] for r in runs) if len(runs) > 1 else 0.0
        summaries.append(summary)
    write_csv(output / "learning_curve_summary.csv", summaries)
    _write_analysis(output, audit, full_rows, summaries, errors, categories)
    _plot(output, summaries)
    print(f"Reports written to {output}")


def _write_analysis(output, audit, metrics_rows, summaries, errors, categories):
    lines = ["# Evaluation report", "", "Reproduce these results with `python -m recon evaluate --truth data/reference_inferred.csv`.", "",
             "## Reference-label limitation", "", audit["reference_warning"],
             "All supplied-data metrics and learning curves using that file are conditional on this mapping being correct. "
             "The stress fixture inherits those pairings. The matching code never reads the reference file or parses ID suffixes.", "",
             "## Supplied data", "", f"{audit['bank_rows']} rows per source; {audit['unique_exact_amount_candidate_rows']} unique exact-amount candidates; {audit['remaining_rows']} remaining rows.", "",
             "Metrics count only confident machine proposals (`status=proposed`), before human corrections. "
             "Review-required/unmatched rows count against recall. Previously reviewed training pairs are excluded from remaining-row and holdout metrics.", "",
             "| Experiment | Correct / proposed | Eligible pairs | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for r in metrics_rows:
        lines.append(f"| {r['experiment']} | {r['correct']} / {r['matches']} | {r['total']} | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} |")
    lines += ["", "## Learning curves", "", "Three fixed random seeds (13, 42, 97), each with a 70% training pool and a fixed 30% holdout. "
              "Nested training subsets simulate increasing reviewed history. Neither held-out pair labels nor their rows train the paired SVD. "
              "The unique-amount stage is disabled throughout these curves to isolate second-stage behavior. "
              "The lexical fallback can use unlabelled candidate text for TF-IDF; this is a transductive retrieval setup. "
              "Training examples represent correct simulated human reviews, including corrections, rather than self-approved predictions.", "",
              "`collision_stress` modifies only held-out rows: groups of up to four same-direction transactions receive the same amount and dates. "
              "Descriptions remain unchanged, and IDs and ordering are independently randomized. "
              "This deliberately removes numeric shortcuts; it is a controlled stress test, not an estimate of production performance. "
              "Similar-category duplicates are retained and can remain ambiguous.", "",
              "| Dataset | Training pairs | Method | Precision mean | Recall mean | F1 mean ± sample SD |", "|---|---:|---|---:|---:|---:|"]
    for r in summaries:
        if r["mode"] == "learned" or r["training_pairs"] == 0:
            lines.append(f"| {r['dataset']} | {r['training_pairs']} | {r['mode']} | {r['precision_mean']:.1%} | {r['recall_mean']:.1%} | {r['f1_mean']:.1%} ± {r['f1_std']:.1%} |")
    flags = Counter(w.split(":")[0] for r in errors for w in r["warnings"].split("|") if w)
    lines += ["", "## Error analysis and interpretation", "",
              f"At maximum training size, {len(errors)} held-out outputs across both datasets and all three seeds are incorrect or require review. "
              f"Warning counts (can overlap): {dict(flags)}. See `errors_and_review_cases.csv` for every such output.", "",
              "Category results below aggregate the three collision-stress runs at 216 training pairs. Categories describe the expected register transaction; repeated holdout appearances across seeds are not independent new observations.", "",
              "| Expected category | Correct / confident proposals | Evaluated pairs across runs | Precision | Recall |", "|---|---:|---:|---:|---:|"]
    category_totals = {}
    for row in categories:
        if row["dataset"] == "collision_stress":
            totals = category_totals.setdefault(row["category"], Counter())
            for key in ("correct", "matches", "total"):
                totals[key] += row[key]
    for category, totals in sorted(category_totals.items(), key=lambda item: item[1]["correct"] / max(1, item[1]["total"])):
        precision = totals["correct"] / totals["matches"] if totals["matches"] else 0
        recall = totals["correct"] / totals["total"] if totals["total"] else 0
        lines.append(f"| {category} | {totals['correct']} / {totals['matches']} | {totals['total']} | {precision:.1%} | {recall:.1%} |")
    lines += ["", "Concrete supplied-data cases: `B0296` / `R0296` has `CHECK #642` versus `Check #6442`, so even its unique amount triggers review. "
              "`ONLINE PMT WATER` versus `Electric bill`, and `GYM MEMBERSHIP` versus `Music subscription`, illustrate conflicting semantics under the provisional reference. "
              "In the stress fixture, one confident grocery assignment is wrong at maximum training size. The error CSV includes both the proposed and expected descriptions so that such ambiguity can be inspected.", "",
              "The original fixture has strong numerical shortcuts, so a numeric baseline may equal or beat the hybrid. "
              "The stress results test whether source-specific text relationships help when these shortcuts disappear. "
              "More training need not improve every metric monotonically: SVD refitting changes the latent space, and confidence/ambiguity thresholds affect coverage.", "",
              "Precision is defined as zero when no confident matches are proposed; this convention must not be confused with observed incorrect matches. "
              "`all_proposals_precision` separately evaluates every assigned pair, including review-required cases, and does not represent automated acceptance accuracy.", "",
              "## Limits", "", "Only 308 synthetic pairs; provisional reference labels; no independent production benchmark; three seeds are descriptive rather than a statistical significance claim. "
              "Random holdout measures within-distribution learning, not future-time generalization. "
              "Weights and confidence thresholds are documented engineering defaults, not calibrated probabilities or parameters optimized on this holdout. "
              "Candidate tolerances can exclude true anomalies. No split/merged transactions, FX conversion, account grouping or multi-currency handling is implemented.", ""]
    (output / "ANALYSIS.md").write_text("\n".join(lines))


def _plot(output, summaries):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Optional matplotlib not installed; CSV/Markdown reports are complete.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, dataset, title in zip(axes, ("original_holdout", "collision_stress"),
                                  ("Original held-out transactions", "Controlled amount/date collisions")):
        for mode, color in (("numeric", "#64748b"), ("lexical", "#d97706"), ("learned", "#2563eb")):
            rows = [r for r in summaries if r["dataset"] == dataset and r["mode"] == mode]
            ax.plot([r["training_pairs"] for r in rows], [r["f1_mean"] for r in rows], marker="o", label=mode, color=color)
        ax.set(title=title, xlabel="Validated training pairs", ylim=(-.03, 1.04))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("F1 of confident proposals (mean of 3 seeds)")
    fig.savefig(output / "learning_curve.png", dpi=160)
    plt.close(fig)

