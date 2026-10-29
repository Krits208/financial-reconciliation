"""Atomic, auditable review state. Predictions never train the model by themselves."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .data import Transaction, fingerprint


def load_state(path: str | Path, bank: list[Transaction], register: list[Transaction]) -> dict:
    path = Path(path)
    current = fingerprint(bank, register)
    if not path.exists():
        return {"version": 1, "dataset_fingerprint": current, "revision": 0,
                "accepted": {}, "rejected": [], "events": []}
    state = json.loads(path.read_text())
    if state.get("version") != 1 or state.get("dataset_fingerprint") != current:
        raise ValueError("Review state belongs to a different dataset/version; use a separate state file")
    return state


def save_state(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def training_pairs(state: dict, bank: list[Transaction], register: list[Transaction]):
    bm, rm = {t.id: t for t in bank}, {t.id: t for t in register}
    return [(bm[b], rm[r]) for b, r in sorted(state["accepted"].items())]


def apply_review(state: dict, path: str | Path, bank: list[Transaction], register: list[Transaction]) -> dict:
    """Validate a whole review batch before applying it; allow corrections and swaps."""
    bm, rm = {t.id: t for t in bank}, {t.id: t for t in register}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    accepted = dict(state["accepted"])
    rejected = {tuple(pair) for pair in state["rejected"]}
    actions, seen = [], set()
    for row in rows:
        decision = (row.get("decision") or "").strip().lower()
        if not decision:
            continue
        if row.get("dataset_fingerprint") != state["dataset_fingerprint"] or row.get("state_revision") != str(state["revision"]):
            raise ValueError("Stale review file or wrong dataset; regenerate the review queue")
        if decision not in {"accept", "reject", "correct"}:
            raise ValueError(f"Unknown review decision: {decision}")
        bid, proposed = row.get("bank_id", "").strip(), row.get("register_id", "").strip()
        if bid not in bm or bid in seen:
            raise ValueError(f"Unknown or repeated bank ID in review: {bid}")
        seen.add(bid)
        if proposed and proposed not in rm:
            raise ValueError(f"Unknown proposed register ID: {proposed}")
        target = (row.get("corrected_register_id") or "").strip() if decision == "correct" else proposed
        if decision in {"accept", "correct"} and target not in rm:
            raise ValueError(f"Unknown/missing register ID: {target}")
        if decision == "reject" and not proposed:
            raise ValueError("Cannot reject an empty proposal")
        if decision in {"accept", "correct"} and bm[bid].direction != rm[target].direction:
            raise ValueError("Cannot validate a pair with opposite transaction directions")
        previous = accepted.pop(bid, None)
        if previous and previous != target:
            rejected.add((bid, previous))
        if decision == "reject" or (decision == "correct" and proposed and proposed != target):
            rejected.add((bid, proposed))
        actions.append((bid, proposed, target, decision, row.get("comment", "")))
    for bid, _, target, decision, _ in actions:
        if decision != "reject":
            accepted[bid] = target
            rejected.discard((bid, target))
    if len(set(accepted.values())) != len(accepted):
        raise ValueError("Review would reuse a register transaction; correct conflicting rows together")
    updated = json.loads(json.dumps(state))
    if actions:
        updated["revision"] += 1
        updated["accepted"] = accepted
        updated["rejected"] = [list(pair) for pair in sorted(rejected)]
        timestamp = datetime.now(timezone.utc).isoformat()
        for bid, proposed, target, decision, comment in actions:
            updated["events"].append({"time": timestamp, "revision": updated["revision"],
                                       "bank_id": bid, "proposed_register_id": proposed,
                                       "target_register_id": target, "decision": decision, "comment": comment})
    return updated

