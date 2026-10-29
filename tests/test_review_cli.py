import csv
from datetime import date
import json

import pytest

from recon.cli import main
from recon.data import Transaction, write_csv
from recon.review import apply_review, load_state, save_state, training_pairs


@pytest.fixture
def sources():
    def t(key, cents):
        return Transaction(key, date(2024, 1, 1), "purchase", cents, "out")
    return [t("b1", 100), t("b2", 200)], [t("r1", 100), t("r2", 200)]


def review_file(tmp_path, state, actions):
    rows = [{"bank_id": b, "register_id": r, "decision": decision, "corrected_register_id": target,
             "dataset_fingerprint": state["dataset_fingerprint"], "state_revision": state["revision"]}
            for b, r, decision, target in actions]
    path = tmp_path / "review.csv"
    write_csv(path, rows)
    return path


def test_review_accepts_then_rejects_without_training_predictions(tmp_path, sources):
    bank, reg = sources
    state = load_state(tmp_path / "state.json", bank, reg)
    assert training_pairs(state, bank, reg) == []
    path = review_file(tmp_path, state, [("b1", "r1", "accept", "")])
    accepted = apply_review(state, path, bank, reg)
    assert len(training_pairs(accepted, bank, reg)) == 1
    assert state["accepted"] == {}  # Input state was not mutated.
    path = review_file(tmp_path, accepted, [("b1", "r1", "reject", "")])
    rejected = apply_review(accepted, path, bank, reg)
    assert rejected["accepted"] == {}
    assert ["b1", "r1"] in rejected["rejected"]


def test_atomic_review_rejects_duplicate_targets(tmp_path, sources):
    bank, reg = sources
    state = load_state(tmp_path / "state.json", bank, reg)
    path = review_file(tmp_path, state, [("b1", "r1", "accept", ""), ("b2", "r2", "correct", "r1")])
    with pytest.raises(ValueError, match="reuse"):
        apply_review(state, path, bank, reg)
    assert state["revision"] == 0 and state["accepted"] == {}


def test_atomic_swap_correction(tmp_path, sources):
    bank, reg = sources
    state = load_state(tmp_path / "state.json", bank, reg)
    state["accepted"] = {"b1": "r2", "b2": "r1"}
    path = review_file(tmp_path, state, [("b1", "r2", "correct", "r1"), ("b2", "r1", "correct", "r2")])
    updated = apply_review(state, path, bank, reg)
    assert updated["accepted"] == {"b1": "r1", "b2": "r2"}


def test_stale_queue_is_rejected(tmp_path, sources):
    bank, reg = sources
    state = load_state(tmp_path / "state.json", bank, reg)
    path = review_file(tmp_path, state, [("b1", "r1", "accept", "")])
    updated = apply_review(state, path, bank, reg)
    with pytest.raises(ValueError, match="Stale"):
        apply_review(updated, path, bank, reg)


def test_state_content_binding_ignores_row_order(tmp_path, sources):
    bank, reg = sources
    path = tmp_path / "state.json"
    save_state(path, load_state(path, bank, reg))
    assert load_state(path, bank[::-1], reg)["revision"] == 0
    with pytest.raises(ValueError, match="different dataset"):
        load_state(path, bank[:1], reg)


def test_cli_match_review_match(tmp_path):
    bank, reg = tmp_path / "bank.csv", tmp_path / "reg.csv"
    bank_rows, reg_rows = [], []
    for i, (btext, rtext) in enumerate([("BP", "fuel"), ("Kroger", "grocery"), ("ATM", "cash"), ("Cafe", "lunch")]):
        common = {"date": "2024-01-01", "amount": str(i+1), "type": "DR"}
        bank_rows.append({**common, "transaction_id": f"b{i}", "description": btext})
        reg_rows.append({**common, "transaction_id": f"r{i}", "description": rtext})
    write_csv(bank, bank_rows)
    write_csv(reg, reg_rows)
    state, output = tmp_path / "state.json", tmp_path / "out"
    args = ["--bank", str(bank), "--register", str(reg), "--state", str(state)]
    assert main(["match", *args, "--output", str(output)]) == 0
    queue = list(csv.DictReader((output / "review.csv").open()))
    for row in queue:
        row["decision"] = "accept"
    review = tmp_path / "edited.csv"
    write_csv(review, queue)
    assert main(["review", *args, "--file", str(review)]) == 0
    assert main(["match", *args, "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["training_pairs"] == 4
    assert summary["latent_model_fitted"] is True
    assert summary["status_counts"] == {"validated": 4}

