from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from recon.data import Transaction, format_cents, money_to_cents, read_transactions, write_csv
from recon.engine import Config, reconcile
from recon.evaluation import collision_fixture, metrics, read_truth
from recon.model import PairModel


def tx(key, text="purchase", cents=1000, day=2, direction="out"):
    return Transaction(key, date(2024, 1, 1) + timedelta(days=day), text, cents, direction)


@pytest.mark.parametrize("value,expected", [("0", 0), ("86.4", 8640), ("0.01", 1), ("147.330", 14733)])
def test_exact_money(value, expected):
    assert money_to_cents(value) == expected


@pytest.mark.parametrize("value", ["nan", "Infinity", "-1.00", "1.001", "bad", ""])
def test_invalid_money(value):
    with pytest.raises(ValueError):
        money_to_cents(value)


def test_csv_validates_duplicate_ids(tmp_path):
    path = tmp_path / "data.csv"
    row = {"transaction_id": "a", "date": "2024-01-01", "description": "p", "amount": "1", "type": "DR"}
    write_csv(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        read_transactions(path)


def test_unique_amount_and_date_warning():
    rows = reconcile([tx("b", day=20)], [tx("r", day=0)], PairModel())
    assert rows[0]["method"] == "unique_amount"
    assert rows[0]["status"] == "review_required"
    assert "unexpected_date_gap" in rows[0]["warnings"]


def test_unique_amount_means_unique_in_entire_input():
    bank, reg = [tx("b1"), tx("b2")], [tx("r1"), tx("r2")]
    result = reconcile(bank, reg, PairModel(), validated={"b1": "r1"})
    assert result[1]["method"] != "unique_amount"


def test_direction_mismatch_never_auto_matches():
    result = reconcile([tx("b")], [tx("r", direction="in")], PairModel())
    assert result[0]["status"] == "unmatched"


def test_global_assignment_avoids_greedy_collision():
    class Scores(PairModel):
        def similarities(self, bank, register, mode="learned"):
            return np.array([[1.0, .9], [.8, 0.0]])
    result = reconcile([tx("b1"), tx("b2")], [tx("r1", day=0), tx("r2", day=0)], Scores(), unique_stage=False)
    assert {row["bank_id"]: row["register_id"] for row in result} == {"b1": "r2", "b2": "r1"}
    assert any(row["status"] == "review_required" for row in result)


def test_identical_candidates_require_review():
    result = reconcile([tx("b1"), tx("b2")], [tx("r1", day=0), tx("r2", day=0)], PairModel())
    assert all(row["status"] == "review_required" for row in result)
    assert all("ambiguous_alternatives" in row["warnings"] for row in result)


def test_rejected_pair_is_not_reproposed_or_used_as_unique_seed():
    result = reconcile([tx("b")], [tx("r")], PairModel(), rejected={("b", "r")})
    assert result[0]["status"] == "unmatched"


def test_abstention_and_unequal_input_sizes():
    result = reconcile([tx("b1"), tx("b2", cents=99999)], [tx("r1")], PairModel())
    assert result[1]["status"] == "unmatched"
    assert reconcile([tx("b")], [], PairModel())[0]["status"] == "unmatched"
    assert reconcile([], [tx("r")], PairModel()) == []


def test_rounding_candidate_and_reference_conflict():
    result = reconcile([tx("b", text="CHECK #22", cents=1003)], [tx("r", text="Check #23", day=0)], PairModel(), unique_stage=False)
    assert result[0]["status"] != "proposed"
    assert tx("x", "CK #0022").check_reference == "22"


def test_shared_space_learns_cross_source_words():
    pairs = []
    for i in range(18):
        bank_word, register_word = [("alpine", "tulip"), ("cedar", "violet"), ("juniper", "orchid")][i % 3]
        pairs.append((tx(f"b{i}", bank_word), tx(f"r{i}", register_word, day=0)))
    model = PairModel(components=2).fit(pairs)
    similarities = model.similarities([tx("new-b", "alpine")], [tx("x", "tulip"), tx("y", "violet"), tx("z", "orchid")])
    assert model.svd is not None
    assert int(similarities.argmax()) == 0
    assert model.training_pairs == 18


def test_ids_do_not_enter_model_features():
    pairs = [(tx("b1", "fuel"), tx("r1", "petrol")),
             (tx("b2", "grocery"), tx("r2", "food")),
             (tx("b3", "transfer"), tx("r3", "savings"))]
    model = PairModel(components=2).fit(pairs)
    bs, rs = [b for b, _ in pairs], [r for _, r in pairs]
    original = model.similarities(bs, rs)
    renamed = model.similarities([replace(t, id=f"arbitrary{i}") for i, t in enumerate(bs)],
                                [replace(t, id=f"unrelated{i}") for i, t in enumerate(rs)])
    np.testing.assert_allclose(original, renamed)


def test_metrics_count_abstention_as_missed_recall():
    rows = [dict(bank_id="b1", register_id="r1", status="proposed"),
            dict(bank_id="b2", register_id="r3", status="proposed"),
            dict(bank_id="b3", register_id="r2", status="review_required")]
    result = metrics(rows, {"b1": "r1", "b2": "r2", "b3": "r3"})
    assert result["precision"] == .5
    assert result["recall"] == 1/3
    assert result["f1"] == pytest.approx(.4)


def test_reference_mapping_rejects_duplicate_targets(tmp_path):
    path = tmp_path / "reference.csv"
    write_csv(path, [{"bank_id": "b1", "register_id": "r1"}, {"bank_id": "b2", "register_id": "r1"}])
    with pytest.raises(ValueError, match="one-to-one"):
        read_truth(path, [tx("b1"), tx("b2")], [tx("r1"), tx("r2")])


def test_stress_fixture_preserves_pairing_and_shuffles_ids():
    pairs = [(tx(f"b{i}", f"bank desc {i}"), tx(f"r{i}", f"reg desc {i}")) for i in range(8)]
    bank, reg, truth = collision_fixture(pairs, 42)
    rm = {r.id: r for r in reg}
    assert len(set(truth.values())) == 8
    for b in bank:
        r = rm[truth[b.id]]
        assert b.description.split()[-1] == r.description.split()[-1]
        assert b.cents == r.cents and (b.date-r.date).days == 2
        assert b.id[2:] != r.id[2:]


def test_invalid_config():
    with pytest.raises(ValueError):
        Config(amount_tolerance_cents=-1)


def test_large_money_output_preserves_every_cent():
    cents = 9007199254740999
    assert format_cents(cents) == "90071992547409.99"
    result = reconcile([tx("b", cents=cents)], [tx("r", cents=cents)], PairModel())
    assert result[0]["bank_amount"] == "90071992547409.99"


def test_out_of_window_rounding_pair_abstains():
    result = reconcile([tx("b", cents=1001, day=100)], [tx("r", day=0)], PairModel())
    assert result[0]["status"] == "unmatched"


def test_negative_lag_unique_amount_requires_review():
    result = reconcile([tx("b", day=0)], [tx("r", day=3)], PairModel())
    assert result[0]["status"] == "review_required"
    assert "unexpected_date_gap:-3d" in result[0]["warnings"]


def test_training_preserves_matching_for_unseen_merchant():
    pairs = [(tx(f"b{i}", b), tx(f"r{i}", r, day=0))
             for i, (b, r) in enumerate([("cedar", "tulip"), ("juniper", "orchid"),
                                         ("alpine", "violet"), ("cedar", "tulip")])]
    bank = [tx("new-bank", "novel merchant", cents=1005)]
    register = [tx("new-register", "novel merchant", day=0)]
    cold = reconcile(bank, register, PairModel(), unique_stage=False)[0]
    trained = reconcile(bank, register, PairModel().fit(pairs), unique_stage=False)[0]
    assert cold["status"] == trained["status"] == "proposed"
    assert trained["text_similarity"] == pytest.approx(1.0)
    assert trained["confidence_score"] == pytest.approx(cold["confidence_score"])


@pytest.mark.parametrize("bank_text,register_text", [
    ("novel merchant", "tulip merchant"),
    ("cedar merchant", "novel merchant"),
])
def test_lexical_fallback_with_one_unseen_source(bank_text, register_text):
    pairs = [(tx(f"b{i}", b), tx(f"r{i}", r, day=0))
             for i, (b, r) in enumerate([("cedar", "tulip"), ("juniper", "orchid"),
                                         ("alpine", "violet")])]
    bank, register = [tx("b-new", bank_text)], [tx("r-new", register_text)]
    model = PairModel().fit(pairs)
    np.testing.assert_allclose(model.similarities(bank, register),
                               model.similarities(bank, register, mode="lexical"))

