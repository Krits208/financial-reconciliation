"""Candidate scoring and global one-to-one assignment, with an abstain option."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import Transaction, format_cents
from .model import PairModel


@dataclass(frozen=True)
class Config:
    amount_tolerance_cents: int = 10
    candidate_window_days: int = 45
    expected_max_lag_days: int = 5
    minimum_score: float = 0.62
    confident_score: float = 0.72
    minimum_margin: float = 0.025

    def __post_init__(self):
        if self.amount_tolerance_cents < 0 or self.candidate_window_days < 0:
            raise ValueError("Candidate tolerances must be nonnegative")
        if not 0 <= self.minimum_score <= self.confident_score <= 1:
            raise ValueError("Require 0 <= minimum_score <= confident_score <= 1")
        if not 0 <= self.minimum_margin <= 1 or self.expected_max_lag_days < 0:
            raise ValueError("Invalid margin or expected lag")


def pair_warnings(bank: Transaction, register: Transaction, config: Config) -> list[str]:
    warnings = []
    gap = (bank.date - register.date).days
    if gap < 0 or gap > config.expected_max_lag_days:
        warnings.append(f"unexpected_date_gap:{gap}d")
    if bank.cents != register.cents:
        warnings.append(f"amount_difference:{abs(bank.cents - register.cents)}c")
    if bank.direction != register.direction:
        warnings.append("direction_conflict")
    if bank.check_reference and register.check_reference and bank.check_reference != register.check_reference:
        warnings.append("check_reference_conflict")
    return warnings


def needs_review(warnings: list[str]) -> bool:
    # Small rounding differences are displayed but do not alone prevent a proposal.
    return any(not warning.startswith("amount_difference:") for warning in warnings)


def reconcile(
    bank: list[Transaction], register: list[Transaction], model: PairModel,
    validated: dict[str, str] | None = None, rejected: set[tuple[str, str]] | None = None,
    config: Config | None = None, mode: str = "learned", unique_stage: bool = True,
) -> list[dict]:
    config = config or Config()
    validated, rejected = validated or {}, rejected or set()
    bm, rm = {t.id: t for t in bank}, {t.id: t for t in register}
    if len(bm) != len(bank) or len(rm) != len(register):
        raise ValueError("Transaction IDs must be unique within each source")
    if len(set(validated.values())) != len(validated):
        raise ValueError("Validated matches must be one-to-one")
    for bid, rid in validated.items():
        if bid not in bm or rid not in rm or (bid, rid) in rejected:
            raise ValueError("Invalid or rejected pair in validated matches")
    results = {}
    used_register = set(validated.values())
    for bid, rid in validated.items():
        results[bid] = _result(bm[bid], rm[rid], "validated", "human", None, None, None,
                               pair_warnings(bm[bid], rm[rid], config), "Human-validated pair")

    # Count uniqueness over each ENTIRE input, before removing reviewed matches.
    bc, rc = Counter(t.cents for t in bank), Counter(t.cents for t in register)
    unique_register = {t.cents: t for t in register if rc[t.cents] == 1}
    if unique_stage:
        for b in bank:
            if b.id in results or bc[b.cents] != 1 or b.cents not in unique_register:
                continue
            r = unique_register[b.cents]
            if r.id in used_register or (b.id, r.id) in rejected or b.direction != r.direction:
                continue
            warnings = pair_warnings(b, r, config)
            gap = (b.date - r.date).days
            status = "review_required" if needs_review(warnings) else "proposed"
            score = 0.90 + 0.08 * model.lag_score(gap, mode == "learned")
            results[b.id] = _result(b, r, status, "unique_amount", score, None, None, warnings,
                                    "Exact amount occurs once in each complete input; direction agrees")
            used_register.add(r.id)

    bs = sorted((b for b in bank if b.id not in results), key=lambda t: t.id)
    rs = sorted((r for r in register if r.id not in used_register), key=lambda t: t.id)
    if bs:
        text_scores = model.similarities(bs, rs, mode)
        scores = np.full((len(bs), len(rs)), -1.0)
        for i, b in enumerate(bs):
            for j, r in enumerate(rs):
                difference = abs(b.cents - r.cents)
                gap = (b.date - r.date).days
                if (b.id, r.id) in rejected or b.direction != r.direction:
                    continue
                if difference > config.amount_tolerance_cents or abs(gap) > config.candidate_window_days:
                    continue
                amount = math.exp(-difference / max(1.0, config.amount_tolerance_cents * 1.5))
                lag = model.lag_score(gap, mode == "learned")
                if mode == "numeric":
                    score = 0.70 * amount + 0.30 * lag
                else:
                    score = 0.65 * amount + 0.20 * lag + 0.15 * float(text_scores[i, j])
                if b.check_reference and r.check_reference:
                    score += 0.03 if b.check_reference == r.check_reference else -0.20
                scores[i, j] = min(1.0, max(0.0, score))

        # One dummy column per bank row permits any number of abstentions.
        # Maximize score above the abstention floor, not raw match cardinality.
        utilities = np.concatenate((scores - config.minimum_score, np.zeros((len(bs), len(bs)))), axis=1)
        row_indices, col_indices = linear_sum_assignment(utilities, maximize=True)
        for i, j in zip(row_indices, col_indices):
            b = bs[i]
            if j >= len(rs) or scores[i, j] <= config.minimum_score:
                candidates = np.flatnonzero(scores[i] >= 0)
                reason = "No candidate within amount/date/direction constraints" if not len(candidates) else "No sufficiently strong available one-to-one match"
                results[b.id] = _result(b, None, "unmatched", "none", None, None, None, [], reason)
                continue
            r = rs[j]
            alternatives = np.delete(scores[i], j)
            competitors = np.delete(scores[:, j], i)
            runner_up = max(config.minimum_score,
                            float(alternatives.max()) if len(alternatives) else -1,
                            float(competitors.max()) if len(competitors) else -1)
            # Conservative local ambiguity check in both directions. A global
            # assignment may legitimately select a local second choice: review it.
            margin = float(scores[i, j] - runner_up)
            warnings = pair_warnings(b, r, config)
            if margin < config.minimum_margin:
                warnings.append("ambiguous_alternatives")
            if scores[i, j] < config.confident_score:
                warnings.append("low_confidence_score")
            status = "proposed" if scores[i, j] >= config.confident_score and not needs_review(warnings) else "review_required"
            alternatives_ranked = sorted(range(len(rs)), key=lambda k: (-scores[i, k], rs[k].id))
            alternative_ids = [rs[k].id for k in alternatives_ranked if k != j and scores[i, k] >= config.minimum_score][:3]
            explanation = f"Amount difference {abs(b.cents-r.cents)}c; date gap {(b.date-r.date).days}d; text similarity {text_scores[i,j]:.3f}"
            results[b.id] = _result(b, r, status, mode, float(scores[i, j]), margin,
                                    float(text_scores[i, j]), warnings, explanation)
            results[b.id]["alternatives"] = "|".join(alternative_ids)
    return [results[b.id] for b in bank]


def _result(b, r, status, method, score, margin, text_score, warnings, explanation):
    return {
        "bank_id": b.id, "register_id": r.id if r else "", "status": status, "method": method,
        "confidence_score": round(score, 6) if score is not None else "",
        "margin": round(margin, 6) if margin is not None else "",
        "text_similarity": round(text_score, 6) if text_score is not None else "",
        "bank_date": b.date.isoformat(), "register_date": r.date.isoformat() if r else "",
        "bank_amount": format_cents(b.cents), "register_amount": format_cents(r.cents) if r else "",
        "bank_description": b.description, "register_description": r.description if r else "",
        "warnings": "|".join(warnings), "alternatives": "", "explanation": explanation,
    }

