"""Strict CSV ingestion. IDs identify rows but never become matching features."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Transaction:
    id: str
    date: date
    description: str
    cents: int
    direction: str
    category: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {**asdict(self), "date": self.date.isoformat()}

    @classmethod
    def from_dict(cls, row: dict) -> Transaction:
        return cls(**{**row, "date": date.fromisoformat(row["date"])})

    @property
    def text(self) -> str:
        return " ".join((self.description, self.category, self.notes)).strip()

    @property
    def check_reference(self) -> str | None:
        found = re.search(r"\b(?:check|ck)\s*#?\s*(\d+)\b", self.description, re.I)
        return found.group(1).lstrip("0") or "0" if found else None


def money_to_cents(value: str) -> int:
    """Reject malformed money instead of silently rounding or using float equality."""
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid amount: {value!r}") from exc
    if not amount.is_finite() or amount < 0 or amount * 100 != (amount * 100).to_integral_value():
        raise ValueError(f"Amount must be finite, nonnegative, and have at most two decimal places: {value!r}")
    return int(amount * 100)


def format_cents(cents: int) -> str:
    """Render money without introducing floating-point rounding, even for large values."""
    return f"{cents // 100}.{cents % 100:02d}"


def read_transactions(path: str | Path) -> list[Transaction]:
    required = {"transaction_id", "date", "description", "amount", "type"}
    direction = {"DEBIT": "out", "DR": "out", "CREDIT": "in", "CR": "in"}
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, raw in enumerate(reader, 2):
            try:
                key = raw["transaction_id"].strip()
                if not key or key in seen:
                    raise ValueError(f"Empty or duplicate transaction ID: {key!r}")
                transaction = Transaction(
                    key, date.fromisoformat(raw["date"].strip()), raw["description"].strip(),
                    money_to_cents(raw["amount"].strip()), direction[raw["type"].strip().upper()],
                    (raw.get("category") or "").strip(), (raw.get("notes") or "").strip(),
                )
            except (ValueError, KeyError, AttributeError, TypeError) as exc:
                raise ValueError(f"{path}, row {line}: {exc}") from exc
            rows.append(transaction)
            seen.add(key)
    if not rows:
        raise ValueError(f"{path}: no transactions")
    return rows


def fingerprint(bank: list[Transaction], register: list[Transaction]) -> str:
    """Bind review state to content, independent of CSV row order."""
    payload = [[t.to_dict() for t in sorted(source, key=lambda t: t.id)] for source in (bank, register)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

