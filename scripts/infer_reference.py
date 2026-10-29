"""Create a PROVISIONAL evaluation mapping from this synthetic fixture's IDs.

This is an explicit dataset assumption, not an independently verified gold label.
The matching application does not import this script or this mapping.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recon.data import read_transactions, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="data/bank_statements.csv")
    parser.add_argument("--register", default="data/check_register.csv")
    parser.add_argument("--output", default="data/reference_inferred.csv")
    args = parser.parse_args()
    bank, register = read_transactions(args.bank), read_transactions(args.register)
    if any(not t.id.startswith("B") or not t.id[1:].isdigit() for t in bank):
        raise ValueError("Unexpected bank ID convention")
    if any(not t.id.startswith("R") or not t.id[1:].isdigit() for t in register):
        raise ValueError("Unexpected register ID convention")
    rm = {t.id[1:]: t for t in register}
    if set(rm) != {t.id[1:] for t in bank}:
        raise ValueError("ID suffix sets do not match")
    rows = []
    for b in bank:
        r = rm[b.id[1:]]
        if b.direction != r.direction or abs(b.cents-r.cents) > 5 or not 0 <= (b.date-r.date).days <= 5:
            raise ValueError(f"Inferred pair fails consistency checks: {b.id}, {r.id}")
        rows.append({"bank_id": b.id, "register_id": r.id})
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} PROVISIONAL pairs to {args.output}; verify the ID convention with the dataset author.")


if __name__ == "__main__":
    main()

