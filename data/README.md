# Data provenance and reference assumptions

`bank_statements.csv` and `check_register.csv` are unmodified copies of the two
synthetic CSVs supplied with the assignment. Each contains 308 transactions. The
assignment says both sources describe the same 308 underlying events. The original
PDFs are not redistributed in this repository.

`reference_inferred.csv` is a derived, provisional evaluation mapping. Each `Bxxxx`
bank ID is paired with the `Rxxxx` register ID sharing its numeric suffix. The suffix
sets agree, and all 308 inferred pairs have matching debit/credit direction, a 0–5
day bank posting lag and an amount difference of at most five cents. Those checks
support the assumption but do not independently verify it. In particular, some
descriptions/check references conflict even under this mapping.

Regenerate explicitly:

```bash
python scripts/infer_reference.py
```

The script is restricted to this fixture's observed ID convention. The production
matching path neither imports the script nor reads the reference file. Replace the
reference with a verified `bank_id,register_id` mapping if the dataset author supplies
one. Do not match on IDs or use their suffixes as model features.

The collision stress fixtures are generated in memory from held-out rows by
`recon.evaluation.collision_fixture`. Descriptions/category/notes are preserved;
amounts, dates, IDs and ordering are deliberately changed. They are evaluated and
reported separately from the supplied data. They inherit the provisional mapping's
label assumption and are not independent gold-standard data.

