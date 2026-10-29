# Evaluation report

Reproduce these results with `python -m recon evaluate --truth data/reference_inferred.csv`.

## Reference-label limitation

If using reference_inferred.csv, labels are inferred from synthetic ID suffixes, not independently verified ground truth.
All supplied-data metrics and learning curves using that file are conditional on this mapping being correct. The stress fixture inherits those pairings. The matching code never reads the reference file or parses ID suffixes.

## Supplied data

308 rows per source; 286 unique exact-amount candidates; 22 remaining rows.

Metrics count only confident machine proposals (`status=proposed`), before human corrections. Review-required/unmatched rows count against recall. Previously reviewed training pairs are excluded from remaining-row and holdout metrics.

| Experiment | Correct / proposed | Eligible pairs | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| unique_amount_only | 285 / 285 | 308 | 100.0% | 92.5% | 96.1% |
| numeric_baseline_all | 307 / 307 | 308 | 100.0% | 99.7% | 99.8% |
| cold_hybrid_all | 301 / 301 | 308 | 100.0% | 97.7% | 98.9% |
| numeric_baseline_remaining | 22 / 22 | 22 | 100.0% | 100.0% | 100.0% |
| cold_hybrid_remaining | 16 / 16 | 22 | 100.0% | 72.7% | 84.2% |
| after_seed_review_remaining | 20 / 20 | 22 | 100.0% | 90.9% | 95.2% |

## Learning curves

Three fixed random seeds (13, 42, 97), each with a 70% training pool and a fixed 30% holdout. Nested training subsets simulate increasing reviewed history. Neither held-out pair labels nor their rows train the paired SVD. The unique-amount stage is disabled throughout these curves to isolate second-stage behavior. The lexical fallback can use unlabelled candidate text for TF-IDF; this is a transductive retrieval setup. Training examples represent correct simulated human reviews, including corrections, rather than self-approved predictions.

`collision_stress` modifies only held-out rows: groups of up to four same-direction transactions receive the same amount and dates. Descriptions remain unchanged, and IDs and ordering are independently randomized. This deliberately removes numeric shortcuts; it is a controlled stress test, not an estimate of production performance. Similar-category duplicates are retained and can remain ambiguous.

| Dataset | Training pairs | Method | Precision mean | Recall mean | F1 mean ± sample SD |
|---|---:|---|---:|---:|---:|
| collision_stress | 0 | learned | 96.9% | 25.7% | 40.5% ± 7.0% |
| collision_stress | 0 | lexical | 96.9% | 25.7% | 40.5% ± 7.0% |
| collision_stress | 0 | numeric | 33.3% | 0.4% | 0.7% ± 1.2% |
| collision_stress | 8 | learned | 97.1% | 32.6% | 47.9% ± 15.6% |
| collision_stress | 32 | learned | 94.2% | 44.9% | 60.6% ± 9.9% |
| collision_stress | 64 | learned | 96.4% | 50.7% | 65.8% ± 11.9% |
| collision_stress | 128 | learned | 93.5% | 53.6% | 68.1% ± 6.2% |
| collision_stress | 216 | learned | 99.5% | 56.9% | 72.1% ± 6.9% |
| original_holdout | 0 | learned | 100.0% | 97.1% | 98.5% ± 0.6% |
| original_holdout | 0 | lexical | 100.0% | 97.1% | 98.5% ± 0.6% |
| original_holdout | 0 | numeric | 100.0% | 99.6% | 99.8% ± 0.3% |
| original_holdout | 8 | learned | 100.0% | 95.3% | 97.6% ± 0.9% |
| original_holdout | 32 | learned | 100.0% | 97.5% | 98.7% ± 0.9% |
| original_holdout | 64 | learned | 100.0% | 98.2% | 99.1% ± 1.2% |
| original_holdout | 128 | learned | 100.0% | 99.3% | 99.6% ± 0.3% |
| original_holdout | 216 | learned | 100.0% | 99.3% | 99.6% ± 0.3% |

## Error analysis and interpretation

At maximum training size, 121 held-out outputs across both datasets and all three seeds are incorrect or require review. Warning counts (can overlap): {'ambiguous_alternatives': 118, 'amount_difference': 1, 'low_confidence_score': 1}. See `errors_and_review_cases.csv` for every such output.

Category results below aggregate the three collision-stress runs at 216 training pairs. Categories describe the expected register transaction; repeated holdout appearances across seeds are not independent new observations.

| Expected category | Correct / confident proposals | Evaluated pairs across runs | Precision | Recall |
|---|---:|---:|---:|---:|
| Payroll | 0 / 0 | 9 | 0.0% | 0.0% |
| Online Purchase | 20 / 20 | 53 | 100.0% | 37.7% |
| Grocery | 22 / 23 | 54 | 95.7% | 40.7% |
| Restaurant | 19 / 19 | 43 | 100.0% | 44.2% |
| Bank Fee | 6 / 6 | 9 | 100.0% | 66.7% |
| Check Payment | 15 / 15 | 20 | 100.0% | 75.0% |
| Transfer | 16 / 16 | 21 | 100.0% | 76.2% |
| Insurance | 8 / 8 | 10 | 100.0% | 80.0% |
| Gas Station | 17 / 17 | 21 | 100.0% | 81.0% |
| Subscription | 11 / 11 | 13 | 100.0% | 84.6% |
| Atm Withdrawal | 9 / 9 | 9 | 100.0% | 100.0% |
| Miscellaneous | 4 / 4 | 4 | 100.0% | 100.0% |
| Utility | 10 / 10 | 10 | 100.0% | 100.0% |

Concrete supplied-data cases: `B0296` / `R0296` has `CHECK #642` versus `Check #6442`, so even its unique amount triggers review. `ONLINE PMT WATER` versus `Electric bill`, and `GYM MEMBERSHIP` versus `Music subscription`, illustrate conflicting semantics under the provisional reference. In the stress fixture, one confident grocery assignment is wrong at maximum training size. The error CSV includes both the proposed and expected descriptions so that such ambiguity can be inspected.

The original fixture has strong numerical shortcuts, so a numeric baseline may equal or beat the hybrid. The stress results test whether source-specific text relationships help when these shortcuts disappear. More training need not improve every metric monotonically: SVD refitting changes the latent space, and confidence/ambiguity thresholds affect coverage.

Precision is defined as zero when no confident matches are proposed; this convention must not be confused with observed incorrect matches. `all_proposals_precision` separately evaluates every assigned pair, including review-required cases, and does not represent automated acceptance accuracy.

## Limits

Only 308 synthetic pairs; provisional reference labels; no independent production benchmark; three seeds are descriptive rather than a statistical significance claim. Random holdout measures within-distribution learning, not future-time generalization. Weights and confidence thresholds are documented engineering defaults, not calibrated probabilities or parameters optimized on this holdout. Candidate tolerances can exclude true anomalies. No split/merged transactions, FX conversion, account grouping or multi-currency handling is implemented.
