# Design decisions and operational behavior

## Model and feature definitions

For each approved pair, normalize description/category/notes to alphabetic tokens,
add word bigrams, and prefix tokens with their source (`b__` or `r__`). Build a
TF-IDF matrix whose rows are paired documents and columns are source-specific terms.
Use sublinear term frequency and standard smoothed IDF. There is no manually written
merchant-to-category dictionary.

Factor the paired matrix with truncated SVD. The number of components is
`min(16, reviewed_pairs - 1, vocabulary_size - 1)`; random state is 42. Project new
single-source TF-IDF rows using the learned components, normalize, and compute cosine
similarity clipped to [0, 1]. A cold model falls back to character word-boundary TF-IDF
with 2–4 character n-grams. This fallback is fitted on unlabelled candidate text.
It also applies when either record has no usable SVD projection, so unseen merchants
retain their lexical evidence rather than being treated as dissimilar.

This is a lightweight cross-language LSA adaptation. It is not an exact reproduction
of the paper: it skips mutual-information term alignment. Repeated co-occurrence of
bank and register terms in reviewed pairs supplies the correspondence signal.

## Numerical evidence and combined score

Let `d` be the absolute amount difference in cents and `g` the signed lag in days
(`bank date - register date`). With amount tolerance `T` (default 10 cents):

```text
amount_score = exp(-d / max(1, 1.5 * T))
lag_score = exp(-0.5 * ((g - center) / scale)^2)
text_score = 0.85 * latent_cosine + 0.15 * lexical_cosine
hybrid_score = 0.65 * amount_score + 0.20 * lag_score + 0.15 * text_score
numeric_baseline_score = 0.70 * amount_score + 0.30 * lag_score
```

The learned lag center is the median reviewed gap. Its scale is
`max(2, 1.4826 * median_absolute_deviation)`. Without reviewed pairs, center=2 and
scale=3. Numeric and lexical baselines always retain those fixed defaults, so they
do not gain supervised information through the date feature.

If both descriptions contain check references, agreement adds 0.03; disagreement
subtracts 0.20 and adds a material warning. Scores are clipped to [0, 1]. The
unique-amount stage uses `0.90 + 0.08 * lag_score` and independent anomaly flags.

These weights encode conservative engineering choices, not learned probabilities.
The model learns vocabulary relationships and typical posting delays; it does not
learn score weights or fine-tune a neural network.

## Assignment and confidence policy

After reserving validated and unique-amount pairs, candidate pairs must agree on
direction and satisfy `d <= 10` cents and `abs(g) <= 45` days by default. Unique exact
amounts can still propose pairs outside that date window, but those are flagged.

The assignment maximizes `score - 0.62`, with one zero-utility dummy column per bank
row. A real assignment must beat that floor. Dummy assignments represent abstention.
This permits unequal row counts and missing counterparts without inventing matches.

A candidate's margin compares its score against the best competing alternative in
both its bank row and its register column, including the abstention floor. A global
assignment can select a local second choice; its margin then flags that ambiguity.
This local margin is conservative and is not an exact global-assignment uncertainty
estimate.

| Status | Meaning |
|---|---|
| `proposed` | Machine score >=0.72, margin >=0.025 where applicable, and no material warnings |
| `review_required` | An assigned pair has weak evidence, ambiguity, or a date/reference anomaly |
| `unmatched` | No eligible, sufficiently strong, available assignment |
| `validated` | A human explicitly accepted or corrected this pair |

The unique-amount stage does not require a margin because amount uniqueness already
isolates a candidate. A 0–5 cent difference is displayed, but a small difference alone
does not require review. Unexpected date gaps (negative or >5 days), check-reference
conflicts and ambiguous alternatives do. A high confidence score and a warning can
coexist: confidence is not a probability of correctness or permission to ignore flags.

## Feedback integrity

- A proposal is never a training example until a reviewer accepts/corrects it.
- Every review file is bound to a hash of both normalized datasets and a state revision.
- Accepted pairs must be one-to-one. A batch with conflicts is rejected without changes.
- Corrections can release and swap targets atomically within one review batch.
- Rejected pairs remain prohibited, unless a reviewer later explicitly accepts them.
- The state stores accepted IDs, rejected pairs and time-stamped events. The source
  records are checked by fingerprint on every load; raw datasets remain unchanged.
- Model parameters are deterministically refitted from current reviewed pairs on each
  match run. Refitting at this scale is cheap and avoids stale serialized models.

The CLI is a single-user prototype. Atomic file replacement prevents partial writes,
but it is not a concurrency-control mechanism or a production audit database.

## Evaluation design

1. Report easy unique-amount coverage separately from the 22 remaining records.
2. Compare numeric, lexical and learned methods without claiming that ML must win.
3. Simulate review of unique-amount pairs, then score only previously unreviewed rows.
4. Use three fixed random splits. Within each split, hold out the same 92 pairs while
   growing nested reviewed-training subsets from 0 to 216 pairs.
5. Disable unique-amount matching in the learning curves to isolate the second stage.
6. Evaluate a separate, explicitly modified collision fixture containing only held-out
   records. Its metadata shortcuts are removed; its original descriptions are retained.
7. Report precision, recall, F1, coverage, seed variation and all-assignment precision.
   Keep error/review rows available for inspection.

No held-out pair is used to fit paired SVD or learned date statistics. The reference
file is accessible only to evaluation/demo code, never to the matcher. ID-renaming
tests check that IDs cannot change text features. Dataset labels remain provisional;
structural consistency checks do not establish an independently verified gold standard.

