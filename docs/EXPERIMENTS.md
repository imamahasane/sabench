# Experiments

Every experiment compares two **selection regimes** for whatever choice a model involves (which two
features, which embedding layer, which regularisation strength):

* **`test_informed`** -- the choice is made by test-set performance. Reproduces the baseline's
  analysis and shows how optimistic such scores are.
* **`train_only`** -- the choice is made by stratified cross-validation inside the training split
  (averaged over `search.cv_repeats` shufflings); the test split is touched once, to report the
  final score.

Significance always comes from a **search-aware permutation null**: the training labels are shuffled
and the *entire* selection procedure is re-run, `permutation_null.n_permutations` times. The p-value
is `(1 + #{null >= observed}) / (B + 1)`. When a speaker has several recordings, labels are permuted
per speaker, inner CV is speaker-disjoint, and bootstrap CIs resample speakers
(`sabench.modeling.groups`).

All result tables are written to `<paths.results>/tables/` and carry `regime`, `seed`, and
`config_hash` columns.

| Stage | Command | What it computes | Output tables |
|---|---|---|---|
| E0 replication | `sabench replicate <ds> --featureset <fs>` | Two-feature test-informed search, checked against the published targets in `replication.targets` (gate G1) | `e0_<ds>_<fs>` |
| E1 search-aware null | `sabench null <ds> --featureset <fs> --regime <r>` | Observed pair search plus its permutation null and p-value | `e1_observed_*`, `e1_null_*` |
| E2 uncertainty | `sabench uncertainty <ds> --featureset <fs>` | Bootstrap CI, Monte Carlo 70/30 resampling, and repeated CV for the selected pair | `e2_uncertainty_*` |
| E3 probes | `sabench probes <ds> --model <m> [--variant text_asr] --probe-null --pair-search-null` | Layer-wise probes on frozen speech or text embeddings; layer chosen by inner CV (the test-picked layer is reported only as an optimistic oracle); probe null and a pair-search null inside the chosen layer | `e3_probe_*`, `e3_probe_null_*`, `e3_probe_observed_*`, `e3_pairsearch_*` |
| E4 segments and channel | `sabench segment-channel <ds> --featureset <fs>` | Train-only best pair on full / speech-only / silence-only audio, plus a channel-descriptor-only baseline | `e4_segment_channel_*` |
| E6 cross-dataset | `scripts/cross_dataset_dedup.py` | Train on one challenge, test on the other, with no exclusion, identical recordings removed, and shared speakers removed | `e6_cross_dedup` |

Supporting analyses (all in `scripts/`):

| Script | Purpose | Output |
|---|---|---|
| `sensitivity.py` | Observed search across `top_k` in {10, 25, 50, 100} and five seeds | `sensitivity_k_seed` |
| `repeats_stability.py` | Seed stability of train-only selection for 1, 5, 10, 20 CV repeats | `repeats_stability` |
| `shortcut_baselines.py` | Age+gender and language-only classifiers (test split, train CV, Monte Carlo) | `shortcut_baselines` |
| `table4_replication.py` | Base paper Table 4: a fixed pair on pyannote silence-only vs speech-only audio | `table4_replication` |
| `asr_quality.py` | Participant-word recall of ASR vs manual transcripts; repetition check | `asr_quality` |
| `summarize_tests.py` | Collects every p-value into families and applies Holm correction | `tests_summary_holm` |

`sabench figures` rebuilds Figure 1 (null distributions, both regimes) and Figure 2 (layer-wise probe
scores) from the tables. `scripts/run_final.sh` runs everything above for all datasets.

## Datasets

| Key | Task | Unit | Split |
|---|---|---|---|
| `adress` | Alzheimer's dementia vs control (English) | 156 speakers, one recording each | official |
| `adresso` | Alzheimer's dementia vs control (English, speech only) | 237 recordings | official |
| `pitt` | Probable AD vs control, Pitt Cookie Theft, first visit per speaker | 246 speakers | fixed stratified 70/30 (seed 1234) |
| `taukadial` | MCI vs normal cognition (English and Mandarin) | 169 speakers x 3 recordings | official, with released test labels |
