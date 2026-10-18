# Design decisions

Decisions that change results, with the reason for each. Newest first.

### Speaker-aware resampling everywhere
When a speaker has several recordings (TAUKADIAL: three each; four ADReSSo speakers: two), inner CV
uses `StratifiedGroupKFold`, permutation nulls shuffle per-speaker labels, and bootstrap CIs resample
whole speakers. Otherwise selection can reward speaker identity, the null is invalid, and intervals
are too narrow. `sabench.modeling.groups.effective_groups` returns `None` when every speaker has one
recording, so those datasets run the original code path unchanged.

### Seed-stable selection: `search.cv_repeats = 20`, `embeddings.probe.cv_repeats = 10`
With a single CV shuffle, the train-only selection depended on the fold seed (up to nine different
feature pairs across ten seeds; p-values from about 0.03 to 0.13). Averaging the criterion over 20
shufflings makes the selected pair nearly seed-independent (`scripts/repeats_stability.py`); 10 do
the same for probe layer selection. Repeat 0 uses the base seed, so `cv_repeats = 1` reproduces
single-shuffle CV exactly.

### Probe layer chosen by inner CV, never by test F1
The probe reports the layer with the best inner-CV score. The test-picked layer is reported only as
an optimistic oracle, because picking it is the test-informed selection E1 quantifies.

### Fast, bit-identical liblinear
`FastLiblinearLR` calls the same `liblinear.train_wrap` routine with the same arguments as
`LogisticRegression(solver="liblinear")`, skipping per-call validation that dominated runtime.
`tests/test_search.py` asserts identical coefficients and predictions against scikit-learn on 300+
problems. Together with per-fold preprocessing caching this makes the search about 31x faster with
identical results.

### Replication targets are ComParE 2016
The base paper's 0.875 / 0.831 come from ComParE 2016 features (its Table 3 names are ComParE
descriptors), so gate G1 is assessed on ComParE; eGeMAPS is reported alongside.

### ADReSSo training counts
The official ADReSSo release has 79 controls and 87 AD in train. The base paper's Table 1 swaps
these; `configs/default.yaml` follows the release.

### Speaker linkage by audio
Challenge files carry no source-corpus ids, and ADReSS transcripts use a different TalkBank PID
namespace from the Pitt transcripts. `scripts/match_pitt.py` matches recordings to Pitt sessions by
duration and loudness-envelope correlation, then uses ADReSSo's own metadata for the rest. Matches are
checked against diagnosis labels, ADReSSo metadata, and transcript text.

### VAD backend is explicit
`vad.backend` is `energy`, `pyannote` (fail instead of falling back), or `auto`. Configs whose
results were produced with one backend pin it, so installing pyannote cannot silently change them.

### Preprocessing is fit inside every fold
Mean imputation and z-scoring are fit on the training rows of each fold only. Fitting once on the
whole training split before inner CV lets each validation fold leak into its own normalisation.

### ASR safeguards
Whisper long-form decoding with temperature fallback, compression and log-probability thresholds,
and no conditioning on previous text; then runs of a word or word pair repeated more than three times
are collapsed to three. Without these, about 4% of transcripts contained hallucinated repetition
loops.
