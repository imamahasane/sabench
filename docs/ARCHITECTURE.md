# Architecture

## Layering

```
cli            Typer commands. Parses args, builds an ExperimentRunner, calls pipeline, prints/exits.
  v
pipeline       The only layer that touches disk. Loads manifests/features, calls modeling, writes
               result tables with provenance columns. stages.py operates on in-memory arrays (Split
               namedtuples) so it's unit-testable without touching disk itself.
  v
modeling       Pure NumPy / pandas / scikit-learn. No file I/O, no logging side effects beyond the
               module logger. Everything here takes arrays in and returns arrays/dataclasses out.
  ^
audio/features  Turn raw recordings into the arrays modeling consumes. The only layer allowed to
               import torch/transformers/opensmile/pyannote.audio, and only behind a lazy import
               that raises ModelUnavailableError if the optional dependency isn't installed.
  ^
data           Manifest loading/validation; the id-based join every other layer relies on.
  ^
config / utils / exceptions / logging_config / constants   Cross-cutting, imported by everything.
```

Dependency direction only ever points downward in this list. `modeling` never imports `pipeline` or
`cli`; `audio`/`features` never import `modeling`. This is what makes `modeling` and `audio`/`features`
independently unit-testable on synthetic data.

## Why a separate `pipeline/stages.py` from `pipeline/io.py` and `cli/app.py`

- `pipeline/io.py`: disk <-> array conversion (CSV/NPZ read-write, manifest alignment, train/test
  splitting). Knows file formats, nothing about experiments.
- `pipeline/stages.py`: one function per E0-E6 stage, operating on `Split` (a plain
  `(X_train, y_train, X_test, y_test)` namedtuple), optional speaker groups, and a `Config`. No disk access. This is what gets
  unit-tested with synthetic arrays, and what a notebook would import directly instead of going
  through the CLI.
- `cli/app.py`: argument parsing and wiring -- load manifest, load features, call `pipeline.io` to
  align/split, call `pipeline.stages`, call `runner.save_table`. Deliberately thin; any logic
  complex enough to need its own unit test belongs in `stages.py` or `modeling/`, not here.

## Modules at a glance

| Module | Responsibility |
|---|---|
| `audio/prepare.py`, `audio/vad.py`, `audio/channel.py` | Load, mono-mix, resample; pyannote or energy VAD; full/speech/silence variants; channel descriptors |
| `features/lld.py` | openSMILE eGeMAPSv02 / ComParE 2016 functionals |
| `features/embeddings.py` | Frozen, layer-wise, mean-pooled WavLM / wav2vec 2.0 / Whisper-encoder embeddings |
| `features/text.py` | CHAT transcript parsing, Whisper ASR with loop safeguards, layer-wise BERT / RoBERTa / XLM-R embeddings |
| `modeling/search.py` | Single-feature ranking and the two-feature pair search (both regimes, repeated inner CV) |
| `modeling/nulls.py` | Search-aware permutation nulls for the pair search and for probes |
| `modeling/probes.py` | Layer-wise regularised-LR probes; CV-based C and layer selection |
| `modeling/groups.py` | Speaker-aware CV splits, label permutation, bootstrap, and holdout |
| `modeling/bootstrap.py`, `modeling/resample.py` | Bootstrap CIs, Monte Carlo resampling, repeated CV |
| `modeling/metrics.py` | Macro-F1 and the bit-identical fast liblinear classifier |
| `pipeline/` | Disk I/O, stage orchestration (E0-E6), provenance on result tables |
| `cli/app.py` | The `sabench` command group |
| `scripts/` | Dataset-specific builders (manifests, Pitt linkage, language ID, transcripts), cross-cutting analyses, and the two driver scripts |

## Speaker-aware statistics

Every split, permutation, and resample goes through `modeling/groups.py`. Callers pass one speaker
key per row (from the manifest's `speaker` column); when every speaker has a single recording the
helpers fall back to the ungrouped code paths, so results on such datasets are unchanged.

## Error handling

Every error a user can hit (bad config, missing manifest, misaligned ids, unavailable optional
dependency, malformed search inputs) is one of the `SabenchError` subclasses in `exceptions.py`. The
CLI catches `SabenchError` at each command's top level and exits with a message and a non-zero code;
anything else propagating out of a command is a bug (an unanticipated failure mode), not a user error,
and is allowed to produce a full traceback rather than being silently swallowed.

## Why pydantic for config instead of a plain dict/YAML

A typo in `configs/default.yaml` (`n_permutation` instead of `n_permutations`, a negative `top_k`, an
unknown regime name) fails at `load_config()` with a precise message, before any audio is touched or
any hour-long permutation job is launched. `Config` is frozen (immutable) so a `config_hash()` taken
at the start of a run is guaranteed to describe exactly what ran.

## Why the pair search takes a classifier factory, not a classifier instance

`sabench.modeling.metrics.prepped_macro_f1` takes a zero-argument factory (`Callable[[], Classifier]`)
rather than a pre-built classifier so that preprocessing + fit always happen together, inside the
function, on whatever subset of rows the caller currently considers "training" -- a fresh,
unfitted classifier every call. Passing a shared, already-fitted instance across bootstrap/CV/Monte
Carlo iterations is exactly the kind of state leakage between iterations this project exists to catch
in someone else's benchmark; the factory pattern makes it structurally impossible here.
