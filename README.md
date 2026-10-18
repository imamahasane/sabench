# sabench

**Search-aware evaluation of speech-based dementia detection benchmarks.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Tests: 137](https://img.shields.io/badge/tests-137-brightgreen.svg)

`sabench` measures how much of the accuracy reported on public dementia-speech benchmarks survives
honest evaluation. It compares models selected on the test set with models selected on training data
only, and judges every score against a **search-aware permutation null**: the labels are shuffled and
the *entire* model search is re-run, so the null shows how high a score the search itself can produce
from noise.

It covers ADReSS, ADReSSo, a Pitt-wide benchmark built from the source corpus, and TAUKADIAL (English
and Mandarin), and applies the same protocol to hand-crafted acoustic features, frozen speech
encoders, and text models.

> **No data is included.** All datasets are distributed by DementiaBank under a licence that forbids
> redistribution. See [Data access](#data-access).

---

## Contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [Installation](#installation)
- [Data access](#data-access)
- [Quick start](#quick-start)
- [Reproducing the full study](#reproducing-the-full-study)
- [Running individual stages](#running-individual-stages)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Method in brief](#method-in-brief)
- [Project structure](#project-structure)
- [Development](#development)
- [Citation](#citation)
- [Licence and data use](#licence-and-data-use)
- [Acknowledgements](#acknowledgements)

## Why this exists

Zafar, Shahin and Ahmed (Interspeech 2026) showed that two low-level acoustic features, chosen on the
test set, reach near-state-of-the-art scores on ADReSS and ADReSSo. That raises a question the field
needs a tool to answer: *is a reported score better than what the model search finds in noise?*

`sabench` answers it, and checks the conditions benchmark results quietly depend on:

- **Selection.** Test-informed vs train-only selection, with p-values that account for the search.
- **Stability.** Whether the selected model depends on the random seed, the search size, or the
  voice-activity detector.
- **Overlap.** Whether two benchmarks share speakers (ADReSS and ADReSSo are both cut from the Pitt
  corpus); every challenge recording is linked to its source speaker.
- **Shortcuts.** Whether age, gender, language, or recording conditions alone predict the label.

## Features

- **Search-aware nulls** for two-feature acoustic search, embedding-layer probes, and pair search
  inside an embedding layer (1,000 permutations by default).
- **Seed-stable train-only selection** through repeated inner cross-validation.
- **Speaker-aware statistics**: grouped CV, per-speaker label permutation, and cluster bootstrap for
  datasets with several recordings per speaker; a no-op when speakers are unique.
- **Features**: openSMILE eGeMAPSv02 and ComParE 2016; frozen WavLM, wav2vec 2.0, and Whisper-encoder
  embeddings (all layers); Whisper ASR with repetition-loop safeguards; BERT, RoBERTa, and XLM-R text
  embeddings.
- **Voice-activity detection** with pyannote segmentation-3.0 or an energy detector, producing full,
  speech-only, and silence-only versions of every recording, plus channel descriptors.
- **Uncertainty**: bootstrap confidence intervals, Monte Carlo resampling, repeated CV, and Holm
  correction across each family of tests.
- **Reproducibility**: one validated YAML config, every result table stamped with `regime`, `seed`,
  and `config_hash`, and a one-command rebuild from raw data.
- **Speed**: a liblinear wrapper that is bit-identical to scikit-learn's (verified in the test suite)
  makes the search about 31x faster, so 1,000-permutation nulls are practical on a laptop.

## Installation

Requires **Python 3.10+** (developed and tested with 3.11) and `libsndfile`
(`brew install libsndfile` on macOS; `apt-get install libsndfile1` on Debian/Ubuntu).

```bash
git clone <repository-url> sabench
cd sabench
python -m venv .venv && source .venv/bin/activate

pip install -e .                               # core: numpy, pandas, scikit-learn, typer, pydantic, ...
pip install -r requirements-extra.txt          # audio + models: opensmile, librosa, pyannote.audio, torch, transformers
pip install -r requirements-dev.txt            # development: pytest, ruff, mypy
```

The test suite and the synthetic demo need only the core install. Real runs need the extras.

For the pyannote voice-activity detector, accept the terms of
[`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) on Hugging Face, then
run `hf auth login` once (`huggingface-cli login` on older versions). Configs that use the energy
detector need no account.

## Data access

Request access to ADReSS, ADReSSo, the Pitt corpus, and TAUKADIAL through
[DementiaBank](https://dementia.talkbank.org/) membership, then place them next to the repository:

```
project/
├── sabench/      # this repository
├── Data/         # ADReSS2020/, AddReSSo2021/, Pitt/, TAUKADIAL/
└── run_full/     # created by the pipeline (derived data; never commit)
```

[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) gives the exact folder layout, the manifest format, and
how to use other locations.

## Quick start

No data needed:

```bash
make test        # or: pytest            (137 tests, synthetic data only)
sabench demo     # synthetic end-to-end: replication -> search-aware null -> uncertainty
sabench --help   # every command; each accepts --config and repeatable --set key=value
```

## Reproducing the full study

With the data in place:

```bash
bash scripts/prepare_all.sh   # manifests, speaker linkage, language ID, audio variants,
                              # LLD features, speech embeddings, transcripts, text embeddings
bash scripts/run_final.sh     # every experiment on all four datasets, summaries, figures
```

`prepare_all.sh` takes several hours on a recent laptop (transcription and embeddings use a GPU or
Apple-silicon MPS if available). `run_final.sh` runs many 1,000-permutation nulls and takes about a
day on a 10-core machine. Both can be resumed stage by stage with the individual commands below.

## Running individual stages

```bash
C=configs/full.yaml

sabench validate --dataset all --config $C                     # check manifests against published counts
sabench prepare-audio adress --config $C                       # resample, VAD, full/speech/silence variants
sabench extract-lld adress --featureset ComParE_2016 --config $C
sabench extract-embeddings adress --model wavlm --config $C

sabench replicate adress --featureset ComParE_2016 --config $C                  # E0
sabench null adress --featureset eGeMAPSv02 --regime train_only --config $C     # E1
sabench uncertainty adress --featureset eGeMAPSv02 --config $C                  # E2
sabench probes adress --model wavlm --probe-null --pair-search-null --config $C # E3 (speech)
sabench segment-channel adress --featureset eGeMAPSv02 --config $C              # E4

sabench transcribe adress --config $C                                           # Whisper ASR
sabench extract-text-embeddings adress --source asr --model roberta --config $C
sabench probes adress --model roberta --variant text_asr --probe-null --config $C  # E3 (text)

sabench figures --config $C
```

Dataset keys: `adress`, `adresso`, `pitt`, `taukadial`. Featuresets: `eGeMAPSv02`, `ComParE_2016`.
Speech models: `wavlm`, `wav2vec2`, `whisper`. Text models: `bert`, `roberta`, `xlmr`.
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) describes every stage and output.

## Configuration

All parameters live in one YAML file, validated at startup (a typo fails immediately, not hours into
a run). Override any value from the command line:

```bash
sabench null adress --config configs/full.yaml --set permutation_null.n_permutations=200 --set seed=7
```

| Config | Use |
|---|---|
| `configs/default.yaml` | Template with every parameter documented |
| `configs/full.yaml` | Full study, energy VAD (−35 dB) |
| `configs/full_pyannote.yaml` | Same, with the pyannote VAD required (no silent fallback) |
| `configs/full_vad30.yaml`, `configs/full_vad40.yaml` | Energy-VAD threshold sensitivity |
| `configs/sample.yaml` | Small-sample smoke test |

Key parameters: `search.top_k` (features considered in the pair search), `search.cv_repeats`
(inner-CV shufflings for train-only selection), `permutation_null.n_permutations`,
`embeddings.probe.cv_repeats`, `vad.backend`.

## Outputs

Every command writes CSV tables to `<paths.results>/tables/`, each with `regime`, `seed`, and
`config_hash` columns, so any number traces back to the exact settings that produced it. The main
ones:

| Table | Content |
|---|---|
| `tests_summary_holm.csv` | Every significance test, grouped into families, with Holm-adjusted p-values |
| `e0_*`, `e1_*`, `e2_*` | Replication, search-aware nulls, uncertainty |
| `e3_probe_*`, `e3_pairsearch_*` | Speech and text probes, with nulls |
| `e4_segment_channel_*` | Full vs speech vs silence vs channel-only |
| `e6_cross_dedup.csv` | Cross-dataset transfer with recording- and speaker-level exclusion |
| `shortcut_baselines.csv` | Age+gender and language-only baselines |
| `sensitivity_k_seed.csv`, `repeats_stability.csv` | Robustness to search size, seed, and CV repeats |
| `asr_quality.csv`, `table4_replication.csv` | ASR quality; base-paper Table 4 replication |

Figures (`sabench figures`) are written as vector PDFs to `<paths.results>/figures/`.

## Method in brief

1. **Two selection regimes.** *Test-informed* picks the model by test score (how much of the
   literature effectively works); *train-only* picks it by cross-validation inside the training split
   and touches the test set once.
2. **Search-aware null.** Shuffle the training labels and re-run the whole selection, 1,000 times;
   the p-value is the share of shuffled runs that score at least as well as the real one.
3. **Seed-stable selection.** The CV criterion is averaged over 20 shufflings (10 for probes), because
   a single shuffle makes the selected model depend on the seed.
4. **Speaker-aware statistics.** With repeated speakers, CV folds, permutations, and bootstrap samples
   operate on speakers, not recordings.
5. **Controls.** Speech-only vs silence-only audio, channel-only features, and demographic and
   language baselines show what a score can be explained by without the speech itself.

Hard rules the code enforces: preprocessing is fit on training rows only (inside every CV fold);
every join is by recording id, never row position; manifest paths are resolved with a traversal-safe
join; raw data is never written to; every random draw derives from `config.seed`.

Design decisions that change results, and why, are recorded in
[`docs/DECISIONS.md`](docs/DECISIONS.md); the code layout is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Project structure

```
sabench/
├── configs/            YAML configs (one source of truth for every parameter)
├── docs/               DATA_ACCESS, EXPERIMENTS, DECISIONS, ARCHITECTURE
├── scripts/            dataset builders, analyses, prepare_all.sh, run_final.sh
├── src/sabench/
│   ├── audio/          loading, resampling, VAD, variants, channel descriptors
│   ├── features/       openSMILE LLDs, speech embeddings, ASR and text embeddings
│   ├── modeling/       search, nulls, probes, speaker-aware resampling, bootstrap
│   ├── pipeline/       disk I/O, stages E0-E6, provenance
│   ├── data/           manifest loading and validation
│   ├── viz/            figures
│   └── cli/            the `sabench` command
└── tests/              137 tests on synthetic data
```

## Development

```bash
pytest            # tests (synthetic data; no audio or model downloads)
ruff check src tests scripts
mypy src
```

Continuous integration runs lint and tests on Python 3.10–3.12 (`.github/workflows/ci.yml`).
Contributions are welcome; please keep the hard rules above and add a test for any change to
`modeling/`.

## Citation

If you use `sabench`, please cite it (see [`CITATION.cff`](CITATION.cff)). The accompanying paper
reference will be added on publication.

## Licence and data use

The code is released under the [MIT licence](LICENSE). The licence covers the code only: it grants no
rights to the DementiaBank datasets or to anything derived from them (transcripts, manifests,
features, embeddings, speaker maps), which remain under DementiaBank's terms and must not be
committed to public repositories. The pyannote model is subject to its own terms on Hugging Face.

## Acknowledgements

The datasets are provided by DementiaBank (TalkBank) and the organisers of the ADReSS, ADReSSo, and
TAUKADIAL challenges. The baseline analysis this work extends is Zafar, Shahin and Ahmed,
*Rethinking Acoustic Variability of ADReSS and ADReSSo Datasets for Dementia Detection*
(Interspeech 2026).
