# Data access

**This repository contains no data.** All four datasets are distributed by
[DementiaBank](https://dementia.talkbank.org/) (TalkBank) under a membership agreement that forbids
redistribution. That restriction also covers material derived from them: transcripts, manifests
(which carry labels, age, sex, and MMSE), extracted features, embeddings, and the speaker mapping
produced by `scripts/match_pitt.py`. Do not commit any of these to a public repository; the
provided `.gitignore` excludes the usual locations.

## 1. Obtaining the data

| Dataset | Source | Notes |
|---|---|---|
| ADReSS (2020) | DementiaBank, ADReSS challenge release | Train/test audio, transcripts, and test labels (`2020Labels.txt`) |
| ADReSSo (2021) | DementiaBank, ADReSSo challenge release | Diagnosis task audio, segmentation, MMSE scores, and test ground truth |
| Pitt Cookie Theft | DementiaBank, Pitt corpus (English, Becker et al. 1994) | Only the `cookie` task of `Control` and `Dementia` is used: media (`.mp3`) and transcripts (`.cha`); diagnosis, age, sex and MMSE are read from the transcripts' `@ID` lines |
| TAUKADIAL (2024) | DementiaBank, TAUKADIAL challenge release | Train/test audio, `groundtruth.csv`, `testgroundtruth.csv`, `meta_test.csv` |

Request access through DementiaBank membership; approval can take several weeks.

The pyannote voice-activity model is gated as well: accept the terms of
[`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) on Hugging Face and
log in once with `hf auth login` (or export `HF_TOKEN`). Without it, use the energy VAD configs.

## 2. Expected layout

The scripts expect the data next to the repository, and write all derived material to a sibling
`run_full/` directory (both paths can be overridden, see below):

```
project/
├── sabench/                      # this repository
├── Data/
│   ├── ADReSS2020/
│   │   ├── 2020Labels.txt
│   │   └── ADReSS-IS2020-data/{train,test}/...
│   ├── AddReSSo2021/
│   │   ├── MMSE scores/dx-mmse.csv
│   │   ├── ground truth of test sets/task{1,2,3}.csv
│   │   └── diagnosis/{train,test-dist}/{audio,segmentation}/...
│   ├── Pitt/
│   │   ├── media/{Control,Dementia}/cookie/*.mp3
│   │   ├── transcripts/{Control,Dementia}/cookie/*.cha
│   │   └── metadata/            (optional; not read by the pipeline)
│   └── TAUKADIAL/
│       ├── train/{*.wav,groundtruth.csv}
│       ├── test/*.wav
│       ├── testgroundtruth.csv
│       └── meta_test.csv
└── run_full/                     # created by scripts/prepare_all.sh
```

To use other locations, set `DATA=/path/to/Data RUN=/path/to/run_full` when calling
`scripts/prepare_all.sh`, and change `paths.*` in the config you use (or pass
`--set paths.raw=...` etc. to any `sabench` command).

## 3. Manifests

Every `sabench` command reads one CSV manifest per dataset. `scripts/build_manifests.py` builds them
from the releases above. Columns:

| Column | Required | Meaning |
|---|---|---|
| `id` | yes | Recording id, unique within the dataset; the join key for every feature table |
| `split` | yes | `train` or `test` (the official challenge split, or a fixed stratified split for Pitt-wide) |
| `label` | yes | `0` = control / normal cognition, `1` = dementia / MCI |
| `path` | yes | Audio path relative to `<paths.raw>/<dataset>/` (resolved with a traversal-safe join) |
| `speaker` | no | Speaker key; enables speaker-aware CV, permutation, and bootstrap when a speaker has several recordings |
| `age`, `gender`, `mmse` | no | Demographics and cognitive score (MMSE is never used as a feature) |
| `pitt_session` | no | Source Pitt session (`<speaker>-<visit>`), added by `scripts/match_pitt.py` |
| `language` | no | Spoken language (`en`/`zh`), added by `scripts/detect_language.py` |

`sabench validate --dataset all --config <config>` checks per-split class counts against the
published numbers in the config's `expected_counts`.
