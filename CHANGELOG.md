# Changelog

## 0.1.0 (2026-10-16)

First public release.

- Search-aware evaluation protocol: test-informed vs train-only selection, and permutation nulls that
  re-run the whole selection procedure (feature-pair search, probe C and layer selection).
- Seed-stable train-only selection via repeated inner cross-validation (`search.cv_repeats`,
  `embeddings.probe.cv_repeats`).
- Speaker-aware cross-validation, permutation, bootstrap, and resampling for datasets with several
  recordings per speaker.
- Features: openSMILE eGeMAPSv02 and ComParE 2016; frozen WavLM, wav2vec 2.0, and Whisper-encoder
  embeddings; Whisper ASR with loop safeguards; BERT, RoBERTa, and XLM-R text embeddings.
- Voice-activity detection with pyannote segmentation-3.0 or an energy detector; full, speech-only,
  and silence-only variants; channel descriptors.
- Dataset support: ADReSS, ADReSSo, a Pitt-wide Cookie Theft benchmark, and TAUKADIAL (English and
  Mandarin), including speaker linkage of the challenge sets to the Pitt corpus and language ID.
- Analyses: replication gate, uncertainty (bootstrap, Monte Carlo, repeated CV), segment/channel
  baselines, cross-dataset transfer with speaker exclusion, K/seed sensitivity, shortcut baselines,
  ASR quality, and Holm-corrected test summaries.
- One-command pipeline (`scripts/prepare_all.sh`, `scripts/run_final.sh`); 137 tests.
