#!/usr/bin/env bash
# Regenerate every result table and figure from the inputs built by scripts/prepare_all.sh.
# Usage (from the repository root, venv active):  bash scripts/run_final.sh
set -euo pipefail

CFG=configs/full.yaml
TABLES=../run_full/results/tables
FEATURESETS=(eGeMAPSv02 ComParE_2016)
DATASETS=(adress adresso pitt taukadial)
CHALLENGES=(adress adresso)
EXTRA_SEEDS=(1 2)
TEXT_PROBES=(adress:text_manual:bert adress:text_manual:roberta adress:text_asr:bert adress:text_asr:roberta
             adresso:text_asr:bert adresso:text_asr:roberta pitt:text_manual:bert pitt:text_manual:roberta
             pitt:text_asr:bert pitt:text_asr:roberta taukadial:text_asr:xlmr)
quiet() { grep -v "^20[0-9][0-9]-" || true; }

# --- acoustic LLD pair search: E0 replication, E2 uncertainty, E4 segments, E1 search-aware nulls
for fs in "${FEATURESETS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo "== $ds / $fs: E0 replicate";       sabench replicate "$ds" --featureset "$fs" --config "$CFG" | quiet
    echo "== $ds / $fs: E2 uncertainty";     sabench uncertainty "$ds" --featureset "$fs" --config "$CFG" | quiet
    echo "== $ds / $fs: E4 segment-channel"; sabench segment-channel "$ds" --featureset "$fs" --config "$CFG" | quiet
    for regime in test_informed train_only; do
      echo "== $ds / $fs: E1 null ($regime)"
      sabench null "$ds" --featureset "$fs" --regime "$regime" --config "$CFG" | quiet
    done
  done
done

# --- E3 speech-embedding probes: CV-selected layer, probe null, pair-search null inside that layer
for ds in "${DATASETS[@]}"; do
  for model in wavlm wav2vec2 whisper; do
    echo "== $ds: E3 speech probes ($model)"
    sabench probes "$ds" --model "$model" --probe-null --pair-search-null --regime train_only --config "$CFG" | quiet
  done
done

# --- E3 text probes: manual and ASR transcripts (BERT/RoBERTa for English, XLM-R for TAUKADIAL)
for spec in "${TEXT_PROBES[@]}"; do
  IFS=: read -r ds variant model <<< "$spec"
  echo "== $ds: E3 text probes ($variant / $model)"
  sabench probes "$ds" --model "$model" --variant "$variant" --probe-null --config "$CFG" | quiet
done

# --- seed robustness of the train_only LLD nulls
for seed in "${EXTRA_SEEDS[@]}"; do
  out="../run_full/results_seed$seed"
  for fs in "${FEATURESETS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      echo "== seed $seed: $ds / $fs: E1 null (train_only)"
      sabench null "$ds" --featureset "$fs" --regime train_only --config "$CFG" \
        --set "seed=$seed" --set "paths.results=$out" --set "logging.file=$out/logs/sabench.log" | quiet
    done
  done
done

# --- VAD robustness of E4: energy thresholds (challenges) and pyannote (all datasets)
for db in 30 40; do
  for ds in "${CHALLENGES[@]}"; do
    echo "== VAD -$db dB: $ds E4"; sabench segment-channel "$ds" --featureset eGeMAPSv02 --config "configs/full_vad$db.yaml" | quiet
  done
done
for fs in "${FEATURESETS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo "== pyannote VAD: $ds / $fs E4"; sabench segment-channel "$ds" --featureset "$fs" --config configs/full_pyannote.yaml | quiet
  done
done

# --- dataset-level analyses
echo "== base paper Table 4 replication (pyannote silence/speech)"
python -W ignore scripts/table4_replication.py configs/full_pyannote.yaml "$CFG" | quiet
echo "== E6 cross-dataset at none / session / speaker exclusion"
python -W ignore scripts/cross_dataset_dedup.py "$CFG" | quiet
echo "== K / seed sensitivity and inner-CV repeat stability"
python -W ignore scripts/sensitivity.py "$CFG" | quiet
python -W ignore scripts/repeats_stability.py "$CFG" | quiet
echo "== shortcut baselines (demographics, language) and ASR quality"
python -W ignore scripts/shortcut_baselines.py "$CFG" | quiet
python -W ignore scripts/asr_quality.py "$CFG" | quiet

echo "== all significance tests (Holm within family) and figures"
python scripts/summarize_tests.py "$TABLES"
sabench figures --config "$CFG" | quiet
