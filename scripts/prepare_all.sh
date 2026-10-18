#!/usr/bin/env bash
# Build every input the analyses need, from the raw data in ../Data, into ../run_full.
# Usage (from the repository root, venv active):  bash scripts/prepare_all.sh
# Override locations with DATA=/path/to/Data RUN=/path/to/run_full (and update paths.* in the configs).
# Needs: `hf auth login` with the pyannote/segmentation-3.0 terms accepted (pyannote VAD), and
# network access on first run (model downloads). Then run scripts/run_final.sh.
set -euo pipefail

RUN=${RUN:-../run_full}
DATA=${DATA:-../Data}
DATASETS=(adress adresso pitt taukadial)
FEATURESETS=(eGeMAPSv02 ComParE_2016)
quiet() { grep -v "^20[0-9][0-9]-\|Loading weights\|LOAD REPORT\|UNEXPECTED\|generation_config\|clean_up_tokenization" || true; }

echo "== manifests and source links"
mkdir -p "$RUN"
DATA_ABS=$(cd "$DATA" && pwd)
python scripts/build_manifests.py "$DATA" "$RUN"
ln -sfn "$DATA_ABS/ADReSS2020/ADReSS-IS2020-data" "$RUN/adress"
ln -sfn "$DATA_ABS/AddReSSo2021/diagnosis" "$RUN/adresso"
ln -sfn "$DATA_ABS/Pitt/media" "$RUN/pitt"
ln -sfn "$DATA_ABS/TAUKADIAL" "$RUN/taukadial"
python -W ignore scripts/match_pitt.py "$RUN" "$DATA/Pitt" --adresso-meta "$DATA/AddReSSo2021/MMSE scores/dx-mmse.csv"
python -W ignore scripts/detect_language.py configs/full.yaml taukadial

for cfg in configs/full.yaml configs/full_pyannote.yaml; do
  for ds in "${DATASETS[@]}"; do
    echo "== $cfg: $ds audio variants and LLD features"
    sabench prepare-audio "$ds" --config "$cfg" 2>&1 | quiet
    for fs in "${FEATURESETS[@]}"; do
      for v in full speech silence; do sabench extract-lld "$ds" --featureset "$fs" --variant "$v" --config "$cfg" 2>&1 | quiet; done
    done
  done
done
for db in 30 40; do
  for ds in adress adresso; do
    echo "== energy VAD -$db dB: $ds"
    sabench prepare-audio "$ds" --config "configs/full_vad$db.yaml" 2>&1 | quiet
    for v in full speech silence; do sabench extract-lld "$ds" --featureset eGeMAPSv02 --variant "$v" --config "configs/full_vad$db.yaml" 2>&1 | quiet; done
  done
done

for ds in "${DATASETS[@]}"; do
  for model in wavlm wav2vec2 whisper; do
    echo "== $ds: speech embeddings ($model)"; sabench extract-embeddings "$ds" --model "$model" --config configs/full.yaml 2>&1 | quiet
  done
done

echo "== transcripts: manual (ADReSS, Pitt) and Whisper ASR (all; TAUKADIAL per speaker language)"
python scripts/chat_transcripts.py configs/full.yaml "$DATA"
for ds in "${DATASETS[@]}"; do sabench transcribe "$ds" --config configs/full.yaml 2>&1 | quiet; done
# English-only sets: BERT and RoBERTa; TAUKADIAL (English + Mandarin): multilingual XLM-R.
for spec in adress:manual:bert adress:manual:roberta adress:asr:bert adress:asr:roberta \
            adresso:asr:bert adresso:asr:roberta pitt:manual:bert pitt:manual:roberta \
            pitt:asr:bert pitt:asr:roberta taukadial:asr:xlmr; do
  IFS=: read -r ds src model <<< "$spec"
  echo "== $ds: text embeddings ($src / $model)"
  sabench extract-text-embeddings "$ds" --source "$src" --model "$model" --config configs/full.yaml 2>&1 | quiet
done
