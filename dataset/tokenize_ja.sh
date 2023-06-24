#!/bin/bash

set -e

source .env/bin/activate

export OUTDIR=dataset/wikipedia/binarized/gpt-2
mkdir -p $OUTDIR

# Tokenize and binarize Japanese
python tools/preprocess_data.py \
  --input dataset/wikipedia/merged/ja/ja_merged.json \
  --output-prefix $OUTDIR/ja_wiki \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --dataset-impl mmap \
  --tokenizer-type GPT2BPETokenizer \
  --workers 14 \
  --append-eod
