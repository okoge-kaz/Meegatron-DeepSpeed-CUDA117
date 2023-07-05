#!/bin/bash
#YBATCH -r rtx6000-ada_1
#SBATCH --job-name=tokenize
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err

set -e

source .env/bin/activate

python dataset/wikidump_download.py

bash dataset/merge_files.sh dataset/wikipedia/processed/en/AA dataset/wikipedia/merged/en en_merged_1

bash dataset/merge_files.sh dataset/wikipedia/processed/en/AB dataset/wikipedia/merged/en en_merged_2

bash dataset/merge_files.sh dataset/wikipedia/merged/en dataset/wikipedia/merged/en en_merged

export OUTDIR=dataset/wikipedia/binarized/gpt-2
mkdir -p $OUTDIR

# Tokenize and binarize English
python tools/preprocess_data.py \
  --input dataset/wikipedia/merged/en/en_merged.json \
  --output-prefix $OUTDIR/en_wiki \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --dataset-impl mmap \
  --tokenizer-type GPT2BPETokenizer \
  --workers 14 \
  --append-eod
