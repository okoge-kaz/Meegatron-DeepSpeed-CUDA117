#!/bin/bash
#YBATCH -r dgx-a100_1
#SBATCH --job-name=text-generation
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

source .env/bin/activate

CHECKPOINT_PATH=checkpoints/gpt-fugaku-cpu/123m_dp4_ja
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt

MAX_OUTPUT_SEQUENCE_LENGTH=256
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=10
OUTPUT_FILE="fugaku_123m_dp4.json"
INPUT_PREFIX=dataset

python tools/generate_samples_gpt.py \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --seq-length 256 \
  --max-position-embeddings 256 \
  --vocab-file $INPUT_PREFIX/$VOCAB_FILE \
  --merge-file $INPUT_PREFIX/$MERGE_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --load $CHECKPOINT_PATH \
  --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
  --temperature $TEMPERATURE \
  --genfile $OUTPUT_FILE \
  --num-samples $NUMBER_OF_SAMPLES \
  --top_p $TOP_P \
  --recompute
