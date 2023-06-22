CHECKPOINT_PATH=checkpoints/124m_dp1024
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt

MAX_OUTPUT_SEQUENCE_LENGTH=1024
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json
INPUT_PREFIX=dataset

python tools/generate_samples_gpt.py \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --micro-batch-size 1 \
  --global-batch-size 1024 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
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
