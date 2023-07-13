#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=gpt
#SBATCH --time=1-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

MASTER_NODE=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

source .env/bin/activate

# distributed settings
GPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP_SIZE=4
PP_SIZE=2
DP_SIZE=$(($WORLD_SIZE / ($TP_SIZE * $PP_SIZE)))

echo "TP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE, DP_SIZE: $DP_SIZE"

# model hyperparameters
NUM_LAYERS=32
HIDDEN_SIZE=2560
NUM_ATTN_HEADS=32
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=2048

# dataset, checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2_1.3b/${NNODES}node-${WORLD_SIZE}gpu-dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun

mkdir -p $CHECKPOINT_PATH

MICRO_BATCHSIZE=8
GLOBAL_BATCHSIZE=$(($MICRO_BATCHSIZE * $DP_SIZE))

# nvlink
nvidia-smi nvlink --status

# Open MPI training

mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_NODE \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO  -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python pretrain_gpt.py \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PP_SIZE \
  --num-layers $NUM_LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $NUM_ATTN_HEADS \
  --micro-batch-size $MICRO_BATCHSIZE \
  --global-batch-size $GLOBAL_BATCHSIZE \
  --seq-length $SEQ_LENGTH \
  --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --save-interval 10000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
  --use-mpi \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --wandb-name "gpt2_1.3b_${NNODES}node_dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun-ylab"
