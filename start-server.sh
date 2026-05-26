#!/bin/bash

# vLLM server startup script

if [ $# -ne 4 ]; then
    echo "Usage: $0 <SCHEDULING_POLICY> <CUDA_VISIBLE_DEVICES> <PORT> <MODEL_PATH>"
    echo "Example: $0 ua 4,5 16666 /data/models/Meta-Llama-3-8B-Instruct"
    echo ""
    echo "Available scheduling policies: fcfs, ssjf, ua, ltr, gmm_ua"
    exit 1
fi

export SCHEDULING_POLICY=$1
export CUDA_VISIBLE_DEVICES=$2
export PORT=$3
export MODEL_PATH=$4


export NCCL_P2P_LEVEL=NVL
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# Validate scheduling policy
case $SCHEDULING_POLICY in
    fcfs|ssjf|ua|ltr|gmm_ua)
        ;;
    *)
        echo "Error: unknown scheduling policy '$SCHEDULING_POLICY'"
        echo "Available policies: fcfs, ssjf, ua, ltr, gmm_ua"
        exit 1
        ;;
esac

# Validate port
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: invalid port '$PORT' (must be 1024-65535)"
    exit 1
fi

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model path does not exist '$MODEL_PATH'"
    exit 1
fi

VISIBLE_GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Predictor-based policies reserve one GPU for the predictor process
if [ "$SCHEDULING_POLICY" = "ua" ] || [ "$SCHEDULING_POLICY" = "gmm_ua" ]; then
    TP_SIZE=$((VISIBLE_GPU_COUNT - 1))
    # Predictor uses the last visible GPU (0-indexed)
    export UA_PREDICTOR_GPU=$((VISIBLE_GPU_COUNT - 1))
    UA_PHYSICAL_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$VISIBLE_GPU_COUNT)
    echo "UA mode: TP_SIZE=$TP_SIZE, predictor physical GPU=$UA_PHYSICAL_GPU"
elif [ "$SCHEDULING_POLICY" = "ltr" ]; then
    TP_SIZE=$((VISIBLE_GPU_COUNT - 1))
    export LTR_PREDICTOR_GPU=$((VISIBLE_GPU_COUNT - 1))
    LTR_PHYSICAL_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$VISIBLE_GPU_COUNT)
    echo "LTR mode: TP_SIZE=$TP_SIZE, predictor physical GPU=$LTR_PHYSICAL_GPU"
elif [ "$SCHEDULING_POLICY" = "ssjf" ]; then
    TP_SIZE=$((VISIBLE_GPU_COUNT - 1))
    export SSJF_PREDICTOR_GPU=$((VISIBLE_GPU_COUNT - 1))
    SSJF_PHYSICAL_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$VISIBLE_GPU_COUNT)
    echo "SSJF mode: TP_SIZE=$TP_SIZE, predictor physical GPU=$SSJF_PHYSICAL_GPU"
else
    TP_SIZE=$VISIBLE_GPU_COUNT
fi

export RAY_DEDUP_LOGS=1

# MAX_NUM_SEQS corresponds to B in the adaptive beta formula (paper Eq.12)
# Must match GPU_BATCH_SIZE in ua_predictor.py, passed via UA_GPU_BATCH_SIZE env var
MAX_NUM_SEQS=32
export UA_GPU_BATCH_SIZE=$MAX_NUM_SEQS

ENFORCE_EAGER_FLAG=""
if [[ "$MODEL_PATH" == *"Qwen3-Next-80B-A3B-Instruct"* ]]; then
    ENFORCE_EAGER_FLAG="--enforce-eager"
    echo "Detected Qwen3-Next-80B-A3B-Instruct, enabling --enforce-eager"
fi

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --trust-remote-code \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.91 \
    --host 0.0.0.0 \
    --port $PORT \
    --disable-uvicorn-access-log \
    --scheduling-policy $SCHEDULING_POLICY \
    --max-model-len 8192 \
    --dtype auto \
    --no-enable-prefix-caching \
    --max-num-seqs $MAX_NUM_SEQS \
    $ENFORCE_EAGER_FLAG
