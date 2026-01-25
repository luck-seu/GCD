#!/bin/bash
# Training script for Conflict Detection with MARTI

set -x

export WANDB_MODE="offline"
export WANDB_DIR="/seu_share/home/220242328/wandb_offline_logs"
WANDB_KEY="65677e742de2e88a9dee16ced38facdad647c3fd"
module load cuda-12.4
module load gcc-9.3.0
# Start Ray cluster
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# Configuration
MODEL_DIR="/seu_share/home/220242328/hf_cache/models/Qwen"
ROOT_DIR=$(pwd)
SHARE_ROOT_DIR="/seu_share/home/220242328/marti_outputs"

DATE=$(date +%m%d_%H%M)
ADVANTAGE="reinforce"
SHORT_NAME="Qwen3-1.7B"
TASK="confiqa"
ALGO="train-2step"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}-TEST"

SAVE_PATH="${SHARE_ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}-${DATE}-TEST/${SHORT_NAME}"

PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"
CKPT_PATH="${SHARE_ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}//${DATE}/${SHORT_NAME}/ckpt"
WORKFLOW_FUNC_PATH="${ROOT_DIR}/marti/worlds/workflows/conflict_workflow.py"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export RAY_PICKLE_VERBOSE_DEBUG=1
export CUDA_HOME="/seu_share/apps/cuda-12.4"
export LD_LIBRARY_PATH=${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Ray job submission environment
ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["data/", "outputs/", ".git/", "local/", "logs/", "scsubmit/"],
  "pip": ["hydra-core"]
}
EOF
)

# Submit training job
ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name "conflict_detection_test" \
    eval_only=True \
    eval_workers=-1 \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=1 \
    default_agent.critic_num_nodes=1 \
    default_agent.critic_num_gpus_per_node=1 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=1 \
    default_agent.vllm_num_engines=1 \
    default_agent.vllm_gpu_memory_utilization=0.9 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.ckpt_path="${CKPT_PATH}" \
    default_agent.advantage_estimator="${ADVANTAGE}" \
    extra_eval_tasks=["macnoiselongpre0","macnoiselongpre1","macnoiselongpre2"] \
    extra_eval_dir="${ROOT_DIR}/data/Bench" \
    workflow_func_path="${WORKFLOW_FUNC_PATH}" \
    use_wandb="${WANDB_KEY}" \
    wandb_project="MARTI-Conflict-TEST" \
    wandb_run_name="${EXP}" \
    prompt_data="${PROMPT_DATA}" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/std/${DATE}-${EXP}.log" \

echo "Model Training Finished. Shutting down..."
