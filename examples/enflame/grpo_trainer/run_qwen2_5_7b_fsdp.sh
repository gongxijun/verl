#!/usr/bin/env bash
# GRPO on Enflame GCU with FSDP2 training + vLLM (vllm_gcu) rollout.
#
# Run on a machine with the TOPS runtime installed, e.g.:
#   source /opt/tops/bin/set_env.sh
#   bash examples/enflame/grpo_trainer/run_qwen2_5_7b_fsdp.sh
set -xeuo pipefail

# ---------------------------------------------------------------- paths
WORK_DIR=${WORK_DIR:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${WORK_DIR}/Qwen2.5-7B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${WORK_DIR}/datasets/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${WORK_DIR}/datasets/gsm8k/test.parquet"}

project_name="verl_grpo_qwen2_5_7b_gcu"
experiment_name="Qwen2_5_7B_Instruct_GCU"

# ---------------------------------------------------------------- algorithm
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ---------------------------------------------------------------- batch
train_batch_size=256
rollout_n=8
ppo_mini_batch_size=64

# ---------------------------------------------------------------- sequence length
max_prompt_length=1024
max_response_length=2048

learning_rate=1e-6

# ---------------------------------------------------------------- parallelism
sp_size=1
gen_tp=2
n_gpus_per_node=${n_gpus_per_node:-8}
offload=True

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_activation_offload=${offload}
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.nccl_timeout=14400

    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False

    actor_rollout_ref.actor.optim.lr=${learning_rate}

    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / sp_size))
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}

    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}

    actor_rollout_ref.actor.use_dynamic_bsz=True
    # torch.compile is not supported by the GCU backend yet.
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.load_format=auto
    # GCU's vLLM build does not support CUDA graph capture.
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.calculate_log_probs=True

    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1

    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / sp_size))
)

REF=(
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload}
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${offload}
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=False

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / sp_size))
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${experiment_name}"
    trainer.n_gpus_per_node=${n_gpus_per_node}
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=15
    # GCU uses PyTorch's PrivateUse1 backend, so the device name is "gcu".
    trainer.device=gcu
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# PPO (algorithm.adv_estimator=gae) additionally needs a critic. On GCU the value
# head must be initialized deterministically, otherwise ranks can start from
# different heads:
#   critic.model.value_head_seed=42
echo "Starting training: project=${project_name} exp=${experiment_name}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${ALGORITHM[@]}" \
    "$@"
