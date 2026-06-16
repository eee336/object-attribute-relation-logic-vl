#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a StarVLA checkout after running:
#   python /path/to/object-attribute-relation-logic-vla/integrations/starvla/install_overlay.py --starvla-root .

export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}

OARLVLA_REPO=${OARLVLA_REPO:-/path/to/object-attribute-relation-logic-vla}
export PYTHONPATH="${OARLVLA_REPO}/src:${PYTHONPATH:-}"

Framework_name=${Framework_name:-OARLVLAQwenPI}
freeze_module_list=${freeze_module_list:-qwen_vl_interface}
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-./examples/LIBERO/train_files/oarlvla_qwenpi_libero.yaml}
libero_data_root=${libero_data_root:-playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${data_mix:-libero_all}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-oarlvla_qwenpi_libero}
num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT:-oarlvla_libero}" \
  --wandb_entity "${WANDB_ENTITY:-your_wandb_entity}"
