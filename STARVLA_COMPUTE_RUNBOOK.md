# StarVLA Compute Runbook for OARL-VLA

This document is written for the compute-machine Codex run. Give Codex this file and ask it to execute the checklist end to end.

Goal:

```text
Run OARL-VLA as a full StarVLA framework variant:
Qwen-VL backbone
  -> OARLReasoningCore
  -> Target Grounding Bottleneck
  -> target-conditioned StarVLA/QwenPI flow action head
  -> LIBERO training/evaluation
```

OARL-VLA is not a StarVLA plug-in. StarVLA is used as the trainer/dataloader/LIBERO evaluation substrate; `OARLVLAQwenPI` is our model framework variant.

References:

- StarVLA repository: https://github.com/starVLA/starVLA
- StarVLA LIBERO quick start: https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/starVLA_guideline.md
- StarVLA framework API location: `starVLA/model/framework/VLM4A/*.py`
- OARL-VLA overlay files in this repo: `integrations/starvla/`

## Codex Task Prompt

Use this prompt on the compute machine:

```text
You are working on the compute machine. Use STARVLA_COMPUTE_RUNBOOK.md as the source of truth.

Objectives:
1. Clone StarVLA and install its environment.
2. Install the OARL-VLA StarVLA overlay.
3. Verify that the OARLVLAQwenPI framework is registered and constructible.
4. Download LIBERO LeRobot-format data and Qwen-VL backbone.
5. Run a small debug training job first.
6. If the debug job passes, launch the full LIBERO training job.
7. Run LIBERO evaluation from StarVLA examples.
8. Save logs, configs, git commits, and a concise status report.

Do not replace OARL-VLA with a StarVLA baseline. OARL-VLA must remain the model:
Qwen-VL -> OARLReasoningCore -> target bottleneck -> flow action policy.
```

## Phase 0: Repositories

```bash
export OARLVLA_REPO=/path/to/object-attribute-relation-logic-vla
git clone https://github.com/starVLA/starVLA.git /path/to/starVLA
cd /path/to/starVLA
git checkout starVLA
```

Use `starVLA` for stable results. Use `starVLA_dev` only if stable branch lacks required Qwen/LIBERO support.

## Phase 1: Environment

Follow StarVLA's own install instructions:

```bash
cd /path/to/starVLA
conda create -n starVLA python=3.10 -y
conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
pip install -e "${OARLVLA_REPO}"
export PYTHONPATH="${OARLVLA_REPO}/src:${PYTHONPATH}"
```

If `flash-attn` fails, inspect:

```bash
nvcc -V
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
```

Then install a flash-attn wheel matching CUDA/PyTorch.

## Phase 2: Install OARL-VLA Overlay

From the OARL-VLA repo:

```bash
python "${OARLVLA_REPO}/integrations/starvla/install_overlay.py" \
  --starvla-root /path/to/starVLA \
  --oarlvla-root "${OARLVLA_REPO}" \
  --force
```

This copies:

```text
integrations/starvla/OARLVLAQwenPI.py
  -> starVLA/model/framework/VLM4A/OARLVLAQwenPI.py

integrations/starvla/oarlvla_qwenpi_libero.yaml
  -> examples/LIBERO/train_files/oarlvla_qwenpi_libero.yaml

integrations/starvla/run_oarlvla_libero_train.sh
  -> examples/LIBERO/train_files/run_oarlvla_libero_train.sh
```

Smoke check:

```bash
cd /path/to/starVLA
python - <<'PY'
from oarlvla.models.oarl_core import OARLReasoningCore
print("OARL core import OK:", OARLReasoningCore)
PY
```

## Phase 3: Data and Backbone

Prepare LIBERO data using StarVLA's script:

```bash
cd /path/to/starVLA
export DEST=/path/to/data
bash examples/LIBERO/data_preparation.sh
```

Download Qwen backbone:

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

If GPU memory is tight, switch to a smaller StarVLA-supported Qwen backbone and update:

```yaml
framework:
  qwenvl:
    base_vlm: ...
```

## Phase 4: Debug Training

Run a tiny job first:

```bash
cd /path/to/starVLA
export OARLVLA_REPO=/path/to/object-attribute-relation-logic-vla
export PYTHONPATH="${OARLVLA_REPO}/src:${PYTHONPATH}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/oarlvla_qwenpi_libero.yaml \
  --framework.name OARLVLAQwenPI \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 20 \
  --trainer.logging_frequency 1 \
  --trainer.freeze_modules qwen_vl_interface \
  --run_root_dir playground/Checkpoints \
  --run_id debug_oarlvla_qwenpi
```

Expected signs:

- Framework registry finds `OARLVLAQwenPI`.
- Parameter groups include `oarl_core`.
- Loss dict includes `action_loss`; `target_loss` appears only when examples include `oarl_target_index`.
- A checkpoint appears under `playground/Checkpoints/debug_oarlvla_qwenpi`.

## Phase 5: Full Training

```bash
cd /path/to/starVLA
export OARLVLA_REPO=/path/to/object-attribute-relation-logic-vla
export PYTHONPATH="${OARLVLA_REPO}/src:${PYTHONPATH}"
export WANDB_PROJECT=oarlvla_libero
export WANDB_ENTITY=your_wandb_entity

bash examples/LIBERO/train_files/run_oarlvla_libero_train.sh
```

Default behavior:

- Framework: `OARLVLAQwenPI`
- Data mix: `libero_all`
- Qwen backbone frozen: `qwen_vl_interface`
- Trainable: `oarl_core`, `action_model`, and unfrozen heads/modules

Important:

The first overlay version supports StarVLA's existing LIBERO examples. If LIBERO batches do not contain object proposals, `fallback_single_scene_token=true` creates one zero scene token so the whole pipeline trains. This is for plumbing and baseline migration only. For the actual paper, add object proposals and target labels.

## Phase 6: Add LIBERO Object Tokens

For paper-grade OARL-VLA, extend the StarVLA LIBERO dataloader or postprocessor so each example may include:

```python
example["oarl_object_features"]        # np.ndarray [N, 35]
example["oarl_relation_edges"]         # np.ndarray [E, 2]
example["oarl_relation_types"]         # np.ndarray [E]
example["oarl_target_index"]           # int, optional target supervision
example["oarl_object_region_features"] # np.ndarray [N, region_dim], optional
```

Start with simulator metadata:

- Object names/categories from LIBERO task metadata.
- Object poses/bboxes projected into camera frame when available.
- Spatial relations from object centers.
- Target index from task language/template metadata.

Then switch:

```yaml
framework:
  oarl:
    fallback_single_scene_token: false
    region_feature_dim: 0
```

The model will then require real object candidates, which is the correct final paper setting.

## Phase 7: Evaluation

Use StarVLA's LIBERO evaluation scripts:

```bash
cd /path/to/starVLA
bash examples/LIBERO/eval_files/install_libero.sh
bash examples/LIBERO/eval_files/eval_libero.sh
```

Set the checkpoint path in the StarVLA eval script to the trained OARL-VLA checkpoint. Also record:

- LIBERO success rate.
- Wrong-object diagnostics where target labels are available.
- OARL target accuracy on OARL-Bench/grid held-out data.

## Required Paper Comparisons

Train/evaluate these under the same StarVLA setup:

```text
QwenPI                  # StarVLA direct flow policy baseline
QwenOFT                 # StarVLA direct continuous action baseline
OARLVLAQwenPI           # our full model
OARLVLAQwenPI w/o graph # set framework.oarl.use_relation_graph=false
OARLVLAQwenPI w/o target labels, if target labels are available
```

## Deliverables From Compute Codex

At the end, create:

```text
outputs/starvla_oarlvla_status.md
outputs/starvla_oarlvla_train_logs/
outputs/starvla_oarlvla_eval_results/
```

The status file must include:

- StarVLA commit hash.
- OARL-VLA commit hash.
- Environment summary: GPU, CUDA, torch, transformers, flash-attn.
- Dataset paths and data mix.
- Framework config.
- Debug job result.
- Full training command and checkpoint path.
- Evaluation command and metrics.
- Known failures/blockers.
