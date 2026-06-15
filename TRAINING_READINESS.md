# OARL-VLA Training Readiness

This file is the handoff checklist for moving the project to a compute machine.

## Current Scope

Final target for this phase:

1. **OARL-Bench**: benchmark and metric suite for wrong-object manipulation.
2. **OARL-VLA**: target-first trainable model with object/group tokens, attributes/states, relation graph, program supervision, target grounding, and action prediction.

Out of scope for this local machine:

- Heavy training runs.
- Large Qwen-VL checkpoint downloads.
- LIBERO / ManiSkill / robomimic environment installation.

These are prepared as optional next-stage integrations.

## Prepared Local Artifacts

Training bundle:

```text
data/training_bundle/
  raw/
    oarlvla_synthetic_full.jsonl
    oarlvla_grid_full.jsonl
  splits/
    synthetic_train.jsonl
    synthetic_val.jsonl
    synthetic_test.jsonl
    grid_train.jsonl
    grid_val.jsonl
    grid_test.jsonl
  grid_images/
  grid_assets/
  web_tasks.jsonl
  web_manifest.jsonl
  oarlvla_web_sft.jsonl
  oarlvla_web_preferences.jsonl
  training_manifest.json
  compute_training_commands.sh
```

Current generated size:

- Synthetic gold: 1000 samples.
- Grid/cutout gold: 1000 samples with images.
- Split ratio: 80/10/10 by task type.
- Web weak: local smoke-test source; replace or expand with Wikimedia / curated local images for stronger Stage-2.

Benchmark outputs:

```text
outputs/benchmark_results.json
outputs/benchmark_results.csv
outputs/benchmark_paper_tables.md
outputs/training_bundle_web_review.html
```

## Moving Training Data to a Compute Machine

Training data is intentionally ignored by git. Move it separately with `tar`,
`scp`, `rsync`, or regenerate it on the compute machine.

### Option A: Copy the Prepared Bundle

From the local machine:

```bash
cd object-attribute-relation-logic-vla
tar -czf oarlvla_training_bundle.tar.gz data/training_bundle
scp oarlvla_training_bundle.tar.gz USER@COMPUTE_HOST:/path/to/object-attribute-relation-logic-vla/
```

On the compute machine:

```bash
cd /path/to/object-attribute-relation-logic-vla
tar -xzf oarlvla_training_bundle.tar.gz
bash data/training_bundle/compute_training_commands.sh
```

For incremental sync:

```bash
rsync -av --progress data/training_bundle/ USER@COMPUTE_HOST:/path/to/object-attribute-relation-logic-vla/data/training_bundle/
```

### Option B: Regenerate a Larger Bundle on Compute

This is recommended for final experiments:

```bash
python3 scripts/prepare_training_bundle.py \
  --force \
  --synthetic-scenes 20000 \
  --grid-scenes 20000 \
  --benchmark-scenes 1000 \
  --build-web \
  --web-source wikimedia \
  --max-per-query 20 \
  --bundle-dir data/training_bundle_large
```

Then run the generated command script:

```bash
bash data/training_bundle_large/compute_training_commands.sh
```

## Compute Machine Setup

```bash
cd object-attribute-relation-logic-vla
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

For Qwen-VL experiments:

```bash
python3 -m pip install transformers qwen-vl-utils accelerate
```

## Training Order

Run the prepared script:

```bash
bash data/training_bundle/compute_training_commands.sh
```

Equivalent explicit order:

```bash
python3 scripts/run_benchmark.py --num-scenes 300 --objects-per-scene 12 --seed 42 --output-dir outputs

python3 scripts/train_vla.py \
  --dataset data/training_bundle/splits/grid_train.jsonl \
  --eval-dataset data/training_bundle/splits/grid_val.jsonl \
  --epochs 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --output checkpoints/oarlvla_grid_stage1.pt

python3 scripts/eval_vla.py \
  --dataset data/training_bundle/splits/grid_test.jsonl \
  --checkpoint checkpoints/oarlvla_grid_stage1.pt \
  --batch-size 32

python3 scripts/train_vla.py \
  --dataset data/training_bundle/splits/synthetic_train.jsonl \
  --web-weak-dataset data/training_bundle/web_tasks.jsonl \
  --eval-dataset data/training_bundle/splits/synthetic_val.jsonl \
  --epochs 5 \
  --batch-size 32 \
  --hidden-dim 128 \
  --init-checkpoint checkpoints/oarlvla_grid_stage1.pt \
  --extend-tokenizer \
  --freeze-modules object_encoder graph_encoder action_head \
  --output checkpoints/oarlvla_stage2_web_weak.pt

python3 scripts/eval_vla.py \
  --dataset data/training_bundle/splits/synthetic_test.jsonl \
  --checkpoint checkpoints/oarlvla_stage2_web_weak.pt \
  --batch-size 32
```

## Stage Freezing Policy

### Stage 0: OARL-Bench

- No neural training.
- Runs symbolic logic reasoner and non-learned baselines.
- Purpose: validate benchmark, task taxonomy, wrong-object metrics, and paper tables.

### Stage 1: Grid/Cutout Gold Pretraining

Command:

```bash
python3 scripts/train_vla.py \
  --dataset data/training_bundle/splits/grid_train.jsonl \
  --eval-dataset data/training_bundle/splits/grid_val.jsonl \
  --epochs 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --output checkpoints/oarlvla_grid_stage1.pt
```

Frozen modules: none.

Trainable modules:

- `text_encoder`
- `object_encoder`
- `graph_encoder`
- `fusion`
- `target_head`
- `program_head`
- `action_head`
- `global_norm`

Purpose: learn object token encoding, relation graph reasoning, target grounding, program/task classification, and target-conditioned action prediction from gold visual grid/cutout data.

### Stage 2: Synthetic + Web Weak Warm-Up

Command:

```bash
python3 scripts/train_vla.py \
  --dataset data/training_bundle/splits/synthetic_train.jsonl \
  --web-weak-dataset data/training_bundle/web_tasks.jsonl \
  --eval-dataset data/training_bundle/splits/synthetic_val.jsonl \
  --epochs 5 \
  --batch-size 32 \
  --hidden-dim 128 \
  --init-checkpoint checkpoints/oarlvla_grid_stage1.pt \
  --extend-tokenizer \
  --freeze-modules object_encoder graph_encoder action_head \
  --output checkpoints/oarlvla_stage2_web_weak.pt
```

Frozen modules:

- `object_encoder`
- `graph_encoder`
- `action_head`

Trainable modules:

- `text_encoder` (including extended tokenizer embeddings)
- `fusion`
- `target_head`
- `program_head`
- `global_norm`

Purpose: adapt language/program/target selection to broader synthetic instructions and weak web tasks without damaging the Stage-1 object/relation/action foundations. Web weak samples have `target_index=-1`, so they only supervise program/task prediction and do not become fake target ground truth.

### Optional Stage 3: Qwen-VL Adapter

Command template:

```bash
python3 scripts/train_vla.py \
  --dataset data/training_bundle/splits/grid_train.jsonl \
  --eval-dataset data/training_bundle/splits/grid_val.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --hidden-dim 128 \
  --vlm-backbone qwen_vl \
  --qwen-model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --output checkpoints/oarlvla_qwenvl_adapter.pt
```

Default frozen modules:

- The internal Qwen-VL base model is frozen by default through `freeze_qwen_vl=True`.

Trainable modules:

- Qwen projection/norm inside `QwenVLBackbone`
- `object_encoder`
- `graph_encoder`
- `fusion`
- `target_head`
- `program_head`
- `action_head`
- `global_norm`

To unfreeze the full Qwen-VL backbone, pass:

```bash
--unfreeze-qwen-vl
```

This is expensive and should only be used with sufficient GPU memory or LoRA/QLoRA-style future extensions.

## AAAI Ablations

```bash
python3 scripts/run_aaai_ablation_suite.py \
  --dataset data/training_bundle/splits/grid_train.jsonl \
  --eval-dataset data/training_bundle/splits/grid_val.jsonl \
  --epochs 5 \
  --batch-size 32 \
  --hidden-dim 128
```

This covers:

- `full`
- `no_relation_graph`
- `no_attribute_state`
- `no_group_candidates`
- `no_program_supervision`

## Metrics to Record

For each trained checkpoint, record:

- Target accuracy.
- Wrong-object rate, computed as `1 - target_accuracy` for gold target rows.
- Program accuracy.
- Action MSE.
- Per-task target accuracy.
- Group grounding accuracy.
- Attribute/state/relation accuracy.

Run:

```bash
python3 scripts/make_paper_tables.py \
  --input outputs/benchmark_results.json \
  --out-md outputs/benchmark_paper_tables.md
```

Model eval currently prints metrics. After the first compute run, copy the terminal metrics into `PAPER.md` and extend `make_paper_tables.py` if unified model-result JSON export is needed.

## Scaling Data Before Final Training

Recommended final data sizes before paper experiments:

- Synthetic gold: 20k-100k samples.
- Grid/cutout gold: 20k-100k samples.
- Web weak: at least 5k-20k reviewed candidates for SFT/preference warm-up.
- Verified real-image eval: 200-1000 manually verified examples with target object ids/bboxes.

Regenerate a larger bundle:

```bash
python3 scripts/prepare_training_bundle.py \
  --force \
  --synthetic-scenes 20000 \
  --grid-scenes 20000 \
  --benchmark-scenes 1000 \
  --build-web \
  --web-source wikimedia \
  --max-per-query 20 \
  --bundle-dir data/training_bundle_large
```

## Future Benchmark Integrations

Config file:

```text
configs/external_benchmarks.yaml
```

Planned order:

1. LIBERO: evaluate target grounding and wrong-object events on language-conditioned manipulation suites.
2. ManiSkill: use simulator metadata to supervise and evaluate object-logic target selection.
3. robomimic: export target-conditioned observation-action trajectories for offline imitation learning.

Keep these behind optional imports; do not make core OARL-Bench depend on heavy simulator packages.
