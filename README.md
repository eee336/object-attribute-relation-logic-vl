# OARL-VLA: Object-Attribute-Relation-Logic-Aware VLA

OARL-VLA is a minimal research prototype for studying why VLA systems fail on everyday robot instructions that require more than coarse object category recognition. The project tests an explicit object-attribute-relation-logic intermediate representation before a target-conditioned action policy.

普通 VLA 在真实指令中常常不是只输在空间关系上，还会输在实例绑定、属性/状态、类别常识、比较级、集合和成组对象、否定逻辑、模糊表达、历史指代、开放词表 grounding 与 affordance 推理上。例如 "Pick the banana that has not turned black" 需要状态建模；"Pick the farthest pair of shoes" 需要 group grounding；"Pick the object suitable for drinking coffee" 需要常识和 affordance。

This MVP has two parts:

1. Synthetic MVP: deterministic 2D/structured household scenes with gold labels for strict reasoning and grounding evaluation.
2. Web Image Dataset Builder: compliant local/Wikimedia-ready weak data pipeline with provenance, pseudo labels, quality scores, SFT/preference export, and human review HTML.

## Architecture

```mermaid
flowchart TD
    A[Image / Synthetic Scene / Web Image] --> B[Object-centric Perception]
    B --> C[Object List]
    C --> D[Attribute & State Modeling]
    D --> E[Category Taxonomy / Ontology]
    E --> F[Relation Graph]
    F --> G[Logic-aware Program Reasoner]
    G --> H[Target Object or Target Group Grounding]
    H --> I[Target-conditioned VLA Policy]
    I --> J[Simulated Action]
    J --> K[Reward Model]
    K --> L[Evaluation / Future RL]

    M[Web Image Sources] --> N[Web Dataset Builder]
    N --> O[Manifest + Provenance]
    O --> P[Pseudo Labels]
    P --> Q[Human Review HTML]
    P --> R[VLM SFT JSONL]
    P --> S[Preference / RL JSONL]
```

## Implemented Modules

- Core schemas: `ObjectInstance`, `ObjectGroup`, `Scene`, `SceneEvent`, `GroundingSample`.
- Taxonomy: fruit, drink, container, footwear, drinkware, utensil, electronics, readable object.
- Attribute/state rules: banana blackening/rotting/edibility, drink opened/fill/empty, cup cleanliness/broken/coffee suitability, shoe pair cleanliness/wearability.
- Relation helpers: left/right/above/below/near/far, nth from left/right, nearest/farthest, not near, between.
- Executable programs: natural language templates map to `ProgramStep` chains, then `ProgramExecutor` runs them over scene objects/groups.
- Logic reasoner: returns target id/type, executable program, reasoning trace, confidence, and failure reason.
- Baselines: random object, random same category, attribute-ignorant, relation-ignorant.
- Policy/reward: target-conditioned simulated grasp point and rule reward breakdown.
- Visualization: annotated PNG output, with matplotlib/Pillow path when installed and a pure-stdlib PNG fallback.
- Web data: local directory source, Wikimedia source, manifest, exact hash dedup, pseudo labeler, quality filter, review HTML, SFT and preference exports.

## Supported Instruction Types

- `spatial_relation`: nearest/farthest/left/right/between.
- `ordinal_relation`: nth object from left/right.
- `attribute_comparison`: largest, smallest, cleanest, dirtiest, fullest, emptiest.
- `state_filtering`: not blackened banana, blackened banana, unopened drink, not empty bottle.
- `category_taxonomy`: largest drink, edible fruit, cleanest drinkware.
- `group_grounding`: farthest/nearest/cleanest pair of shoes.
- `negation`: fruit not near trash bin, not opened drink, not empty bottle.
- `history_reference`: object just put down or moved most recently.
- `affordance`: object suitable for drinking coffee.

The first version uses English templates. The parser and generator are structured so Chinese templates can be added later.

## Categories, Attributes, States, Groups

Object categories include `apple`, `banana`, `orange`, `bottle`, `water_bottle`, `can`, `soda_can`, `juice_box`, `cup`, `mug`, `shoe`, `spoon`, `bowl`, `trash_bin`, `book`, and `remote`.

Attributes include size, color, shape, material, volume, liquid type, ripeness, black spot ratio, cleanliness, side, capacity, and brightness hooks.

States include `is_blackened`, `is_rotten`, `is_edible`, `is_opened`, `fill_level`, `is_empty`, `is_broken`, `is_usable`, and `is_wearable`.

Group types include `pair_of_shoes`, `stack_of_books`, and extension points for `set_of_cups` and `group_of_fruits`. Shoe-pair instructions target the group id, not an individual shoe.

## Installation

Python 3.10+ is recommended. The code is intentionally lightweight and the synthetic path can run without GPU or model weights.

```bash
cd object-attribute-relation-logic-vla
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On systems where `python` is not available, use `python3` in the commands below.

## Demo

```bash
python scripts/run_demo.py --seed 0 --instruction-type state_filtering
python scripts/run_demo.py --seed 1 --instruction-type attribute_comparison
python scripts/run_demo.py --seed 2 --instruction-type group_grounding
python scripts/run_demo.py --seed 3 --instruction-type negation
```

Demo PNGs are saved under `outputs/`, for example `outputs/demo_state_filtering_seed0.png`.

## Synthetic Benchmark

```bash
python scripts/run_benchmark.py \
  --num-scenes 100 \
  --objects-per-scene 12 \
  --seed 42
```

Outputs:

- `outputs/benchmark_results.json`
- `outputs/benchmark_results.csv`
- `outputs/example_success.png`
- `outputs/example_failure.png` if a logic failure occurs
- `outputs/example_attribute_task.png`
- `outputs/example_group_task.png`

Example small-run result from this environment: the logic reasoner reached 1.000 target accuracy on generated gold tasks; random/category-only/attribute-ignorant/relation-ignorant baselines were lower on the task types they ignore.

## Synthetic Dataset Export

```bash
python scripts/generate_dataset.py \
  --num-scenes 50 \
  --objects-per-scene 12 \
  --seed 42 \
  --output data/oarlvla_synthetic.jsonl
```

Each JSONL row contains a scene, instruction, executable program, gold target id/type, task type, object/group lists, and reasoning steps.

## Web Dataset Builder

The web pipeline is designed for provenance and review, not blind scraping. It supports:

- Local directory source: user-provided images.
- Wikimedia Commons API: open image search with license/author/source metadata when network and `requests` are available.
- Reserved extension points: Hugging Face datasets, Open Images, COCO/LVIS/Objects365, Unsplash API via `UNSPLASH_ACCESS_KEY`.

Configure query intents in `configs/web_queries.yaml`.

Local import:

```bash
python scripts/build_web_dataset.py \
  --source local \
  --input-dir tests/fixtures/images \
  --queries configs/web_queries.yaml \
  --output-dir data/web_dataset \
  --mode metadata_only
```

Wikimedia, if network is available:

```bash
python scripts/build_web_dataset.py \
  --source wikimedia \
  --queries configs/web_queries.yaml \
  --max-per-query 5 \
  --output-dir data/web_dataset \
  --mode metadata_only
```

If network or `requests` is unavailable, Wikimedia returns no records and the project remains usable through local source.

Outputs:

- `data/web_manifest.jsonl`
- `data/web_tasks.jsonl`
- `data/annotations/{image_id}.json`
- `data/oarlvla_web_sft.jsonl`
- `data/oarlvla_web_preferences.jsonl`
- `outputs/web_dataset_report.json`

## Compliance Notes

Do not indiscriminately crawl images. Use open datasets or open-license sources first. Each `WebImageRecord` stores source URL, license, author, query, download/import time, dimensions, sha256, split, and raw metadata. Do not download content that requires login/payment, violates site rules, or contains sensitive personal data such as face closeups, children, IDs, license plates, NSFW, violence, illegal content, or medical privacy material.

Real images and generated data files are ignored by `.gitignore`; only schema examples and tiny fixtures are committed.

## Label Quality

Synthetic gold labels are for strict evaluation, relation/logic module validation, and program executor tests.

Web weak labels are for visual diversity expansion, VLM SFT candidates, and human review queues. They must not be treated as final evaluation ground truth.

Model-assisted labels can add bbox/mask/candidate targets through future GroundingDINO, SAM/SAM2, Florence-2, OWL-ViT, CLIP, Qwen-VL, InternVL, or LLaVA integrations, but still require quality filtering.

Manual verified labels are the right source for final real-image eval sets or high-quality SFT.

## Pseudo Label Modes

- `metadata_only`: query and provenance metadata only.
- `heuristic`: metadata plus lightweight image statistics/filename/query rules.
- `model_assisted`: extension point; no large model is required in this MVP and it falls back safely.

Weak web tasks never invent object-level `target_id` without verified bbox/mask/candidate evidence. They use target descriptions and `requires_manual_verification=true`.

## Review HTML

```bash
python scripts/inspect_web_dataset.py \
  --manifest data/web_manifest.jsonl \
  --export-review-html outputs/review.html
```

The HTML shows the image, source, license, query, generated instruction, pseudo labels, quality score, and manual-review requirement.

## VLM SFT and Preference/RL Data

SFT rows are saved to `data/oarlvla_web_sft.jsonl`. Assistant messages contain a program, target description, confidence, and label quality, not fake ids.

Preference rows are saved to `data/oarlvla_web_preferences.jsonl`, with rule-generated chosen/rejected candidate programs. Weak preference data is suitable for pretraining/candidate filtering, not strict final evaluation.

The reward model exposes grounding, attribute, relation, action, success, and wrong-object terms. It can be used later for rejection sampling fine-tuning, DPO/preference optimization, PPO/GRPO, or trajectory-level RL.

## Active Query Suggestion

```bash
python scripts/suggest_queries.py \
  --benchmark-report outputs/benchmark_results.json \
  --existing-queries configs/web_queries.yaml \
  --output outputs/suggested_queries.yaml
```

The first version uses rules to suggest more data for low-accuracy or under-sampled task types such as blackened bananas, pair of shoes, coffee-suitable cups, and largest drinks.

## OARL-VLA Model Architecture

The trainable model in `src/oarlvla/models/` is a tiny symbolic OARL-VLA prototype. It is not a large-scale pretrained VLA. Its role is to verify that the project schema can drive a learnable loop:

```text
scene/image features + instruction + object tokens + relation graph
→ multimodal fusion
→ target grounding
→ program/task prediction
→ action prediction
→ multi-task loss
```

```mermaid
flowchart TD
    A[Instruction] --> B[Text Encoder]
    C[Image / Scene] --> D[Image or Symbolic Encoder]
    E[Object Features] --> F[Object Token Encoder]
    G[Relation Graph] --> H[Relation Graph Encoder]
    B --> I[Multimodal Fusion]
    D --> I
    F --> H
    H --> I
    I --> J[Target Grounding Head]
    I --> K[Program Head]
    J --> L[Selected Target Token]
    L --> M[Action Head]
    K --> N[Program Type / Logic Class]
    M --> O[Robot Action]
```

Current components:

- `vlm_backbone="tiny"` keeps the lightweight `SimpleTokenizer` plus GRU `TextEncoder`, with no large tokenizer dependency.
- `vlm_backbone="qwen_vl"` loads a Qwen-VL/Qwen2.5-VL style Hugging Face backbone through `QwenVLBackbone`; its pooled multimodal hidden state becomes the instruction/image embedding fused with object and relation tokens.
- `QwenVLProcessorAdapter` builds Qwen chat messages and uses `AutoProcessor`; when available, `qwen-vl-utils` handles image/video preprocessing.
- `ObjectEncoder` over a fixed 35-dimensional symbolic feature vector.
- Attribute/state inputs: `black_spot_ratio`, `ripeness`, `is_blackened`, `is_rotten`, `is_edible`, `volume_ml`, `fill_level`, `is_opened`, `is_empty`, `cleanliness`, `is_broken`, `is_wearable`.
- `SimpleRelationGraphEncoder`, a lightweight batched message-passing layer over `edge_index` and `edge_type`; it does not require PyG.
- `CrossAttentionFusion`, where object tokens attend to instruction/image context.
- `TargetGroundingHead`, producing `[batch, num_candidates]` target logits.
- `ProgramHead`, currently predicting task/program type classification rather than autoregressive programs.
- `ActionHead`, predicting `[x, y, gripper]` from the selected/fused target token.
- Multi-task loss: `target_loss + 0.5 * action_loss + 0.2 * program_loss`.

The raw image path is reserved through both `SimpleCNNImageEncoder` (`image_mode=cnn_stub`) and the Qwen-VL adapter. The default fast training path remains `vlm_backbone=tiny`, `image_mode=symbolic`, using synthetic scene object features.

Install PyTorch before model training:

```bash
pip install torch
```

or:

```bash
pip install -r requirements.txt
```

If PyTorch is unavailable, model scripts fail with a clear install message. GPU is not required; all scripts default to CPU.

For Qwen-VL-backed experiments, install the optional VLM dependencies and choose a Qwen model checkpoint:

```bash
pip install transformers qwen-vl-utils accelerate
```

Example Qwen-VL configuration:

```bash
python scripts/train_vla.py \
  --dataset data/oarlvla_synthetic.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --hidden-dim 128 \
  --vlm-backbone qwen_vl \
  --qwen-model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --output checkpoints/oarlvla_qwenvl_adapter.pt
```

This downloads/loads the Qwen checkpoint through Transformers. Use a small batch size on CPU; GPU or MPS is recommended for real Qwen-VL training. By default the Qwen-VL backbone is frozen and only the OARL-VLA projection/fusion/heads train.

### Train Tiny VLA

```bash
python scripts/generate_dataset.py \
  --num-scenes 200 \
  --objects-per-scene 12 \
  --seed 42 \
  --output data/oarlvla_synthetic.jsonl

python scripts/train_vla.py \
  --dataset data/oarlvla_synthetic.jsonl \
  --epochs 2 \
  --batch-size 16 \
  --hidden-dim 128 \
  --output checkpoints/oarlvla_tiny.pt
```

The checkpoint stores model weights, config, tokenizer vocabulary, feature metadata, and training history. Checkpoints are ignored by Git except `checkpoints/.gitkeep`.

### Stage 1 Grid/Cutout Pretraining

Stage 1 is a controllable 2D grid world with transparent household-object cutouts instead of text labels or letter placeholders. It is designed to train object binding, spatial/ordinal relations, group grounding, history reference, affordance, state filtering, and target-conditioned action before moving to noisy real RGB images.

The asset set covers common household objects such as apples, bananas, oranges, bottles, cups, mugs, shoes, spoons, bowls, trash bins, books, and remotes. By default, assets can be downloaded from Wikimedia Commons, converted into transparent cutouts, and tracked with source URL, license, author, and score metadata.

Each scene samples a fresh layout, object scale jitter, state combination, shoe-pair/book-stack placement, and optional distractor objects while preserving the relation constraints needed by the instruction templates.

Download and preprocess web cutout assets:

```bash
python scripts/download_grid_assets.py \
  --asset-dir data/grid_assets \
  --raw-dir data/grid_asset_raw \
  --manifest data/grid_assets_manifest.json \
  --candidates-per-query 5 \
  --sprite-size 192 \
  --force
```

Then generate a gold grid/cutout dataset:

```bash
python scripts/generate_grid_dataset.py \
  --num-scenes 1000 \
  --grid-size 8 \
  --cell-size 64 \
  --seed 42 \
  --output data/oarlvla_grid_sprites.jsonl \
  --image-dir data/grid_images \
  --asset-dir data/grid_assets
```

These commands write ignored local artifacts under `data/grid_images`, `data/grid_assets`, `data/grid_asset_raw`, `data/grid_assets_manifest.json`, and `data/oarlvla_grid_sprites.jsonl`.

Each row includes:

```text
image_path
objects / groups
instruction
program
target_id / target_type
target_bbox / target_center
label_quality=gold
source=synthetic_grid
```

Train and evaluate the Stage 1 VLA baseline:

```bash
python scripts/train_vla.py \
  --dataset data/oarlvla_grid_sprites.jsonl \
  --epochs 20 \
  --batch-size 32 \
  --hidden-dim 128 \
  --output checkpoints/oarlvla_grid_stage1.pt

python scripts/eval_vla.py \
  --dataset data/oarlvla_grid_sprites.jsonl \
  --checkpoint checkpoints/oarlvla_grid_stage1.pt
```

Last recorded Stage 1 baseline result on 1000 generated samples:

```text
Target Accuracy: 0.651
Program Accuracy: 0.659
Action MSE: 0.020135
```

Strong task slices include history reference, ordinal relation, affordance, and group grounding. Negation and some comparison/spatial cases remain useful targets for the next curriculum iteration.

### Evaluate Tiny VLA

```bash
python scripts/eval_vla.py \
  --dataset data/oarlvla_synthetic.jsonl \
  --checkpoint checkpoints/oarlvla_tiny.pt
```

The evaluator prints target accuracy, program accuracy, action MSE, task breakdown, and rule/baseline comparison.

### Tiny Overfit Sanity Check

```bash
python scripts/overfit_tiny_batch.py
```

This generates a tiny synthetic batch and trains until the target grounding head can overfit it. It is a fast sanity check for forward pass, loss, optimizer, dataset, collate, and checkpoint plumbing.

### Model Limits and Upgrade Path

Current implementation has two modes: a tiny symbolic mode for fast CPU tests and a Qwen-VL adapter mode for VLM-backed experiments. It is still not a pretrained robotics foundation model. It validates object tokens, attribute/state features, relation graph encoding, Qwen/text instruction encoding, target grounding, action prediction, and multi-task training.

Upgrade path:

- Use the Qwen-VL adapter as the default multimodal backbone, then fine-tune with LoRA/QLoRA instead of full-weight training.
- Replace or compare Qwen-VL with CLIP, SigLIP, DINOv2, InternVL, LLaVA, or a task-specific visual encoder.
- Feed object candidates from GroundingDINO/OWL-ViT plus SAM/SAM2 masks.
- Replace the MLP action head with diffusion policy, action tokenizer, or OpenVLA-style action decoder.
- Use web weak data for SFT/preference warm-up, then promote verified data into evaluation.
- Use the existing reward model for RL or preference post-training.

## Tests

```bash
pytest -q
```

Tests cover taxonomy, states, attributes, groups, relations, instruction generation, program execution, benchmark execution, visualization, manifest schema, query config loading, local directory import, sha256 dedup, quality filtering, pseudo labeling, web task export, review HTML, model forward shape, model backward training step, checkpoint save/load, and tiny-batch overfit. Network logic is not required by tests.

## Future VLM / VLA Integration

`src/oarlvla/interfaces.py` defines extension points for `ObjectDetector`, `VLMReasoner`, `ProgramGenerator`, `VLAActionPolicy`, `PreferenceDataBuilder`, and `RLTrainer`.

Suggested next integrations:

- GroundingDINO or OWL-ViT for open-vocabulary boxes.
- SAM/SAM2 for masks.
- Qwen-VL, InternVL, LLaVA, or Florence-2 for attribute/state descriptions.
- A learned program generator trained on synthetic gold programs.
- A real VLA policy that consumes image, object candidates, instruction, and grounded target.

## GitHub Push

Initialize and commit:

```bash
git init
git add .
git commit -m "Initial MVP for object-attribute-relation-logic-aware VLA"
```

If a remote already exists:

```bash
git branch -M main
git push -u origin main
```

If GitHub CLI is installed and authenticated:

```bash
gh repo create object-attribute-relation-logic-vla --public --source=. --remote=origin --push
```

Never commit API keys, tokens, cookies, SSH private keys, `.env`, model weights, virtual environments, caches, large datasets, or real image dumps.
