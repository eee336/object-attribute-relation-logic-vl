# AAAI Submission Plan for OARL-VLA

This document tracks how to reshape the current OARL-VLA prototype into an AAAI-style submission.

## Target Venue

Target: AAAI-27 Main Technical Track.

Important constraints from the official AAAI-27 call:

- Abstract deadline: 2026-07-21, 11:59 PM UTC-12.
- Full paper deadline: 2026-07-28, 11:59 PM UTC-12.
- Supplementary material and code deadline: 2026-07-31, 11:59 PM UTC-12.
- Main paper: up to 7 pages of technical content plus references.
- Critical evidence must be in the main paper; supplementary material is optional and reviewers are not required to read it.

Official links:

- AAAI-27 Main Technical Track Call: https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- AAAI-26 submission instructions, still useful for formatting/reproducibility details: https://aaai.org/conference/aaai/aaai-26/submission-instructions/
- AAAI reproducibility checklist: https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/

## Recent AAAI VLA Landscape

Recent AAAI VLA papers cluster around several themes:

| Theme | Representative AAAI papers | What they optimize | Gap for OARL-VLA |
|---|---|---|---|
| Ambiguous instruction reasoning and 3D interaction | GraphCoT-VLA, AAAI-26 | 3D pose-object graph, structured CoT, task planning under ambiguous instructions | Mostly spatial/3D and task planning; less explicit coverage of attributes, object states, negation, group grounding, and wrong-object metrics |
| Failure recovery | TCoT, AAAI-26 | trajectory-level planning and recovery after failure | Focuses on recovering from failed trajectories; OARL should prevent the wrong target before acting |
| Action-space alignment | OC-VLA, AAAI-26 | predicting actions in camera/observation space instead of robot-base space | Addresses action coordinate mismatch; OARL addresses language-to-target mismatch |
| Efficient VLA deployment | VLA-Adapter, MoLe-VLA, TTF-VLA, AAAI-26 | smaller backbones, layer skipping, token fusion, inference speed | Strong engineering baseline, but not primarily a benchmark of target grounding correctness |
| Perception/semantic alignment | ReconVLA, CCoL, AAAI-26 | visual representation reconstruction, semantic-physical alignment | Related, but still mainly evaluates downstream task success rather than wrong-object manipulation |
| Dexterous manipulation | DexGraspVLA, AAAI-26 | dexterous grasp execution and general grasping | Complementary: better target grounding can feed dexterous policies |

Primary sources:

- GraphCoT-VLA: https://ojs.aaai.org/index.php/AAAI/article/view/38896
- TCoT: https://ojs.aaai.org/index.php/AAAI/article/view/37577
- OC-VLA: https://ojs.aaai.org/index.php/AAAI/article/view/38947
- VLA-Adapter: https://ojs.aaai.org/index.php/AAAI/article/view/38931
- ReconVLA: https://ojs.aaai.org/index.php/AAAI/article/view/38921
- MoLe-VLA: https://ojs.aaai.org/index.php/AAAI/article/view/38945
- TTF-VLA: https://ojs.aaai.org/index.php/AAAI/article/view/38910
- CCoL: https://ojs.aaai.org/index.php/AAAI/article/view/39677
- DexGraspVLA: https://ojs.aaai.org/index.php/AAAI/article/view/38953

## Recommended AAAI Positioning

Do not pitch this as "a new VLA model" only. AAAI reviewers will compare it against a crowded VLA-model literature.

Recommended thesis:

> Current VLA evaluation often reports task success, but under compositional instructions a robot may execute a plausible action on the wrong object. We formalize wrong-object manipulation as a measurable failure mode and propose OARL-Bench plus an object-logic-aware grounding-and-action framework to diagnose and reduce it.

The core paper should be:

1. A benchmark/problem paper with a trainable method, not only a system paper.
2. A target-grounding paper, not an action-efficiency paper.
3. A diagnostic and mitigation paper for wrong-object manipulation.

## Contributions to Claim

Use exactly three or four crisp contributions:

1. **Problem and Metric**: define wrong-object manipulation for VLA and separate target grounding correctness from downstream action success.
2. **OARL-Bench**: introduce an object-attribute-relation-logic benchmark covering state, attribute, comparison, negation, group, affordance, history, fuzzy, and open-vocabulary instructions.
3. **OARL-VLA Framework**: propose object/group tokens, attribute-state features, relation graph encoding, executable program supervision, target grounding head, and target-conditioned action head.
4. **Data/Training Pipeline**: combine synthetic gold data, grid/cutout stage-1 data, and weak web data without treating unverified weak labels as target ground truth.

Avoid overclaiming:

- Do not claim large-scale robot foundation model performance unless we run real VLA baselines.
- Do not claim real-world manipulation success unless real robot or accepted simulator experiments are added.
- Do not claim final open-world perception solved; web weak labels are review candidates.

## Experiments Required for a Serious AAAI Submission

Minimum acceptable package:

1. **OARL-Bench symbolic benchmark**
   - Methods: random, category-only, attribute-ignorant, relation-ignorant, rule OARL, learned OARL.
   - Metrics: target accuracy, wrong-object rate, task success, per-task accuracy.
   - Current status: implemented.

2. **Stage-1 grid/cutout learned VLA**
   - Train tiny OARL-VLA on generated grid/cutout data.
   - Report target accuracy, program accuracy, action MSE, per-task breakdown.
   - Current status: data/training/eval scripts implemented; needs torch run on local/GPU environment.

3. **Ablations**
   - w/o relation graph.
   - w/o attribute/state features.
   - w/o group candidates.
   - w/o program supervision.
   - Current status: training flags and `scripts/run_aaai_ablation_suite.py` are implemented; needs torch execution and result logging.

4. **VLM/VLA comparison**
   - Qwen-VL direct target answer on grid images.
   - Qwen-VL direct policy without OARL target bottleneck.
   - Full OARL-VLA-Qwen with OARL reasoning core and target-conditioned flow policy.
   - OpenVLA-style or VLA-Adapter/OpenVLA baseline if feasible.
   - Current status: Qwen-VL-backed OARL-VLA path exists; direct VLM/direct policy baselines are not yet implemented.

5. **StarVLA / LIBERO execution path**
   - Implement `OARLVLAQwenPI` as a StarVLA framework variant.
   - Use StarVLA for LIBERO dataloaders, action normalization, training, checkpointing, and evaluation.
   - Keep OARL-VLA as the model architecture: `Qwen-VL -> OARLReasoningCore -> Target Grounding Bottleneck -> flow action policy`.
   - Current status: overlay files and compute runbook are implemented; compute-machine execution pending.

6. **Real-image verified subset**
   - Build 200-1000 manually verified examples from web/local images.
   - Must include bbox/target id for strict evaluation.
   - Current status: web weak builder exists; manual verified evaluation set missing.

7. **Reproducibility package**
   - One command for dataset generation/training/eval.
   - Seeds and configs recorded.
   - Current status: `scripts/run_stage_pipeline.py` exists; needs torch execution logs and final result tables.

## Main Paper Shape for 7 Pages

Recommended allocation:

- 0.5 page: abstract + problem teaser.
- 1 page: introduction and wrong-object manipulation motivation.
- 1 page: related work, compressed into categories.
- 1.5 pages: OARL representation, OARL-Bench, metrics.
- 1 page: OARL-VLA method.
- 1.5 pages: experiments and ablations.
- 0.5 page: limitations, reproducibility, conclusion.

Main figure priority:

1. Failure case: right action, wrong object.
2. OARL-VLA architecture.
3. Task taxonomy and metric decomposition.
4. Result table/bar chart.

## Immediate Engineering Roadmap

Priority 0: make the current repository trainable on the target machine.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_stage_pipeline.py --stage all --quick
```

Priority 1: produce the first AAAI result table.

```bash
python3 scripts/run_stage_pipeline.py --stage stage0 --force
python3 scripts/run_stage_pipeline.py --stage stage1 --quick
python3 scripts/run_stage_pipeline.py --stage stage2 --quick
```

Priority 2: add learned ablations.

- Implemented model/data/training flags for relation graph, attribute-state channels, group candidates, and program loss.
- Use `scripts/run_aaai_ablation_suite.py` to run each setting and export command logs.
- Next: run with torch installed and aggregate metrics into the paper table.

Priority 3: add VLM direct grounding baseline.

- Feed grid images + instruction to Qwen-VL.
- Ask for target id/description.
- Parse answer against candidate ids/categories.
- Report wrong-object rate.

Priority 4: build verified real-image OARL-Bench subset.

- Use web/local builder only for candidate collection.
- Add manual annotation JSON with bbox, category, attributes/states, target id.
- Do not evaluate on weak labels.

## Reviewer Risk Register

| Risk | Likely reviewer concern | Mitigation |
|---|---|---|
| Synthetic-only | "This is toy data." | Include grid/cutout visuals and a verified real-image subset; emphasize diagnostic benchmark rather than robotics deployment |
| Rule reasoner too strong | "The method is just hand-coded." | Report learned OARL-VLA and ablations; use rule reasoner as oracle/upper bound |
| No real VLA baseline | "Not competitive with recent VLA papers." | Run OARL-VLA and direct VLA baselines under StarVLA/LIBERO, plus Qwen-VL/OpenVLA-style direct grounding baselines |
| Too broad task taxonomy | "Many tasks, shallow treatment." | Keep taxonomy but report systematic per-task analysis and failure modes |
| Weak web labels | "Noisy labels." | State weak labels are for pretraining/candidate mining only; final eval uses gold/verified labels |
| Page limit | "Unclear contributions." | Main paper claims only problem, benchmark, method, experiments; pipeline details in supplement |
