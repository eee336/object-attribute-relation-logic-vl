"""StarVLA framework variant for OARL-VLA-QwenPI.

This file follows StarVLA's MIT-licensed framework API and QwenPI data flow,
with OARL-VLA's target-grounded reasoning core inserted before the action head.

Copy this file into:

    starVLA/model/framework/VLM4A/OARLVLAQwenPI.py

inside a StarVLA checkout. The implementation keeps StarVLA's QwenPI
flow-matching action infrastructure, but inserts the OARL-VLA reasoning core
and target-grounding bottleneck before the action head.

The OARL-VLA repository must be importable in the StarVLA environment:

    export PYTHONPATH=/path/to/object-attribute-relation-logic-vla/src:$PYTHONPATH
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

from oarlvla.models.oarl_core import OARLReasoningCore


logger = initialize_overwatch(__name__)


@dataclass
class OARLVLAQwenPIDefaultConfig:
    """Default StarVLA config for the OARL-VLA-QwenPI framework variant."""

    name: str = "OARLVLAQwenPI"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 36,
        }
    )

    oarl: dict = field(
        default_factory=lambda: {
            "object_feature_dim": 35,
            "region_feature_dim": 0,
            "num_relation_types": 8,
            "dropout": 0.1,
            "use_relation_graph": True,
            "target_loss_weight": 0.2,
            # When LIBERO examples do not yet contain object proposals, use one
            # zero-valued scene token. This keeps the StarVLA training loop alive
            # while simulator/detector-derived object tokens are being added.
            "fallback_single_scene_token": True,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("OARLVLAQwenPI")
@FRAMEWORK_REGISTRY.register("OARL-VLA-QwenPI")
class OARLVLA_QwenPI(baseframework):
    """Full OARL-VLA model running inside StarVLA's trainer/evaluator.

    Components:
      - Qwen-VL backbone for global visual-language tokens.
      - OARLReasoningCore for object-centric reasoning and target grounding.
      - StarVLA Layerwise Flow-Matching action head conditioned on Qwen tokens
        plus OARL target/action-context tokens.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(OARLVLAQwenPIDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(getattr(text_cfg, "num_hidden_layers"))
        llm_hidden_size = int(getattr(text_cfg, "hidden_size", None) or getattr(vlm_hf_cfg, "hidden_size"))
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.oarl_core = OARLReasoningCore(
            object_feature_dim=int(self.config.framework.oarl.object_feature_dim),
            hidden_dim=llm_hidden_size,
            num_relation_types=int(self.config.framework.oarl.num_relation_types),
            dropout=float(self.config.framework.oarl.dropout),
            use_relation_graph=bool(self.config.framework.oarl.use_relation_graph),
            region_feature_dim=int(self.config.framework.oarl.region_feature_dim),
        )

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_vl_layers,
        )
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> tuple:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list, attention_mask

    def _append_oarl_context(self, vl_embs_list, attention_mask, examples):
        base_hidden = vl_embs_list[-1]
        pooled_text = self._masked_pool(base_hidden, attention_mask)
        oarl_inputs = self._build_oarl_inputs(examples, base_hidden)
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if base_hidden.is_cuda else nullcontext()
        with autocast_ctx:
            core_outputs = self.oarl_core(
                text_embedding=pooled_text,
                object_features=oarl_inputs["object_features"],
                relation_features={
                    "edge_index": oarl_inputs["relation_edges"],
                    "edge_type": oarl_inputs["relation_types"],
                },
                object_region_features=oarl_inputs["object_region_features"],
                object_mask=oarl_inputs["object_mask"],
                relation_mask=oarl_inputs["relation_mask"],
                use_relation_graph=bool(self.config.framework.oarl.use_relation_graph),
            )
        oarl_tokens = core_outputs["action_context_tokens"].to(dtype=base_hidden.dtype)
        oarl_mask = core_outputs["action_context_mask"].to(device=base_hidden.device, dtype=torch.bool)
        augmented_hidden = [torch.cat([hidden, oarl_tokens], dim=1) for hidden in vl_embs_list]
        if attention_mask is None:
            base_mask = torch.ones(
                base_hidden.shape[:2],
                device=base_hidden.device,
                dtype=torch.bool,
            )
        else:
            base_mask = attention_mask.to(device=base_hidden.device, dtype=torch.bool)
        augmented_mask = torch.cat([base_mask, oarl_mask], dim=1)
        return augmented_hidden, augmented_mask, core_outputs, oarl_inputs

    @staticmethod
    def _masked_pool(hidden, attention_mask):
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _build_oarl_inputs(self, examples, base_hidden):
        object_feature_dim = int(self.config.framework.oarl.object_feature_dim)
        region_feature_dim = int(self.config.framework.oarl.region_feature_dim)
        fallback_ok = bool(self.config.framework.oarl.fallback_single_scene_token)
        device = base_hidden.device
        dtype = base_hidden.dtype

        object_rows = []
        region_rows = []
        target_indices = []
        max_objects = 0
        max_edges = 0
        raw_edges = []
        raw_edge_types = []
        for example in examples:
            objects = example.get("oarl_object_features", example.get("object_features"))
            if objects is None:
                if not fallback_ok:
                    raise KeyError(
                        "OARL-VLA requires oarl_object_features/object_features. "
                        "Set framework.oarl.fallback_single_scene_token=true for plumbing-only runs."
                    )
                objects = np.zeros((1, object_feature_dim), dtype=np.float32)
            objects = np.asarray(objects, dtype=np.float32)
            if objects.ndim == 1:
                objects = objects[None, :]
            if objects.shape[-1] < object_feature_dim:
                objects = np.pad(objects, ((0, 0), (0, object_feature_dim - objects.shape[-1])))
            elif objects.shape[-1] > object_feature_dim:
                objects = objects[:, :object_feature_dim]
            object_rows.append(objects)
            max_objects = max(max_objects, objects.shape[0])

            regions = example.get("oarl_object_region_features", example.get("object_region_features"))
            if regions is not None:
                regions = np.asarray(regions, dtype=np.float32)
                if regions.ndim == 1:
                    regions = regions[None, :]
                region_rows.append(regions)
            else:
                region_rows.append(None)

            edges = np.asarray(example.get("oarl_relation_edges", []), dtype=np.int64).reshape(-1, 2)
            edge_types = np.asarray(example.get("oarl_relation_types", []), dtype=np.int64).reshape(-1)
            raw_edges.append(edges)
            raw_edge_types.append(edge_types)
            max_edges = max(max_edges, edges.shape[0])

            target_indices.append(int(example.get("oarl_target_index", example.get("target_index", -1))))

        object_features = torch.zeros(len(examples), max_objects, object_feature_dim, device=device, dtype=dtype)
        object_mask = torch.zeros(len(examples), max_objects, device=device, dtype=torch.bool)
        relation_edges = torch.zeros(len(examples), max_edges, 2, device=device, dtype=torch.long)
        relation_types = torch.zeros(len(examples), max_edges, device=device, dtype=torch.long)
        relation_mask = torch.zeros(len(examples), max_edges, device=device, dtype=torch.bool)
        object_region_features = None
        if region_feature_dim > 0 or any(row is not None for row in region_rows):
            final_region_dim = region_feature_dim or max(row.shape[-1] for row in region_rows if row is not None)
            object_region_features = torch.zeros(len(examples), max_objects, final_region_dim, device=device, dtype=dtype)

        for idx, objects in enumerate(object_rows):
            n = objects.shape[0]
            object_features[idx, :n] = torch.as_tensor(objects, device=device, dtype=dtype)
            object_mask[idx, :n] = True
            if object_region_features is not None and region_rows[idx] is not None:
                regions = region_rows[idx]
                dim = min(regions.shape[-1], object_region_features.shape[-1])
                object_region_features[idx, : regions.shape[0], :dim] = torch.as_tensor(
                    regions[:, :dim],
                    device=device,
                    dtype=dtype,
                )
            e = raw_edges[idx].shape[0]
            if e:
                relation_edges[idx, :e] = torch.as_tensor(raw_edges[idx], device=device, dtype=torch.long)
                relation_types[idx, :e] = torch.as_tensor(raw_edge_types[idx][:e], device=device, dtype=torch.long)
                relation_mask[idx, :e] = True

        return {
            "object_features": object_features,
            "object_region_features": object_region_features,
            "object_mask": object_mask,
            "relation_edges": relation_edges,
            "relation_types": relation_types,
            "relation_mask": relation_mask,
            "target_index": torch.as_tensor(target_indices, device=device, dtype=torch.long),
        }

    def _target_loss(self, core_outputs, oarl_inputs):
        target_index = oarl_inputs["target_index"]
        valid = target_index >= 0
        if not valid.any():
            return None
        return F.cross_entropy(core_outputs["target_logits"][valid].float(), target_index[valid])

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        vl_embs_list, backbone_attention_mask, core_outputs, oarl_inputs = self._append_oarl_context(
            vl_embs_list,
            backbone_attention_mask,
            examples,
        )
        base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            attention_repeated = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(dtype=torch.bool)

            state_repeated = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state_t.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=attention_repeated,
            )

        out = {"action_loss": action_loss}
        target_loss = self._target_loss(core_outputs, oarl_inputs)
        if target_loss is not None:
            out["target_loss"] = target_loss * float(self.config.framework.oarl.target_loss_weight)
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        vl_embs_list, backbone_attention_mask, _, _ = self._append_oarl_context(
            vl_embs_list,
            backbone_attention_mask,
            examples,
        )
        base_hidden = vl_embs_list[-1]
        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state_t,
                encoder_attention_mask=backbone_attention_mask.to(dtype=torch.bool),
            )
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/oarlvla_qwenpi_libero.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    model = OARLVLA_QwenPI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "Pick up the target object.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        output = model([sample, sample])
        print(f"Action Loss: {output['action_loss'].item()}")
    else:
        print("Constructed OARL-VLA-QwenPI; CUDA is required for Qwen-VL smoke forward.")
