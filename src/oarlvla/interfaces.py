from __future__ import annotations


class ObjectDetector:
    def detect(self, image):
        raise NotImplementedError


class VLMReasoner:
    def predict_target(self, image, objects, instruction):
        raise NotImplementedError


class ProgramGenerator:
    def generate_program(self, image, objects, instruction):
        raise NotImplementedError


class VLAActionPolicy:
    def predict_action(self, image, objects, instruction, target):
        raise NotImplementedError


class PreferenceDataBuilder:
    def build_pairs(self, scene, instruction, correct_target, wrong_targets):
        raise NotImplementedError


class RLTrainer:
    def train(self, policy, reward_model, dataset):
        raise NotImplementedError

