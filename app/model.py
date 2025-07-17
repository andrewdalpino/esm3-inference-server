import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(self, model_name: str, device: str):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        model = ESM3.from_pretrained(model_name, device=torch.device(device))

        model.eval()

        self.model = model
        self.device = device

    @torch.no_grad()
    def sequence_to_sequence(
        self, sequence: str, num_steps: int, temperature: float
    ) -> ESMProtein:
        if len(sequence) == 0:
            raise ValueError("Sequence must not be empty.")

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        protein = ESMProtein(sequence=sequence)

        config = GenerationConfig(
            track="sequence", num_steps=num_steps, temperature=temperature
        )

        protein = self.model.generate(protein, config)

        return protein

    @torch.no_grad()
    def sequence_to_structure(self, sequence: str, num_steps: int) -> ESMProtein:
        if len(sequence) == 0:
            raise ValueError("Sequence must not be empty.")

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        protein = ESMProtein(sequence=sequence)

        config = GenerationConfig(track="structure", num_steps=num_steps)

        protein = self.model.generate(protein, config)

        return protein
