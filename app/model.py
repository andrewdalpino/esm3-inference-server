import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(self, model_name: str, context_length: int, device: str):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if context_length <= 0:
            raise ValueError("Context length must be greater than 0.")

        model = ESM3.from_pretrained(model_name, device=torch.device(device))

        model.eval()

        self.model = model
        self.context_length = context_length
        self.device = device

    @torch.no_grad()
    def sequence_to_sequence(
        self, sequence: str, num_steps: int, temperature: float
    ) -> ESMProtein:
        if len(sequence) > self.context_length:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds context length {self.context_length}."
            )

        protein = ESMProtein(sequence=sequence)

        config = GenerationConfig(
            track="sequence", num_steps=num_steps, temperature=temperature
        )

        protein = self.model.generate(protein, config)

        return protein

    @torch.no_grad()
    def sequence_to_structure(self, sequence: str, num_steps: int) -> ESMProtein:
        if len(sequence) > self.context_length:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds context length {self.context_length}."
            )

        protein = ESMProtein(sequence=sequence)

        config = GenerationConfig(track="structure", num_steps=num_steps)

        protein = self.model.generate(protein, config)

        return protein
