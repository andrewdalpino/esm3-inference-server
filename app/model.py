import torch

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(self, name: str, quantize: bool, device: str):
        if name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        model = ESM3.from_pretrained(name, device=torch.device("cpu"))

        model = torch.compile(model)

        model = model.to(device)

        if quantize:
            quantize_(model, Int8WeightOnlyConfig())

        model.eval()

        self.name = name
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(self, protein: ESMProtein, config: GenerationConfig) -> ESMProtein:
        return self.model.generate(protein, config)
