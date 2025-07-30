import torch

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(
        self, model_name: str, quantize: bool, quant_group_size: int, device: str
    ):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        model = ESM3.from_pretrained(model_name, device=torch.device("cpu"))

        model = torch.compile(model)

        if quantize:
            quantize_(model, Int8WeightOnlyConfig(group_size=quant_group_size))

        model = model.to(device)

        model.eval()

        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(self, protein: ESMProtein, config: GenerationConfig) -> ESMProtein:
        return self.model.generate(protein, config)
