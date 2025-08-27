import torch

from torch.cuda import is_available as cuda_is_available, is_bf16_supported

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(self, name: str, quantize: bool, device: str):
        """
        Args:
            name (str): The name of the pretrained ESM3 model to load.
            quantize (bool): Whether to quantize the model weights to int8.
            device (str): The device to load the model on.
        """

        if name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if "cuda" in device and not cuda_is_available():
            raise ValueError("CUDA is not supported on this device.")

        model = ESM3.from_pretrained(name, device=torch.device("cpu"))

        # Preload the heads.
        model.get_structure_encoder()
        model.get_structure_decoder()
        model.get_function_decoder()

        model = torch.compile(model)

        if quantize:
            quantize_(model, Int8WeightOnlyConfig())

        model = model.to(device)

        model.eval()

        self.name = name
        self.model = model
        self.device = device

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""

        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def generate(self, protein: ESMProtein, config: GenerationConfig) -> ESMProtein:
        return self.model.generate(protein, config)
