import torch

from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

from threading import Semaphore


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(
        self,
        name: str,
        device: str,
        quantize: bool,
        max_concurrency: int,
    ):
        """
        Args:
            name (str): The name of the pretrained ESM3 model to load.
            quantize (bool): Whether to quantize the model weights to int8.
            device (str): The device to load the model on.
            max_concurrency (int): The maximum number of concurrent generations.
        """

        if name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if "cuda" in device and not cuda_is_available():
            raise ValueError("CUDA is not supported on this device.")

        if "mps" in device and not mps_is_available():
            raise ValueError("MPS is not supported on this device.")

        if max_concurrency < 1:
            raise ValueError("Max concurrency must be at least 1.")

        model = ESM3.from_pretrained(name, device=torch.device("cpu"))

        # Preload additional encoder/decoder networks.
        model.get_structure_encoder()
        model.get_structure_decoder()
        model.get_function_decoder()

        model = torch.compile(model)

        if quantize:
            quantize_(model, Int8WeightOnlyConfig())

        model = model.to(device)

        model.eval()

        limiter = Semaphore(max_concurrency)

        self.name = name
        self.model = model
        self.device = device
        self.limiter = limiter

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""

        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def generate(self, protein: ESMProtein, config: GenerationConfig) -> ESMProtein:
        """Generate tokens for the given protein and configuration."""

        with self.limiter:
            return self.model.generate(protein, config)
