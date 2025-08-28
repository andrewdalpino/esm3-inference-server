from types import MethodType

import torch

from torch.cuda import is_available as cuda_is_available

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(self, name: str, quantize: bool, device: str, cpu_offloading: bool):
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

        # Preload additional encoder/decoder networks.
        structure_encoder = model.get_structure_encoder()
        structure_decoder = model.get_structure_decoder()
        function_decoder = model.get_function_decoder()

        model = torch.compile(model)

        if quantize:
            quantize_(model, Int8WeightOnlyConfig())

        if cpu_offloading:
            model.encoder = model.encoder.to(device)
            model.transformer = model.transformer.to(device)
            model.output_heads = model.output_heads.to(device)
        else:
            model = model.to(device)

        model.get_structure_encoder = MethodType(self.get_structure_encoder, model)
        model.get_structure_decoder = MethodType(self.get_structure_decoder, model)
        model.get_function_decoder = MethodType(self.get_function_decoder, model)

        model.eval()

        self.name = name
        self.model = model
        self.device = device

        self.structure_encoder = structure_encoder
        self.structure_decoder = structure_decoder
        self.function_decoder = function_decoder

        self.cpu_offloading = cpu_offloading

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""

        return sum(p.numel() for p in self.model.parameters())

    def get_structure_encoder(self, esm3):
        if self.cpu_offloading and "cpu" not in self.device:
            self.structure_decoder = self.structure_decoder.to("cpu")
            self.function_decoder = self.function_decoder.to("cpu")

            self.structure_encoder = self.structure_encoder.to(self.device)

        return self.structure_encoder

    def get_structure_decoder(self, esm3):
        if self.cpu_offloading and "cpu" not in self.device:
            self.structure_encoder = self.structure_encoder.to("cpu")
            self.function_decoder = self.function_decoder.to("cpu")

            self.structure_decoder = self.structure_decoder.to(self.device)

        return self.structure_decoder

    def get_function_decoder(self, esm3):
        if self.cpu_offloading and "cpu" not in self.device:
            self.structure_encoder = self.structure_encoder.to("cpu")
            self.structure_decoder = self.structure_decoder.to("cpu")

            self.function_decoder = self.function_decoder.to(self.device)

        return self.function_decoder

    @torch.no_grad()
    def generate(self, protein: ESMProtein, config: GenerationConfig) -> ESMProtein:
        return self.model.generate(protein, config)
