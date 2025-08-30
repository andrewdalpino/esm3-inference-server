from types import MethodType

import torch

from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
from torch.nn import Module

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.models.function_decoder import FunctionTokenDecoder

from asyncio import Semaphore


class ESM3Model:
    AVAILABLE_MODELS = {"esm3-open"}

    def __init__(
        self,
        name: str,
        quantize: bool,
        device: str,
        max_concurrency: int,
        cpu_offloading: bool,
    ):
        """
        Args:
            name (str): The name of the pretrained ESM3 model to load.
            quantize (bool): Whether to quantize the model weights to int8.
            device (str): The device to load the model on.
            max_concurrency (int): The maximum number of concurrent generations.
            cpu_offloading (bool): Whether to offload parts of the model to CPU when not in use.
        """

        if name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if "cpu" in device and cpu_offloading:
            raise ValueError("CPU offloading cannot be used with CPU device.")

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

        if cpu_offloading:
            # Replace encoder/decoder getters to support CPU offloading.
            model.get_structure_encoder = MethodType(self.get_structure_encoder, model)
            model.get_structure_decoder = MethodType(self.get_structure_decoder, model)
            model.get_function_decoder = MethodType(self.get_function_decoder, model)

            def pin_module(module: Module):
                for parameter in module.parameters():
                    parameter.pin_memory()

            # Pin additional networks for faster memory swaps.
            pin_module(model._structure_encoder)
            pin_module(model._structure_decoder)
            pin_module(model._function_decoder)

            model.encoder = model.encoder.to(device)
            model.transformer = model.transformer.to(device)
            model.output_heads = model.output_heads.to(device)

        else:
            model = model.to(device)

        limiter = Semaphore(max_concurrency)

        model.eval()

        self.name = name
        self.model = model
        self.device = device
        self.limiter = limiter

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""

        return sum(p.numel() for p in self.model.parameters())

    def get_structure_encoder(self, model: ESM3) -> StructureTokenEncoder:
        """Return the encoder section of the structure variational autoencoder."""

        model._structure_encoder = model._structure_encoder.to(
            self.device, non_blocking=True
        )
        model._structure_decoder = model._structure_decoder.to("cpu", non_blocking=True)
        model._function_decoder = model._function_decoder.to("cpu", non_blocking=True)

        return model._structure_encoder

    def get_structure_decoder(self, model: ESM3) -> StructureTokenDecoder:
        """Return the decoder section of the structure variational autoencoder."""

        model._structure_encoder = model._structure_encoder.to("cpu", non_blocking=True)
        model._structure_decoder = model._structure_decoder.to(
            self.device, non_blocking=True
        )
        model._function_decoder = model._function_decoder.to("cpu", non_blocking=True)

        return model._structure_decoder

    def get_function_decoder(self, model: ESM3) -> FunctionTokenDecoder:
        """Return the function annotation token decoder."""

        model._structure_encoder = model._structure_encoder.to("cpu", non_blocking=True)
        model._structure_decoder = model._structure_decoder.to("cpu", non_blocking=True)
        model._function_decoder = model._function_decoder.to(
            self.device, non_blocking=True
        )

        return model._function_decoder

    @torch.inference_mode()
    def generate(self, protein: ESMProtein, config: GenerationConfig) -> ESMProtein:
        """Generate tokens for the given protein and configuration."""

        with self.limiter:
            return self.model.generate(protein, config)
