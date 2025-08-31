from typing import Any

from pydantic import BaseModel, Field

from enum import Enum

from fastapi import APIRouter, Request

from esm.sdk.api import ESMProtein, GenerationConfig, ESMProteinError
from esm.utils.types import FunctionAnnotation


class ModelInfoResponse(BaseModel):
    name: str = Field(
        description="The name of the ESM3 model variant.",
    )

    num_parameters: int = Field(
        description="The number of parameters in the model.",
    )

    device: str = Field(
        description="The device the model is running on.",
    )

    quantize: bool = Field(
        description="Whether the model weights are quantized to Int8.",
    )

    max_concurrency: int = Field(
        description="The maximum number of concurrent generations.",
    )


class Schedule(str, Enum):
    """The sampling schedule to use when determining the number of tokens to unmask per step."""

    linear = "linear"
    cosine = "cosine"


class Strategy(str, Enum):
    """The sampling strategy to use when generating tokens."""

    random = "random"
    entropy = "entropy"


class GenerateRequest(BaseModel):
    """A generic generation request for the ESM3 model."""

    sequence: str | None = Field(
        description="An amino acid protein sequence.",
        default=None,
    )

    secondary_structure: str | None = Field(
        description="The secondary structure of the sequence.",
        default=None,
    )

    sasa: list[float | None] | None = Field(
        description="The solvent accessible surface area (SASA) of the sequence.",
        default=None,
    )

    function_annotations: list[dict[str, Any]] | None = Field(
        description="InterPro function annotations for the sequence.",
        default=None,
    )

    num_steps: int = Field(
        description="The number of sampling steps used to generate the sequence.",
        default=20,
        ge=1,
    )

    schedule: Schedule = Field(
        description="The sampling schedule used to determine the number of tokens to unmask per step.",
        default=Schedule.cosine,
    )

    strategy: Strategy = Field(
        description="The sampling strategy used when generating tokens.",
        default=Strategy.random,
    )

    temperature: float = Field(
        description="The temperature used for sampling.", default=1.0, gt=0.0
    )

    temperature_annealing: bool = Field(
        description="Should we anneal the temperature at each step?",
        default=True,
    )

    top_p: float = Field(
        description="The top probability value used for nucleus sampling.",
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    condition_on_coordinates_only: bool = Field(
        description="Should we only condition on the coordinates when generating the sequence?",
        default=True,
    )


class GenerateSequenceRequest(GenerateRequest):
    pass


class GenerateStructureRequest(GenerateRequest):
    pass


class GenerateFunctionAnnotationsRequest(GenerateRequest):
    pass


class GenerateSecondaryStructureRequest(GenerateRequest):
    pass


class GenerateSASARequest(GenerateRequest):
    pass


class GenerateSequenceResponse(BaseModel):
    sequence: str = Field(
        description="An amino acid protein sequence.",
    )


class GenerateStructureResponse(BaseModel):
    pdb: str = Field(
        description="The structural coordinates of the sequence in pdf format.",
    )

    plddt: list[float] | None = Field(
        description="The predicted local distance difference test (pLDDT) scores for the sequence.",
        default=None,
    )

    ptm: float | None = Field(
        description="The predicted template modeling (PTM) scores for the sequence.",
        default=None,
    )


class GenerateFunctionAnnotationsResponse(BaseModel):
    function_annotations: list[dict[str, Any]] = Field(
        description="InterPro function annotations for the sequence.",
    )


class GenerateSecondaryStructureResponse(BaseModel):
    secondary_structure: str = Field(
        description="The secondary structure of the sequence.",
    )


class GenerateSASAResponse(BaseModel):
    sasa: list[float | None] = Field(
        description="The solvent accessible surface area (SASA) of the sequence.",
    )


router = APIRouter(prefix="/model")


@router.get("/", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    model = request.app.state.model

    return ModelInfoResponse(
        name=model.name,
        num_parameters=model.num_parameters,
        device=model.device,
        quantize=model.quantize,
        max_concurrency=model.max_concurrency,
    )


@router.post("/generate/sequence", response_model=GenerateSequenceResponse)
def generate_sequence(
    request: Request, input: GenerateSequenceRequest
) -> GenerateSequenceResponse:
    function_annotations = (
        [
            FunctionAnnotation(
                label=annotation.label, start=annotation.start, end=annotation.end
            )
            for annotation in input.function_annotations
        ]
        if input.function_annotations is not None
        else None
    )

    protein = ESMProtein(
        sequence=input.sequence,
        secondary_structure=input.secondary_structure,
        sasa=input.sasa,
        function_annotations=function_annotations,
    )

    config = GenerationConfig(
        track="sequence",
        num_steps=input.num_steps,
        schedule=input.schedule,
        strategy=input.strategy,
        temperature=input.temperature,
        temperature_annealing=input.temperature_annealing,
        top_p=input.top_p,
        condition_on_coordinates_only=input.condition_on_coordinates_only,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSequenceResponse(
        sequence=protein.sequence,
    )


@router.post("/generate/structure", response_model=GenerateStructureResponse)
def generate_structure(
    request: Request, input: GenerateStructureRequest
) -> GenerateStructureResponse:
    function_annotations = (
        [
            FunctionAnnotation(
                label=annotation.label, start=annotation.start, end=annotation.end
            )
            for annotation in input.function_annotations
        ]
        if input.function_annotations is not None
        else None
    )

    protein = ESMProtein(
        sequence=input.sequence,
        secondary_structure=input.secondary_structure,
        sasa=input.sasa,
        function_annotations=function_annotations,
    )

    config = GenerationConfig(
        track="structure",
        num_steps=input.num_steps,
        schedule=input.schedule,
        strategy=input.strategy,
        temperature=input.temperature,
        temperature_annealing=input.temperature_annealing,
        top_p=input.top_p,
        condition_on_coordinates_only=input.condition_on_coordinates_only,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    pdb = protein.to_pdb_string() if protein.coordinates is not None else None

    plddt = protein.plddt.tolist() if protein.plddt is not None else None
    ptm = protein.ptm.item() if protein.ptm is not None else None

    return GenerateStructureResponse(
        pdb=pdb,
        plddt=plddt,
        ptm=ptm,
    )


@router.post(
    "/generate/function_annotations", response_model=GenerateFunctionAnnotationsResponse
)
def generate_function_annotations(
    request: Request, input: GenerateFunctionAnnotationsRequest
) -> GenerateFunctionAnnotationsResponse:
    function_annotations = (
        [
            FunctionAnnotation(
                label=annotation.label, start=annotation.start, end=annotation.end
            )
            for annotation in input.function_annotations
        ]
        if input.function_annotations is not None
        else None
    )

    protein = ESMProtein(
        sequence=input.sequence,
        secondary_structure=input.secondary_structure,
        sasa=input.sasa,
        function_annotations=function_annotations,
    )

    config = GenerationConfig(
        track="function",
        num_steps=input.num_steps,
        schedule=input.schedule,
        strategy=input.strategy,
        temperature=input.temperature,
        temperature_annealing=input.temperature_annealing,
        top_p=input.top_p,
        condition_on_coordinates_only=input.condition_on_coordinates_only,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    function_annotations = (
        [
            {
                "label": annotation.label,
                "start": annotation.start,
                "end": annotation.end,
            }
            for annotation in protein.function_annotations
        ]
        if protein.function_annotations is not None
        else []
    )

    return GenerateFunctionAnnotationsResponse(
        function_annotations=function_annotations,
    )


@router.post(
    "/generate/secondary_structure", response_model=GenerateSecondaryStructureResponse
)
def generate_secondary_structure(
    request: Request, input: GenerateSecondaryStructureRequest
) -> GenerateSecondaryStructureResponse:
    function_annotations = (
        [
            FunctionAnnotation(
                label=annotation.label, start=annotation.start, end=annotation.end
            )
            for annotation in input.function_annotations
        ]
        if input.function_annotations is not None
        else None
    )

    protein = ESMProtein(
        sequence=input.sequence,
        secondary_structure=input.secondary_structure,
        sasa=input.sasa,
        function_annotations=function_annotations,
    )

    config = GenerationConfig(
        track="secondary_structure",
        num_steps=input.num_steps,
        schedule=input.schedule,
        strategy=input.strategy,
        temperature=input.temperature,
        temperature_annealing=input.temperature_annealing,
        top_p=input.top_p,
        condition_on_coordinates_only=input.condition_on_coordinates_only,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSecondaryStructureResponse(
        secondary_structure=protein.secondary_structure,
    )


@router.post("/generate/sasa", response_model=GenerateSASAResponse)
def generate_secondary_structure(
    request: Request, input: GenerateSASARequest
) -> GenerateSASAResponse:
    function_annotations = (
        [
            FunctionAnnotation(
                label=annotation.label, start=annotation.start, end=annotation.end
            )
            for annotation in input.function_annotations
        ]
        if input.function_annotations is not None
        else None
    )

    protein = ESMProtein(
        sequence=input.sequence,
        secondary_structure=input.secondary_structure,
        sasa=input.sasa,
        function_annotations=function_annotations,
    )

    config = GenerationConfig(
        track="sasa",
        num_steps=input.num_steps,
        schedule=input.schedule,
        strategy=input.strategy,
        temperature=input.temperature,
        temperature_annealing=input.temperature_annealing,
        top_p=input.top_p,
        condition_on_coordinates_only=input.condition_on_coordinates_only,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSASAResponse(sasa=protein.sasa)
