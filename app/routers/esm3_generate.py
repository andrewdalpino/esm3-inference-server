from typing import Any

from pydantic import BaseModel, Field

from fastapi import APIRouter, Request

from esm.sdk.api import ESMProtein, GenerationConfig, ESMProteinError
from esm.utils.types import FunctionAnnotation


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
        default=20,
        ge=1,
        description="The number of sampling steps used to generate the sequence.",
    )

    temperature: float = Field(
        default=1.0, gt=0.0, description="The temperature used for sampling."
    )

    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="The top p value used for nucleus sampling.",
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


@router.post("/generate/sequence", response_model=GenerateSequenceResponse)
async def generate_sequence(
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
        temperature=input.temperature,
        top_p=input.top_p,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSequenceResponse(
        sequence=protein.sequence,
    )


@router.post("/generate/structure", response_model=GenerateStructureResponse)
async def generate_structure(
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
        temperature=input.temperature,
        top_p=input.top_p,
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
async def generate_function_annotations(
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
        temperature=input.temperature,
        top_p=input.top_p,
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
async def generate_secondary_structure(
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
        temperature=input.temperature,
        top_p=input.top_p,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSecondaryStructureResponse(
        secondary_structure=protein.secondary_structure,
    )


@router.post("/generate/sasa", response_model=GenerateSASAResponse)
async def generate_secondary_structure(
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
        temperature=input.temperature,
        top_p=input.top_p,
    )

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    plddt = protein.plddt.tolist() if protein.plddt is not None else None
    ptm = protein.ptm.item() if protein.ptm is not None else None

    return GenerateSASAResponse(
        sasa=protein.sasa,
        plddt=plddt,
        ptm=ptm,
    )
