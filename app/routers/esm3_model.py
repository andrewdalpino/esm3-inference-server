from typing import Literal

from pydantic import BaseModel, Field

from fastapi import APIRouter, Request

from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils.types import FunctionAnnotation


class GenerationRequest(BaseModel):
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

    function_annotations: list[str] | None = Field(
        description="InterPro function annotations for the sequence.",
        default=None,
    )

    track: Literal[
        "sequence", "structure", "function", "secondary_structure", "sasa"
    ] = Field(
        default="sequence",
        description="The track to use for generation.",
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


class GenerationResponse(BaseModel):
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

    function_annotations: list[str] | None = Field(
        description="InterPro function annotations for the sequence.",
        default=None,
    )

    pdb: str | None = Field(
        description="The structural coordinates of the sequence in pdf format.",
        default=None,
    )

    plddt: list[float] | None = Field(
        description="The predicted local distance difference test (pLDDT) scores for the sequence.",
        default=None,
    )

    ptm: float | None = Field(
        description="The predicted template modeling (PTM) scores for the sequence.",
        default=None,
    )


router = APIRouter(prefix="/model")


@router.post("/generate", response_model=GenerationResponse)
async def sequence_to_sequence(
    request: Request, input: GenerationRequest
) -> GenerationResponse:
    function_annotations = (
        [
            FunctionAnnotation(label=annotation)
            for annotation in input.function_annotations
        ]
        if input.function_annotations
        else None
    )

    protein = ESMProtein(
        sequence=input.sequence,
        secondary_structure=input.secondary_structure,
        sasa=input.sasa,
        function_annotations=function_annotations,
    )

    config = GenerationConfig(
        track=input.track,
        num_steps=input.num_steps,
        temperature=input.temperature,
        top_p=input.top_p,
    )

    protein = request.app.state.model.generate(protein, config)

    function_annotations = (
        [annotation.label for annotation in protein.function_annotations]
        if protein.function_annotations
        else None
    )

    pdb = protein.to_pdb_string() if protein.coordinates is not None else None

    plddt = protein.plddt.tolist() if protein.plddt is not None else None
    ptm = protein.ptm.item() if protein.ptm is not None else None

    return GenerationResponse(
        sequence=protein.sequence,
        secondary_structure=protein.secondary_structure,
        sasa=protein.sasa,
        function_annotations=function_annotations,
        pdb=pdb,
        plddt=plddt,
        ptm=ptm,
    )
