from typing import Any, Self

from pydantic import BaseModel, Field

from enum import Enum

from fastapi import APIRouter, Request

from esm.sdk.api import ESMProtein, GenerationConfig, ESMProteinError, ProteinChain
from esm.utils.types import FunctionAnnotation

from io import StringIO


class Schedule(str, Enum):
    """The sampling schedule to use when determining the number of tokens to unmask per step."""

    linear = "linear"
    cosine = "cosine"


class Strategy(str, Enum):
    """The sampling strategy to use when generating tokens."""

    random = "random"
    entropy = "entropy"


class GenerateSettings(BaseModel):
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

    def to_config(self, track: str) -> GenerationConfig:
        return GenerationConfig(
            track=track,
            num_steps=self.num_steps,
            schedule=self.schedule,
            strategy=self.strategy,
            temperature=self.temperature,
            temperature_annealing=self.temperature_annealing,
            top_p=self.top_p,
            condition_on_coordinates_only=self.condition_on_coordinates_only,
        )


class GenerateRequest(GenerateSettings):
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

    def to_protein(self) -> ESMProtein:
        function_annotations = (
            [
                FunctionAnnotation(
                    label=annotation.label, start=annotation.start, end=annotation.end
                )
                for annotation in self.function_annotations
            ]
            if self.function_annotations is not None
            else None
        )

        return ESMProtein(
            sequence=self.sequence,
            secondary_structure=self.secondary_structure,
            sasa=self.sasa,
            function_annotations=function_annotations,
        )


class GenerateSequenceRequest(GenerateRequest):
    pass


class GenerateStructureRequest(GenerateRequest):
    pass


class GenerateSecondaryStructureRequest(GenerateRequest):
    pass


class GenerateSASARequest(GenerateRequest):
    pass


class GenerateFunctionAnnotationsRequest(GenerateRequest):
    pass


class GenerateSequenceResponse(BaseModel):
    sequence: str = Field(
        description="The generated amino acid protein sequence.",
    )

    @classmethod
    def from_protein(cls, protein: ESMProtein) -> Self:
        return cls(sequence=protein.sequence)


class GenerateStructureResponse(BaseModel):
    pdb: str = Field(
        description="The structural coordinates of the sequence in PDB format.",
    )

    plddt: list[float] = Field(
        description="The predicted local distance difference test (pLDDT) scores for the sequence.",
    )

    ptm: float = Field(
        description="The predicted template modeling (PTM) scores for the sequence.",
    )

    @classmethod
    def from_protein(cls, protein: ESMProtein) -> Self:
        return cls(
            pdb=protein.to_pdb_string(),
            plddt=protein.plddt.tolist(),
            ptm=protein.ptm.item(),
        )


class GenerateSecondaryStructureResponse(BaseModel):
    secondary_structure: str = Field(
        description="The secondary structure of the sequence.",
    )

    @classmethod
    def from_protein(cls, protein: ESMProtein) -> Self:
        return cls(secondary_structure=protein.secondary_structure)


class GenerateSASAResponse(BaseModel):
    sasa: list[float | None] = Field(
        description="The solvent accessible surface area (SASA) of the sequence.",
    )

    @classmethod
    def from_protein(cls, protein: ESMProtein) -> Self:
        return cls(sasa=protein.sasa)


class GenerateFunctionAnnotationsResponse(BaseModel):
    function_annotations: list[dict[str, Any]] = Field(
        description="InterPro function annotations for the sequence.",
    )

    @classmethod
    def from_protein(cls, protein: ESMProtein) -> Self:
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

        return cls(function_annotations=function_annotations)


class PDBGenerateRequest(GenerateSettings):
    pdb: str = Field(
        description="A PDB encoded string representing the structure of a protein.",
    )

    with_annotations: bool = Field(
        description="Should we include secondary structure and SASA annotations in the input?",
        default=False,
    )

    chain_id: str = Field(
        description="The ID of the protein chain.",
        default="detect",
    )

    sequence_id: str | None = Field(
        description="The ID of the protein sequence.",
        default=None,
    )

    is_predicted: bool = Field(
        description="Should we read the b factor as the confidence score?",
        default=False,
    )

    def to_protein(self) -> ESMProtein:
        buffer = StringIO(self.pdb)

        protein_chain = ProteinChain.from_pdb(
            path=buffer,
            chain_id=self.chain_id,
            id=self.sequence_id,
            is_predicted=self.is_predicted,
        )

        return ESMProtein.from_protein_chain(
            protein_chain, with_annotations=self.with_annotations
        )


class PDBGenerateFunctionAnnotationsRequest(PDBGenerateRequest):
    pass


router = APIRouter(prefix="/model")


@router.post("/generate/sequence", response_model=GenerateSequenceResponse)
def generate_sequence(
    request: Request, input: GenerateSequenceRequest
) -> GenerateSequenceResponse:
    protein = input.to_protein()

    config = input.to_config(track="sequence")

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSequenceResponse.from_protein(protein)


@router.post("/generate/structure", response_model=GenerateStructureResponse)
def generate_structure(
    request: Request, input: GenerateStructureRequest
) -> GenerateStructureResponse:
    protein = input.to_protein()

    config = input.to_config(track="structure")

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateStructureResponse.from_protein(protein)


@router.post(
    "/generate/function_annotations", response_model=GenerateFunctionAnnotationsResponse
)
def generate_function_annotations(
    request: Request, input: GenerateFunctionAnnotationsRequest
) -> GenerateFunctionAnnotationsResponse:
    protein = input.to_protein()

    config = input.to_config(track="function")

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateFunctionAnnotationsResponse.from_protein(protein)


@router.post(
    "/generate/secondary_structure", response_model=GenerateSecondaryStructureResponse
)
def generate_secondary_structure(
    request: Request, input: GenerateSecondaryStructureRequest
) -> GenerateSecondaryStructureResponse:
    protein = input.to_protein()

    config = input.to_config(track="secondary_structure")

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSecondaryStructureResponse.from_protein(protein)


@router.post("/generate/sasa", response_model=GenerateSASAResponse)
def generate_secondary_structure(
    request: Request, input: GenerateSASARequest
) -> GenerateSASAResponse:
    protein = input.to_protein()

    config = input.to_config(track="sasa")

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateSASAResponse.from_protein(protein)


@router.post(
    "/pdb/generate/function_annotations",
    response_model=GenerateFunctionAnnotationsResponse,
)
def pdb_generate_function_annotations(
    request: Request, input: PDBGenerateFunctionAnnotationsRequest
) -> GenerateFunctionAnnotationsResponse:
    protein = input.to_protein()

    config = input.to_config(track="function")

    protein = request.app.state.model.generate(protein, config)

    if isinstance(protein, ESMProteinError):
        raise ValueError(f"Error generating sequence: {protein.error_msg}")

    return GenerateFunctionAnnotationsResponse.from_protein(protein)
