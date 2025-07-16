from pydantic import BaseModel, Field

from fastapi import APIRouter, Request


class SequenceToSequenceRequest(BaseModel):
    sequence: str = Field(
        min_length=1, description="A masked amino acid sequence prompt."
    )

    num_steps: int = Field(
        default=8,
        ge=1,
        description="The number of sampling steps used to generate the sequence.",
    )

    temperature: float = Field(
        default=1.0, gt=0.0, description="The temperature used for sampling."
    )


class SequenceToSequenceResponse(BaseModel):
    sequence: str = Field(description="The completed amino acid sequence.")


class SequenceToStructureRequest(BaseModel):
    sequence: str = Field(
        min_length=1, description="An amino acid sequence to predict."
    )

    num_steps: int = Field(
        default=8,
        ge=1,
        description="The number of sampling steps used to generate structure.",
    )


class SequenceToStructureResponse(BaseModel):
    pdb: str = Field(description="The predicted structure in pdb format.")


router = APIRouter(prefix="/model")


@router.post("/sequence-to-sequence", response_model=SequenceToSequenceResponse)
async def sequence_to_sequence(
    request: Request, input: SequenceToSequenceRequest
) -> SequenceToSequenceResponse:
    protein = request.app.state.model.sequence_to_sequence(
        input.sequence, input.num_steps, input.temperature
    )

    return SequenceToSequenceResponse(sequence=protein.sequence)

@router.post("/sequence-to-structure", response_model=SequenceToStructureResponse)
async def sequence_to_structure(
    request: Request, input: SequenceToStructureRequest
) -> SequenceToStructureResponse:
    protein = request.app.state.model.sequence_to_structure(
        input.sequence, input.num_steps
    )

    return SequenceToStructureResponse(pdb=protein.to_pdb_string())
