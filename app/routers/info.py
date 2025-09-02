from pydantic import BaseModel, Field

from fastapi import APIRouter, Request


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
