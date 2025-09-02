from os import environ

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware import ExceptionHandler, TokenAuthentication, ResponseTime

from routers import info, generate, health

from model import ESM3Model

from huggingface_hub import login as hf_login

import uvicorn


TITLE = "ESM3 Inference Server"

DESCRIPTION = "ESM3 evolutionary protein modelling inference server."

VERSION = "0.0.12"

hf_token = environ.get("HF_TOKEN", "")
api_token = environ.get("API_TOKEN", "")
model_name = environ.get("MODEL_NAME", "esm3-open")
device = environ.get("DEVICE", "cpu")
quantize = environ.get("QUANTIZE", "false").lower() == "true"
max_concurrency = int(environ.get("MAX_CONCURRENCY", "1"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The ESM3 model requires a license agreement.
    hf_login(token=hf_token)

    model = ESM3Model(model_name, device, quantize, max_concurrency)

    app.state.model = model

    yield


app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(ExceptionHandler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if api_token:
    app.add_middleware(TokenAuthentication, api_token=api_token)

app.add_middleware(ResponseTime)

app.include_router(info.router)
app.include_router(generate.router)
app.include_router(health.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
