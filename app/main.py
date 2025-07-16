from os import environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware import ExceptionHandler, TokenAuthentication, ResponseTime

from routers import esm3_model, health

from model import ESM3Model

from huggingface_hub import login as hf_login

import uvicorn


api_token = environ.get("API_TOKEN", "")
hf_token = environ.get("HF_TOKEN", "")
model_name = environ.get("MODEL_NAME", "esm3-open")
context_length = int(environ.get("CONTEXT_LENGTH", 2048))
device = environ.get("DEVICE", "cpu")

app = FastAPI(
    title="ESM3 Inference Server",
    description="ESM3 inference server for protein sequence prediction.",
    version="0.0.1",
)

# The ESM3 model requires a license agreement.
hf_login(token=hf_token)

model = ESM3Model(
    model_name=model_name,
    context_length=context_length,
    device=device,
)

app.state.model = model

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

app.include_router(esm3_model.router)
app.include_router(health.router)

app.add_middleware(ResponseTime)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
