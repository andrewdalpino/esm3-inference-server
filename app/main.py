from os import environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import esm3_model
from middleware import ExceptionHandler, TokenAuthentication, ResponseTime

from routers import health

from model import ESM3Model

from huggingface_hub import login as hf_login

import uvicorn


hf_token = environ.get("HF_TOKEN", "")
api_token = environ.get("API_TOKEN", "")
model_name = environ.get("MODEL_NAME", "esm3-open")
quantize = environ.get("QUANTIZE", "false").lower() == "true"
device = environ.get("DEVICE", "cpu")

app = FastAPI(
    title="ESM3 Inference Server",
    description="ESM3 evolutionary protein modelling inference server.",
    version="0.0.9",
)

# The ESM3 model requires a license agreement.
hf_login(token=hf_token)

model = ESM3Model(model_name, quantize, device)

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

app.add_middleware(ResponseTime)

app.include_router(esm3_model.router)
app.include_router(health.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
