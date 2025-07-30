from os import environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware import ExceptionHandler, TokenAuthentication, ResponseTime

from routers import esm3_model, health

from model import ESM3Model

from huggingface_hub import login as hf_login

import uvicorn


hf_token = environ.get("HF_TOKEN", "")
api_token = environ.get("API_TOKEN", "")
model_name = environ.get("MODEL_NAME", "esm3-open")
quantize = environ.get("QUANTIZE", "true").lower() == "true"
quant_group_size = int(environ.get("QUANT_GROUP_SIZE", 16))
device = environ.get("DEVICE", "cpu")

app = FastAPI(
    title="ESM3 Inference Server",
    description="ESM3 inference server for protein sequence generation.",
    version="0.0.8",
)

# The ESM3 model requires a license agreement.
hf_login(token=hf_token)

model = ESM3Model(model_name, quantize, quant_group_size, device)

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
