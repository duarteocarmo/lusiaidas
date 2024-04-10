# Description: This file contains the FastAPI application that serves the model.
# It loads the model from the Hugging Face Hub and provides an endpoint to generate completions.
# It also contains a timer to unload the model after a period of inactivity, because I have a small server.
# Additionally, there is a lock to prevent multiple requests from generating completions at the same time.
# This code is hacked together and not production-ready. It is only meant to be a proof of concept.
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger
from llama_cpp import Llama
import pathlib
import json
from threading import Lock, Timer
from pydantic import BaseModel

REPO_ID = "duarteocarmo/lusiaidas-v0.1-q4_k_m"
CACHE_TIME_SECONDS = 60 * 60 # 1 hour
GENERATION_CONFIG = {
    "max_tokens": 400,
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 0.95,
    "repeat_penalty": 1.2,
    "stream": True,
}
PROMPT_PREFIX = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Escreve uma estrofe ao estilo de Os Lusíadas

### Response:
{}"""


task_lock = Lock()
model_timer = None
LLM = None


app = FastAPI()


def load_model():
    global LLM
    if LLM is None:
        LLM = Llama.from_pretrained(repo_id=REPO_ID, filename="*.gguf", verbose=True)
        logger.info("Model loaded")


def unload_model():
    global LLM
    if LLM is not None:
        LLM = None
        logger.info("Model unloaded due to inactivity")


def reset_timer():
    global model_timer
    if model_timer:
        model_timer.cancel()
    model_timer = Timer(CACHE_TIME_SECONDS, unload_model)
    model_timer.start()
    logger.info("Timer reset!")


def generate_completion(prompt: str):
    reset_timer()
    load_model()

    task_lock.acquire()
    try:
        completion = ""
        assert LLM is not None, "Model not loaded"
        for item in LLM(**GENERATION_CONFIG, prompt=PROMPT_PREFIX.format(prompt)):
            choice = item["choices"][0]  # type: ignore
            completion += choice["text"]  # type: ignore
            yield "data: {}\n\n".format(json.dumps(choice))

        logger.debug("Completion generated")
    finally:
        task_lock.release()


class Prompt(BaseModel):
    prompt: str


@app.post("/completion")
def completion(prompt: Prompt):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    return StreamingResponse(generate_completion(prompt.prompt))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        content=pathlib.Path("assets/index.html").read_text(), status_code=200
    )


@app.on_event("startup")
async def startup_event():
    load_model()
    reset_timer()


@app.on_event("shutdown")
async def shutdown_event():
    global model_timer
    if model_timer:
        model_timer.cancel()
