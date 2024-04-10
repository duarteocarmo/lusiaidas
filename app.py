from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger
from llama_cpp import Llama
from pydantic import BaseModel
import pathlib
import json
from threading import Lock

REPO_ID = "duarteocarmo/lusiaidas-v0.1-q4_k_m"
CACHE_TIME_SECONDS = 7200
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
LLM = Llama.from_pretrained(repo_id=REPO_ID, filename="*.gguf", verbose=True)
task_lock = Lock()


class Prompt(BaseModel):
    prompt: str


app = FastAPI()


def generate_completion(prompt: str, model: Llama):
    task_lock.acquire()
    try:
        logger.debug("Task lock acquired")
        completion = ""
        for item in model(**GENERATION_CONFIG, prompt=PROMPT_PREFIX.format(prompt)):
            choice = item["choices"][0]  # type: ignore
            completion += choice["text"]  # type: ignore
            yield "data: {}\n\n".format(json.dumps(choice))
    finally:
        task_lock.release()
        logger.debug("Task lock released")

    logger.debug("Completion generated")
    logger.debug(prompt + completion)


@app.post("/completion")
def completion(prompt: Prompt):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    return StreamingResponse(generate_completion(prompt.prompt, model=LLM))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        content=pathlib.Path("assets/index.html").read_text(), status_code=200
    )
