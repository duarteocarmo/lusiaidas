from fastapi import FastAPI, HTTPException
from fastapi.background import BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger
from llama_cpp import Llama
from pydantic import BaseModel
import time
import typing as t
import pathlib
import json

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


class Prompt(BaseModel):
    prompt: str


class LLMCache:
    def __init__(self):
        self.llm: t.Optional[Llama] = None

    def load_model(self) -> Llama:
        logger.info("Loading model")
        if not self.llm:
            self.llm = Llama.from_pretrained(
                repo_id=REPO_ID, filename="*.gguf", verbose=True
            )
        return self.llm

    def unload_model(self):
        logger.info("Sleeping for 60 seconds")
        time.sleep(CACHE_TIME_SECONDS)
        if self.llm:
            self.llm = None
        logger.info("Model unloaded")


app = FastAPI()
llm_cache = LLMCache()


def generate_completion(prompt: str, model: Llama):
    completion = ""
    for item in model(**GENERATION_CONFIG, prompt=PROMPT_PREFIX.format(prompt)):
        choice = item["choices"][0]  # type: ignore
        completion += choice["text"]  # type: ignore
        yield "data: {}\n\n".format(json.dumps(choice))

    logger.debug("Completion generated")
    logger.debug(prompt + completion)


@app.post("/completion")
def completion(prompt: Prompt, background_tasks: BackgroundTasks):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    model = llm_cache.load_model()
    background_tasks.add_task(llm_cache.unload_model)

    return StreamingResponse(generate_completion(prompt.prompt, model=model))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        content=pathlib.Path("assets/index.html").read_text(), status_code=200
    )
