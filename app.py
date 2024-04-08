from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel
import pathlib
import json

REPO_ID = "duarteocarmo/lusiaidas-v0.1-q4_k_m"
GENERATION_CONFIG = {
    "max_tokens": 400,
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 0.95,
    "repeat_penalty": 1.2,
    "stream": True,
}
LLM = Llama.from_pretrained(repo_id=REPO_ID, filename="*.gguf", verbose=True)
PROMPT_PREFIX = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Escreve uma estrofe ao estilo de Os Lusíadas

### Response:
{}"""


class Prompt(BaseModel):
    prompt: str


app = FastAPI()


def generate_completion(prompt: str):
    print("Generating...")
    for item in LLM(**GENERATION_CONFIG, prompt=PROMPT_PREFIX.format(prompt)):
        choice = item["choices"][0]  # type: ignore
        yield "data: {}\n\n".format(json.dumps(choice))
    print("Done")


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
