import os
import glob
import requests
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Optional

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CONFIGURATION
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# GLOBAL STATE
current_llm = None
current_model_name = "None"

class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    instructions: List[str]

class LoadModelRequest(BaseModel):
    url: str

class SelectModelRequest(BaseModel):
    filename: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # List models in volume
    models = [os.path.basename(x) for x in glob.glob(f"{MODEL_DIR}/*.gguf")]
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "models": models,
        "current_model": current_model_name
    })

@app.post("/load_url")
async def load_model_from_url(payload: LoadModelRequest):
    global current_llm, current_model_name
    
    url = payload.url
    filename = url.split("/")[-1]
    filepath = os.path.join(MODEL_DIR, filename)

    # Download if not exists
    if not os.path.exists(filepath):
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    # Load Model
    try:
        current_llm = Llama(model_path=filepath, n_ctx=2048, verbose=False)
        current_model_name = filename
        return {"status": "loaded", "model": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load GGUF: {str(e)}")

@app.post("/select_model")
async def select_model(payload: SelectModelRequest):
    global current_llm, current_model_name
    filepath = os.path.join(MODEL_DIR, payload.filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        current_llm = Llama(model_path=filepath, n_ctx=2048, verbose=False)
        current_model_name = payload.filename
        return {"status": "loaded", "model": payload.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load: {str(e)}")

@app.post("/chat")
async def chat(payload: ChatRequest):
    if not current_llm:
        return {"response": "Error: No model loaded. Please load a model first."}

    # Construct Prompt
    system_prompt = "System: You are a helpful assistant."
    if payload.instructions:
        system_prompt += " " + " ".join(payload.instructions)
    
    prompt_text = f"{system_prompt}\n"
    for msg in payload.history:
        role = "User" if msg['role'] == 'user' else "Assistant"
        prompt_text += f"{role}: {msg['content']}\n"
    
    prompt_text += f"User: {payload.message}\nAssistant:"

    # Inference
    output = current_llm(
        prompt_text, 
        max_tokens=256, 
        stop=["User:", "\nUser"], 
        echo=False
    )
    
    response_text = output['choices'][0]['text'].strip()
    return {"response": response_text}
