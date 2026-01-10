import os
import glob
import json
import sys
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Optional

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CONFIGURATION
MODEL_DIR = "/app/models"
DATA_DIR = "/app/data"
INSTRUCTIONS_FILE = os.path.join(DATA_DIR, "instructions.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# GLOBAL STATE
current_llm = None
current_model_name = "None"

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    instructions: List[str]

class LoadModelRequest(BaseModel):
    url: str

class SelectModelRequest(BaseModel):
    filename: str

class SaveInstructionsRequest(BaseModel):
    instructions: List[dict]

# --- Helpers ---
def load_stored_instructions():
    if os.path.exists(INSTRUCTIONS_FILE):
        try:
            with open(INSTRUCTIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    # Defaults if file doesn't exist
    return [
        {"text": "Be concise", "active": False},
        {"text": "Act like a pirate", "active": False},
        {"text": "Use JSON format", "active": False}
    ]

def save_stored_instructions(inst_list):
    with open(INSTRUCTIONS_FILE, 'w') as f:
        json.dump(inst_list, f, indent=2)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    models = []
    # Recursively walk through MODEL_DIR to find files in subdirectories
    for root, dirs, files in os.walk(MODEL_DIR):
        for f in files:
            # Skip hidden files like .DS_Store or partial downloads starting with '.'
            if f.startswith('.'):
                continue
            
            # Get the path relative to MODEL_DIR (e.g., 'subfolder/model.gguf')
            rel_path = os.path.relpath(os.path.join(root, f), MODEL_DIR)
            models.append(rel_path)
            
    models.sort()
    
    stored_inst = load_stored_instructions()
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "models": models,
        "current_model": current_model_name,
        "saved_instructions": stored_inst
    })

@app.post("/save_instructions")
async def save_instructions_endpoint(payload: SaveInstructionsRequest):
    try:
        save_stored_instructions(payload.instructions)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_url")
async def load_model_from_url(payload: LoadModelRequest):
    global current_llm, current_model_name
    
    url = payload.url
    
    # Clean filename from URL arguments
    filename = url.split("/")[-1].split("?")[0]
    
    if "." not in filename:
        filename += ".gguf"
        
    filepath = os.path.join(MODEL_DIR, filename)

    # Download if not exists
    if not os.path.exists(filepath):
        try:
            print(f"Starting download: {filename}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            wrote = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        wrote += len(chunk)
                        
                        if total_size > 0:
                            percent = (wrote / total_size) * 100
                            sys.stdout.write(f"\rDownloading... {percent:.2f}% ({wrote//1024//1024}MB / {total_size//1024//1024}MB)")
                        else:
                            sys.stdout.write(f"\rDownloading... {wrote//1024//1024}MB")
                        sys.stdout.flush()
            
            print("\nDownload complete.")
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
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

    system_prompt = "System: You are a helpful assistant."
    if payload.instructions:
        full_inst = " ".join(payload.instructions).replace("\n", " ")
        system_prompt += f" {full_inst}"
    
    prompt_text = f"{system_prompt}\n"
    for msg in payload.history:
        role = "User" if msg['role'] == 'user' else "Assistant"
        prompt_text += f"{role}: {msg['content']}\n"
    
    prompt_text += f"User: {payload.message}\nAssistant:"

    try:
        output = current_llm(
            prompt_text, 
            max_tokens=512, 
            stop=["User:", "\nUser"], 
            echo=False
        )
        response_text = output['choices'][0]['text'].strip()
        return {"response": response_text}
    except Exception as e:
        return {"response": f"Error during inference: {str(e)}"}
