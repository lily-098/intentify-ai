from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from sqlalchemy.orm import Session
from sqlalchemy import text
import models
from database import engine, get_db

app = FastAPI(title="Hinglish Intent Classifier")

API_KEY = os.getenv("API_KEY", "intentify-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API KEY")
    return api_key_header

# Path to the HuggingFace model repository
MODEL_PATH = "Lily-28/intentify-minilm"  

# Global variables for model and tokenizer
model = None
tokenizer = None
id2label = {}

class QueryRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    log_id: int
    feedback: int

@app.on_event("startup")
def load_model():
    models.Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE query_logs ADD COLUMN feedback INTEGER DEFAULT 0"))
    except Exception:
        pass
    global model, tokenizer, id2label
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval() # Set to evaluation mode
        
        # Load the label mapping
        label_map_path = "./label_map.json"
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                labels = json.load(f)
                id2label = {int(k): v for k, v in labels["id2label"].items()}
        else:
            print("Warning: label_map.json not found locally. Predictions will return raw IDs.")
            
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

# Mount the static directory to serve HTML, CSS, JS
# We mount it at /static, but we will serve index.html directly from the root
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read the index.html file from the static folder
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI not found. Make sure static/index.html exists!</h1>"

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    dashboard_path = os.path.join("static", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard UI not found. Make sure static/dashboard.html exists!</h1>"

@app.get("/api/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(models.QueryLog).order_by(models.QueryLog.timestamp.desc()).limit(100).all()
    return logs

@app.post("/predict")
def predict_intent(request: QueryRequest, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    if not model or not tokenizer:
        return {"error": "Model failed to load on startup."}
        
    # Tokenize the input text
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=96)
    
    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # Apply softmax to get confidence scores
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get the top prediction
    top_prob, top_idx = torch.max(probabilities, dim=0)
    
    # Map to label if available, otherwise use ID string
    if id2label:
        intent = id2label.get(top_idx.item(), str(top_idx.item()))
    else:
        intent = str(top_idx.item())
    
    confidence = top_prob.item()
    
    # Confidence thresholding (Fallback for out-of-domain)
    if confidence < 0.60:
        intent = "fallback_to_human"

    # Map the snake_case intents to readable UI intents
    readable_intent = intent.replace("_", " ").title() if intent != "fallback_to_human" else "Human Agent Handoff"

    # Save to Database
    db_log = models.QueryLog(
        text=request.text,
        predicted_intent=intent,
        confidence=confidence
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    return {
        "text": request.text,
        "predicted_intent": intent,
        "readable_intent": readable_intent,
        "confidence": round(confidence, 4),
        "log_id": db_log.id
    }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    log = db.query(models.QueryLog).filter(models.QueryLog.id == request.log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    log.feedback = request.feedback
    db.commit()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    # This allows you to run it via `python app.py` instead of uvicorn command if preferred
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
