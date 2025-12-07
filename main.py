from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

app = FastAPI(
    title="Fake News Generator & Detector API",
    description="Generate Fake News using GPT-2 and Detect Fake News using DistilBERT",
    version="1.0"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Fake News Detector
# -----------------------------
detector_path = "models/fakenews_model"
detector_model = AutoModelForSequenceClassification.from_pretrained(detector_path).to(device)
detector_tokenizer = AutoTokenizer.from_pretrained(detector_path)

# -----------------------------
# Load Fake News Generator (GPT-2)
# -----------------------------
generator_path = "models/gpt2_fakenews"
generator_model = AutoModelForCausalLM.from_pretrained(generator_path).to(device)
generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)

# -----------------------------
# Request Body Schemas
# -----------------------------
class TextInput(BaseModel):
    text: str

class GenerateInput(BaseModel):
    prompt: str = "Breaking News:"
    num: int = 3

# -----------------------------
# Fake News Detection Endpoint
# -----------------------------
@app.post("/detect")
def detect_fake_news(data: TextInput):
    inputs = detector_tokenizer(data.text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = detector_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
    label = int(probs.argmax())

    return {
        "text": data.text,
        "prediction": "FAKE" if label == 1 else "REAL",
        "confidence": float(probs[label])
    }

# -----------------------------
# Fake News Generation Endpoint
# -----------------------------
@app.post("/generate")
def generate_news(data: GenerateInput):
    inputs = generator_tokenizer(data.prompt, return_tensors="pt").to(device)

    outputs = generator_model.generate(
        **inputs,
        max_length=40,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.9,
        num_return_sequences=data.num,
        pad_token_id=generator_tokenizer.eos_token_id
    )

    generated = [
        generator_tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]

    return {"generated_news": generated}

# -----------------------------
# Full Pipeline Endpoint
# -----------------------------
@app.post("/pipeline")
def generate_and_detect(data: GenerateInput):
    # Generate fake news
    inputs = generator_tokenizer(data.prompt, return_tensors="pt").to(device)
    outputs = generator_model.generate(
        **inputs,
        max_length=40,
        do_sample=True,
        num_return_sequences=data.num,
        top_k=50,
        top_p=0.92,
        temperature=0.85,
        pad_token_id=generator_tokenizer.eos_token_id
    )

    headlines = [generator_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # Detect each
    results = []
    for h in headlines:
        inputs = detector_tokenizer(h, return_tensors="pt", truncation=True, padding=True).to(device)
        out = detector_model(**inputs)
        probs = torch.softmax(out.logits, dim=1)[0].detach().cpu().numpy()
        label = int(probs.argmax())

        results.append({
            "headline": h,
            "prediction": "FAKE" if label == 1 else "REAL",
            "confidence": float(probs[label])
        })

    return {"results": results}
