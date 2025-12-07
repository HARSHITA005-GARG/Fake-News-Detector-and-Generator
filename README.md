📰 Fake News Generator & Detector
GPT-2 Fake News Generator + DistilBERT Fake News Classifier
FastAPI Backend + Streamlit Frontend + Full Deployment Guide
📌 Overview

This project is an end-to-end Machine Learning system that can:

✔ Generate realistic fake news headlines (GPT-2)
✔ Detect whether a news headline is FAKE or REAL (DistilBERT)
✔ Provide a full API backend using FastAPI
✔ Provide a frontend UI using Streamlit
✔ Run locally or be deployed online (Render + Streamlit Cloud)

It is designed for MLOps workflow demonstration, college projects, research work, or real-world prototyping.

🚀 Features
🔷 Fake News Generator (GPT-2)

Fine-tuned GPT-2 model

Generates realistic & diverse fake news headlines

Uses top-k, top-p sampling & temperature control

🔶 Fake News Detector (DistilBERT)

Binary classifier (REAL = 0, FAKE = 1)

Achieves 99%+ accuracy, precision, recall, F1-score

Trained on FakeNewsNet style headline dataset

🔷 Full Pipeline

Generates multiple headlines → Detects each headline → Returns prediction + confidence score.

🔶 Web API (FastAPI)

Endpoints:

Endpoint	Description
/generate	Generate news headlines using GPT-2
/detect	Predict FAKE or REAL
/pipeline	Generate & classify in a single request
🔷 Streamlit Frontend

Simple UI for:

Entering prompts

Generating headlines

Checking fake/real

Running full pipeline

📁 Project Structure
fake-news-project/
│
├── main.py                 # FastAPI backend
├── streamlit_app.py        # Streamlit frontend UI
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── .gitignore              # Ignore system + model files
│
├── models/
│   ├── fakenews_model/     # DistilBERT classifier
│   └── gpt2_news/          # GPT-2 generator
│
└── render.yaml (optional for deployment)

🧠 Model Details
Generator: GPT-2 (headline generation)

Fine-tuned on cleaned fake-news style dataset

Produces coherent & deceptive headlines

Detector: DistilBERT (binary classification)

Label mapping:

0 → REAL

1 → FAKE

Performance:

Accuracy:   0.99+
Precision:  0.99+
Recall:     0.99+
F1-score:   0.99+

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/fake-news-project.git
cd fake-news-project

2️⃣ Install Dependencies
pip install -r requirements.txt


The project supports CPU-only PyTorch, making it suitable for laptops.

▶️ Run FastAPI Backend
python -m uvicorn main:app --reload


Open API docs:

http://127.0.0.1:8000/docs

▶️ Run Streamlit Frontend
streamlit run streamlit_app.py


Streamlit launches at:

http://localhost:8501

🌐 Deployment Guide

This project supports simple FREE deployment.

⭐ 1. Deploy FastAPI Backend to Render (Free)

Push project to GitHub

Add render.yaml:

services:
  - type: web
    name: fake-news-backend
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port 10000"
    plan: free


Visit
https://render.com

Create a New Web Service → link your repo → deploy

Render gives a public API URL like:

https://fake-news-backend.onrender.com

⭐ 2. Deploy Streamlit Frontend to Streamlit Cloud

Visit
https://share.streamlit.io

Create a new app → choose your repo

Set main file → streamlit_app.py

Update API URL inside your Streamlit file:

FASTAPI_URL = "https://fake-news-backend.onrender.com"


Your app becomes publicly accessible. ✔

🧪 Testing the Pipeline

Example request:

POST /pipeline
{
  "prompt": "Breaking News: A leaked government report reveals",
  "num": 3
}


Response:

{
  "results": [
    {
      "headline": "...",
      "prediction": "FAKE",
      "confidence": 0.82
    }
  ]
}

🛠️ Technologies Used

Python

PyTorch

HuggingFace Transformers

FastAPI

Uvicorn

Streamlit

Render

Streamlit Cloud

🤝 Contributing

Pull requests are welcome!
Feel free to add more datasets, improve UI, or integrate Docker.

📄 License

MIT License — free for academic and commercial use.

⭐ Acknowledgements

FakeNewsNet Dataset

HuggingFace Transformers

Streamlit

FastAPI

Kaggle for training environment


