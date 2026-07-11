# Intentify AI 🚀

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Intentify AI** is a production-grade NLP application that classifies highly informal, code-mixed **Hinglish** (Hindi + English) customer support queries into 5 distinct intents. It features a fine-tuned `microsoft/Multilingual-MiniLM-L12-H384` transformer model, an adversarial synthetic dataset generation pipeline, and a modern Dark Mode Chat UI powered by FastAPI.

## 🎯 Problem Statement
Customer support systems in South Asia struggle to parse queries because users rarely type in pure English or pure Hindi. Instead, they use **Hinglish**—a code-mixed language heavily reliant on Latin script, abbreviations, and informal slang (e.g., *"mera refund kb aayega", "app kaam ni karra"*). Standard NLP models fail on this unstructured data. **Intentify AI** solves this by fine-tuning a multilingual model on synthetically generated, adversarial edge-cases to accurately route customer intents, reducing manual support overhead.

## 💻 Tech Stack
- **Machine Learning:** PyTorch, Transformers (HuggingFace), Scikit-Learn
- **Backend:** Python, FastAPI, SQLAlchemy, SQLite
- **Frontend:** HTML5, Vanilla CSS (Glassmorphism), JavaScript, Chart.js
- **DevOps:** Docker, Uvicorn
- **Data Generation:** Groq API (LLaMA-3)

## 🔄 System Workflow
1. **User Input:** User submits a Hinglish query via the SPA Frontend.
2. **API Authentication:** Request is sent to FastAPI backend using a secure `X-API-Key`.
3. **Inference:** The query is tokenized and passed through the fine-tuned `Multilingual-MiniLM` model.
4. **Logging:** The predicted intent and confidence score are instantly logged to the SQLite database.
5. **UI Rendering:** The result is rendered with dynamic intent-colored micro-animations. If confidence is `< 60%`, it triggers a Human Agent Handoff.
6. **RLHF Loop:** The user provides thumbs up/down feedback on the UI, which updates the database log, creating a curated dataset for future model fine-tuning.

## 🌟 Key Features
- **Hinglish Understanding:** Handles extreme slang, typos, and abbreviations (e.g., *"mera order trckng ni hori"*, *"kya kachra bhej dia"*).
- **Adversarial Hard-Negative Mining:** Model is trained on synthetic edge cases generated via Groq (LLaMA-3) to prevent shortcut learning.
- **Premium SPA UI:** A beautiful, responsive Single Page Application (SPA) built with glassmorphism effects. Features a built-in **Analytics Dashboard** using Chart.js to visualize live model predictions.
- **RLHF (Reinforcement Learning from Human Feedback):** Includes a thumbs up/down feedback loop in the chat UI, dynamically logging feedback to a SQLite database for future model fine-tuning.
- **Production-Ready Architecture:** Containerized via **Docker**, backed by **SQLAlchemy/SQLite**, and secured with FastAPI **API Key Authentication**.

## 🧠 Supported Intents
1. `complaint`: General dissatisfaction, bad quality, or rude customer service.
2. `order_status`: Missing deliveries, tracking issues, or delivery boy contact.
3. `refund_request`: Soft and hard refund requests ("paisa wapas", "return karna hai").
4. `product_query`: Pre-purchase questions regarding warranty, COD, or sizing.
5. `general`: App crashes, login bugs, or account detail changes.

## 🚀 How to Run Locally

Because the fine-tuned MiniLM model weights are **~470 MB**, they are not included in this repository. You must train/download the model first.

### 1. Download Model Weights
1. Run the `train_model.py` script on Kaggle or Google Colab (requires T4 GPU).
2. Download the resulting `model/` folder.
3. Place the following files directly in the root directory of this project:
   - `model.safetensors`
   - `config.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `label_map.json`

### 2. Start the FastAPI Server

**Option A: Using Docker (Recommended)**
```bash
docker-compose up --build
```

**Option B: Using Local Python Environment**
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

### 3. Test the App
Open your web browser and go to **http://localhost:8000** to interact with the Chatbot UI! You can toggle between the chat and the Analytics Dashboard using the top header.

## 🛠️ Generating New Synthetic Data (Optional)
If you want to generate more Hinglish training data:
1. Create a `.env` file in the root directory.
2. Add your Groq API key: `GROQ_API_KEY="your_api_key_here"`.
3. Run `python generate_dataset.py`.
