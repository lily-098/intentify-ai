# Intentify AI 🚀

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface)

**Intentify AI** is a production-grade NLP application that classifies highly informal, code-mixed **Hinglish** (Hindi + English) customer support queries into 5 distinct intents. It features a fine-tuned `xlm-roberta-base` transformer model, an adversarial synthetic dataset generation pipeline, and a modern Dark Mode Chat UI powered by FastAPI.

## 🌟 Key Features
- **Hinglish Understanding:** Handles extreme slang, typos, and abbreviations (e.g., *"mera order trckng ni hori"*, *"kya kachra bhej dia"*).
- **Adversarial Hard-Negative Mining:** Model is trained on synthetic edge cases generated via Groq (LLaMA-3) to prevent shortcut learning (e.g., differentiating between *"otp for delivery"* vs *"otp for password reset"*).
- **Confidence Thresholding:** Automatically routes out-of-domain (OOD) queries to a human agent if prediction confidence is `< 60%`.
- **Premium Full-Stack UI:** A beautiful, responsive chat interface built with glassmorphism effects and micro-animations.

## 🧠 Supported Intents
1. `complaint`: General dissatisfaction, bad quality, or rude customer service.
2. `order_status`: Missing deliveries, tracking issues, or delivery boy contact.
3. `refund_request`: Soft and hard refund requests ("paisa wapas", "return karna hai").
4. `product_query`: Pre-purchase questions regarding warranty, COD, or sizing.
5. `general`: App crashes, login bugs, or account detail changes.

## 🚀 How to Run Locally

Because the fine-tuned XLM-RoBERTa model weights are **1.1 GB**, they are not included in this repository. You must train/download the model first.

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
Open a terminal in the project directory and install the requirements:
```bash
pip install -r requirements.txt
```
Start the local server:
```bash
uvicorn app:app --reload
```
### 3. Test the App
Open your web browser and go to **http://localhost:8000** to interact with the Chatbot UI!

## 🛠️ Generating New Synthetic Data (Optional)
If you want to generate more Hinglish training data:
1. Create a `.env` file in the root directory.
2. Add your Groq API key: `GROQ_API_KEY="your_api_key_here"`.
3. Run `python generate_dataset.py`.
