import os
import pandas as pd
import requests
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
BASE_URL = "https://api.groq.com/openai/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Fetch available active models dynamically so we don't hit "decommissioned" errors
print("Fetching active Groq models...")
models_req = requests.get(f"{BASE_URL}/models", headers=HEADERS)
active_models = [m['id'] for m in models_req.json().get('data', [])]
# Prefer llama-3.3, then llama-3.1
model_id = None
for prefer in ["llama-3.3-70b", "llama-3.1-8b", "llama3", "mixtral"]:
    for m in active_models:
        if prefer in m.lower():
            model_id = m
            break
    if model_id: break

if not model_id:
    model_id = active_models[0] # fallback to any active model
print(f"Using active model: {model_id}")

PROMPTS = {
    "complaint": "Generate 20 unique Hinglish (Hindi written in English alphabet) customer support messages for the intent 'complaint'. Do NOT use angry words like bekar, kharab, gussa, or ghatiya. Make them purely factual descriptions of what went wrong (e.g. missing items, broken seals, ignored by support, wrong size). Output only the sentences, one per line.",
    
    "order_status": "Generate 20 unique Hinglish customer support messages for the intent 'order_status'. Make sure at least half of them use phrases like 'kaam nahi kar raha', 'show nahi ho raha', or 'invalid' (e.g. regarding tracking links or delivery apps not updating). Output only the sentences, one per line.",
    
    "refund_request": "Generate 20 unique Hinglish customer support messages for the intent 'refund_request'. Make them 'soft intents' where the user says things like 'wapas karna hai', 'return dalna hai', 'fit nahi hua' but doesn't explicitly mention the word 'paisa' or 'money'. Output only the sentences, one per line.",
    
    "product_query": "Generate 20 unique Hinglish customer support messages for the intent 'product_query'. Ask pre-purchase questions about restocks, sizing, COD availability, warranty, or materials. Use natural abbreviations like 'cod', 'wrnty', etc. Output only the sentences, one per line.",
    
    "general": "Generate 20 unique Hinglish customer support messages for the intent 'general'. These should be about app crashes, login failures, OTP not coming, or changing account details. Use heavy abbreviations and typos like 'app chal ni ra', 'otp nai aya'. Output only the sentences, one per line."
}

def generate_synthetic_data(num_batches_per_class=10):
    new_data = []
    
    for intent, prompt in PROMPTS.items():
        print(f"\nGenerating data for: {intent}")
        for batch in range(num_batches_per_class):
            print(f"  Batch {batch+1}/{num_batches_per_class}...")
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant generating synthetic data for a customer support bot."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.8
            }
            try:
                response = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    sentences = data["choices"][0]["message"]["content"].strip().split('\n')
                    for s in sentences:
                        clean_s = s.strip('- 1234567890.*"\'').strip()
                        if clean_s and len(clean_s) > 5:
                            new_data.append({"text": clean_s, "label": intent})
                else:
                    print(f"  Groq API Error: {response.status_code}, {response.text}")
                time.sleep(2.5)
            except Exception as e:
                print(f"  Request failed: {e}")
                time.sleep(5)
                
    return pd.DataFrame(new_data)

if __name__ == "__main__":
    print("Starting dataset generation...")
    # Generating 10 batches per class * 20 rows * 5 classes = ~1000 rows
    df_synthetic = generate_synthetic_data(num_batches_per_class=10)
    
    print("\nGeneration complete. Merging with original dataset...")
    # Load original dataset
    original_path = "final_dataset.csv"
    if os.path.exists(original_path):
        df_original = pd.read_csv(original_path)
        print(f"Loaded {len(df_original)} original rows.")
        
        # Merge datasets
        df_combined = pd.concat([df_original, df_synthetic], ignore_index=True)
        # Shuffle
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output_path = "train_synthetic_augmented.csv"
        df_combined.to_csv(output_path, index=False)
        print(f"SUCCESS! Saved final combined dataset ({len(df_combined)} rows) to {output_path}")
    else:
        print("Could not find final_dataset.csv, saving just the new rows.")
        df_synthetic.to_csv("train_synthetic_augmented.csv", index=False)
