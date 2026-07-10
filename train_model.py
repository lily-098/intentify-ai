import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# =====================================================================
# ---- EDIT THESE PATHS to match your uploaded Kaggle datasets ----
# =====================================================================
TRAIN_DATA_PATH = "/kaggle/input/hinglish-support/train_synthetic_augmented.csv"
VAL_DATA_PATH = "/kaggle/input/hinglish-support/val_real_world.csv"

OUTPUT_DIR = "/kaggle/working/model"
BASE_MODEL = "xlm-roberta-base"

# =====================================================================
# ---- OPTIMIZED HYPERPARAMETERS ----
# =====================================================================
NUM_TRAIN_EPOCHS = 8              
WEIGHT_DECAY = 0.01               
LEARNING_RATE = 1e-5              
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.05             
MAX_LENGTH = 96
SEED = 42

LABELS = ["complaint", "order_status", "refund_request", "product_query", "general"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

print("GPU available:", torch.cuda.is_available())

# =====================================================================
# ---- DATASET CLASS ----
# =====================================================================
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# =====================================================================
# ---- METRICS CALCULATION ----
# =====================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(LABELS))))
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
    for i, label in enumerate(LABELS):
        metrics[f"f1_{label}"] = per_class_f1[i]
    return metrics

# =====================================================================
# ---- LOAD DATA EXPLICITLY (No train_test_split) ----
# =====================================================================
print("\nLoading datasets...")
train_df = pd.read_csv(TRAIN_DATA_PATH).dropna(subset=["text", "label"])
val_df = pd.read_csv(VAL_DATA_PATH).dropna(subset=["text", "label"])

train_df["label_id"] = train_df["label"].map(LABEL2ID)
val_df["label_id"] = val_df["label"].map(LABEL2ID)

print("\n--- TRAIN Class Distribution (Synthetic Augmented) ---")
print(train_df["label"].value_counts())

print("\n--- VAL Class Distribution (Real-World Holdout) ---")
print(val_df["label"].value_counts())

print(f"\nTotal Train Size: {len(train_df)} | Total Val Size: {len(val_df)}\n")

# =====================================================================
# ---- INITIALIZE MODEL & TOKENIZER ----
# =====================================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
)

train_dataset = IntentDataset(train_df["text"], train_df["label_id"], tokenizer)
val_dataset = IntentDataset(val_df["text"], val_df["label_id"], tokenizer)

# =====================================================================
# ---- TRAINING ARGUMENTS ----
# =====================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",   
    greater_is_better=True,             
    label_smoothing_factor=LABEL_SMOOTHING,
    logging_steps=25,
    report_to=[],
    seed=SEED,
    bf16=torch.cuda.is_available(),
)

# =====================================================================
# ---- START TRAINING ----
# =====================================================================
trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training process...")
trainer.train()

print("\nEvaluating best model on Real-World Validation Set...")
metrics = trainer.evaluate()
print("Final validation metrics:", metrics)

# =====================================================================
# ---- SAVE ARTIFACTS ----
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

print(f"\nSuccess! Saved model, tokenizer, and label map to {OUTPUT_DIR}")
