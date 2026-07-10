import pandas as pd

CSV_PATH = "c:/Users/hp/Desktop/NLP_change/train_synthetic_augmented.csv"

# 20 highly targeted "Silver Bullet" examples to fix the 8 specific confusion errors
silver_bullet_data = [
    # Fix: "app me" & "otp" belonging to order_status (not general)
    {"text": "delivery boy ka number nahi dikh raha app me", "label": "order_status"},
    {"text": "app me tracking update nahi ho rahi kal se", "label": "order_status"},
    {"text": "otp nahi aa raha delivery accept karne ke liye", "label": "order_status"},
    {"text": "delivery confirm karne ka otp mobile par nahi aaya", "label": "order_status"},
    {"text": "courier partner ki website down hai status check nahi ho raha", "label": "order_status"},
    {"text": "app me order out for delivery dikha raha hai par aaya nahi", "label": "order_status"},
    {"text": "delivery otp send kardo number par", "label": "order_status"},

    # Fix: "kachra" & "color manga" belonging to complaint
    {"text": "bhai yr kya kachra bhej dia ekdum ghatiya", "label": "complaint"},
    {"text": "jo color manga tha wo nahi bheja dusra de diya", "label": "complaint"},
    {"text": "bakwas quality hai pura kachra nikla andar se", "label": "complaint"},
    {"text": "wrong size bhej diya jo order kiya tha wo nahi hai", "label": "complaint"},
    {"text": "customer care wale phone kaat dete hai", "label": "complaint"},

    # Fix: "return policy kya hai" & soft refunds
    {"text": "is item ka return policy kya hai", "label": "refund_request"},
    {"text": "agar size fit na ho to return policy kaisi hai", "label": "refund_request"},
    {"text": "paisa wapas chahiye iska return process batao", "label": "refund_request"},
    {"text": "kya mai 7 days me return karke refund le sakta hu", "label": "refund_request"},

    # Fix: "cod h kya" & abbreviations like "trckng"
    {"text": "cod h kya isme", "label": "product_query"},
    {"text": "cash on delivery available hai is pin code par", "label": "product_query"},
    {"text": "mera order trckng ni hori", "label": "order_status"},
    {"text": "trckng id invalid aa rahi hai check karke batao", "label": "order_status"}
]

# Load existing, append, and save
df = pd.read_csv(CSV_PATH)
df_silver = pd.DataFrame(silver_bullet_data)

# Combine and shuffle
df_combined = pd.concat([df, df_silver], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

df_combined.to_csv(CSV_PATH, index=False)
print(f"Successfully added {len(silver_bullet_data)} silver bullet rows! Total rows: {len(df_combined)}")
