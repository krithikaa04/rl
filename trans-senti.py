


#pip install transformers datasets scikit-learn torch pandas
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


# Load and prepare dataset
df = pd.read_csv("C:/Users/krith/Downloads/sentimentdataset.csv")

# Data validation
df = df.dropna(subset=["Text", "Sentiment"])  # Remove NaN
df["Text"] = df["Text"].astype(str).replace("", None)  # Convert empty strings to None
df = df.dropna(subset=["Text"])  # Remove None in Text
df = df[df["Text"].str.strip() != ""]  # Remove whitespace-only Text

# Map sentiments to labels
label_map = {"Negative": 0, "Positive": 1, "Neutral": 2}
df["Sentiment"] = df["Sentiment"].str.strip().map(label_map)

# Check for unmapped labels
if df["Sentiment"].isna().any():
    print("Warning: Some Sentiment values could not be mapped. Dropping invalid rows.")
    print("Unmapped rows:", df[df["Sentiment"].isna()][["Text", "Sentiment"]])
    df = df.dropna(subset=["Sentiment"])

# Convert to Dataset
dataset = Dataset.from_pandas(df.reset_index(drop=True))

# Split into train and test
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Preprocessing
def preprocess(example):
    text = str(example["Text"]) if example["Text"] else ""
    if not text.strip():
        return {"input_ids": [], "attention_mask": [], "labels": example["Sentiment"]}
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors=None
    )
    encoding["labels"] = int(example["Sentiment"])
    return encoding

# Apply preprocessing
tokenized = dataset.map(preprocess, batched=False)

# Filter out invalid examples
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

# Minimal training arguments
training_args = TrainingArguments(
    output_dir="./tmp",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    #evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="no",
    report_to="none"
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Example usage: Sentiment prediction
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return {0: "Negative", 1: "Positive", 2: "Neutral"}[predicted_class]

# Test the prediction
example = "I hate this phone"
print(f"Predicted sentiment: {predict_sentiment(example)}")