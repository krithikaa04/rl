import pandas as pd
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer

# Load a smaller dataset sample
data = pd.read_csv("C:/Users/krith/Downloads/summary.txt",sep = '\t')

sample_size = min(100, len(data))
data = data.sample(n=sample_size, random_state=42)
dataset = Dataset.from_pandas(data)

# Use smaller DistilBART model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Preprocess
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments (CPU optimized)
training_args = TrainingArguments(
    output_dir="./finetuned_bart_summary_cpu",
    #evaluation_strategy="no",  # skip eval
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # keep low for CPU
    num_train_epochs=3,  # 1 epoch for testing
    logging_strategy="no",
    save_strategy="no",
    report_to="none"  # no wandb or hub
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train (fast on CPU)
trainer.train()

# Save the model
model.save_pretrained("finetuned-bart-summary-cpu")
tokenizer.save_pretrained("finetuned-bart-summary-cpu")

from transformers import pipeline

def summarize(text, model_path="finetuned-bart-summary-cpu"):
    summarization_model = pipeline(
        "summarization",
        model=model_path,
        tokenizer=model_path
    )

    # Generate the summary
    summary = summarization_model(text)[0]['summary_text']
    return summary

# Example
text = '''Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.
          Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
          ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine.The application of ML to business problems is known as predictive analytics.
          Statistics and mathematical optimization (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning.
          From a theoretical viewpoint, probably approximately correct learning provides a framework for describing machine learning.'''

print(summarize(text))