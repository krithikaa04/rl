# -*- coding: utf-8 -*-
"""nlp_lab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14_41wLkw7OZ0j31v6cDtPo5M6VBSTnjR

# **CYK and PCYK**

# CYK and PCYK - Deeksha
"""

from collections import defaultdict
from nltk import Tree
import nltk

# Ensure nltk resources are available
nltk.download('punkt')

# ---------------------------
# CYK PARSER (Non-Probabilistic)
# ---------------------------
def cyk_parser(grammar, tokens):
    n = len(tokens)
    table = [[set() for _ in range(n)] for _ in range(n)]
    back = [[defaultdict(list) for _ in range(n)] for _ in range(n)]

    # Inverse grammar mapping: RHS -> LHS
    rhs_to_lhs = defaultdict(set)
    for lhs, rules in grammar.items():
        for rhs in rules:
            rhs_to_lhs[tuple(rhs)].add(lhs)

    # Fill diagonals
    for i, token in enumerate(tokens):
        for lhs in rhs_to_lhs.get((token,), []):
            table[i][i].add(lhs)

    # Fill upper triangle
    for l in range(2, n+1):
        for i in range(n - l + 1):
            j = i + l - 1
            for k in range(i, j):
                for B in table[i][k]:
                    for C in table[k+1][j]:
                        for A in rhs_to_lhs.get((B, C), []):
                            table[i][j].add(A)
                            back[i][j][A].append((k, B, C))

    # Recursive tree building
    def build_tree(i, j, symbol):
        if i == j:
            return (symbol, tokens[i])
        for k, B, C in back[i][j].get(symbol, []):
            left = build_tree(i, k, B)
            right = build_tree(k+1, j, C)
            return (symbol, left, right)

    if 'S' in table[0][n-1]:
        return build_tree(0, n-1, 'S')
    else:
        return None

# ---------------------------
# EXAMPLE USAGE
# ---------------------------
tokens = ['the', 'cat', 'chased', 'the', 'mouse']

# CYK grammar
grammar = {
    'S': [['NP', 'VP']],
    'VP': [['V', 'NP']],
    'NP': [['Det', 'N']],
    'Det': [['the']],
    'N': [['cat'], ['mouse']],
    'V': [['chased']]
}

# Run CYK
cyk_tree = cyk_parser(grammar, tokens)
if cyk_tree:
    print("CYK Parse Tree (tuple):")
    print(cyk_tree)
    nltk_tree = tuple_to_nltk_tree(cyk_tree)
    nltk_tree.pretty_print()
else:
    print("No parse tree found.Invalid input sentence")

"""# CYK and PCYK - KV"""

from collections import defaultdict
import math

from nltk import Tree

# --- Extract rules from trees ---
def extract_rules_from_tree(tree):
    rules = []
    if len(tree) == 2 and isinstance(tree[1], str):  # Terminal
        lhs, word = tree
        rules.append((lhs, (word,)))
    elif len(tree) == 3:
        lhs, left, right = tree
        rules.append((lhs, (left[0], right[0])))
        rules += extract_rules_from_tree(left)
        rules += extract_rules_from_tree(right)
    return rules

# --- Learn rule probabilities ---
def compute_rule_probabilities(trees):
    rule_counts = defaultdict(int)
    lhs_counts = defaultdict(int)

    for tree in trees:
        rules = extract_rules_from_tree(tree)
        for lhs, rhs in rules:
            rule_counts[(lhs, rhs)] += 1
            lhs_counts[lhs] += 1

    rule_probs = {
        (lhs, rhs): count / lhs_counts[lhs]
        for (lhs, rhs), count in rule_counts.items()
    }

    return rule_probs

# --- Probabilistic CYK Parser ---
def probabilistic_cyk_parser(rule_probs, sentence):
    n = len(sentence)
    table = [[defaultdict(lambda: (-math.inf, None)) for _ in range(n)] for _ in range(n)]

    # Fill in terminals
    for i in range(n):
        word = sentence[i]
        for (lhs, rhs), prob in rule_probs.items():
            if rhs == (word,):
                table[i][i][lhs] = (math.log(prob), (word,))

    # Fill in non-terminals
    for span in range(2, n+1):  # span length
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for (lhs, rhs), prob in rule_probs.items():
                    if len(rhs) == 2:
                        B, C = rhs
                        prob_B, back_B = table[i][k].get(B, (-math.inf, None))
                        prob_C, back_C = table[k+1][j].get(C, (-math.inf, None))
                        if prob_B > -math.inf and prob_C > -math.inf:
                            total_prob = math.log(prob) + prob_B + prob_C
                            if total_prob > table[i][j][lhs][0]:
                                table[i][j][lhs] = (total_prob, (B, C, i, k, k+1, j))

    # Check if S can generate the sentence
    if 'S' not in table[0][n-1]:
        return None, 0.0

    # Build the best tree
    def build_tree(sym, i, j):
        prob, back = table[i][j][sym]
        if len(back) == 1:
            return (sym, back[0])  # Terminal
        else:
            B, C, i1, k1, k2, j2 = back
            return (sym, build_tree(B, i1, k1), build_tree(C, k2, j2))

    best_tree = build_tree('S', 0, n-1)
    best_log_prob = table[0][n-1]['S'][0]
    return best_tree, math.exp(best_log_prob)

# --- Example usage ---
training_trees = [
    ('S', ('NP', 'she'), ('VP', ('V', 'eats'), ('NP', 'fish'))),
    ('S', ('NP', 'fish'), ('VP', 'eats')),
    ('S', ('NP', 'she'), ('VP', 'eats')),
    ('S', ('NP', 'fish'), ('VP', ('V', 'eats'), ('NP', 'fish')))
]

# Learn rule probabilities
rule_probs = compute_rule_probabilities(training_trees)

# Input sentence
sentence = ['she', 'eats', 'fish']

# Parse and get best tree
best_tree, tree_prob = probabilistic_cyk_parser(rule_probs, sentence)

# --- Output ---
print("Best Parse Tree:")
print(best_tree)
print(f"\nProbability: {tree_prob:.6f}")

print("\nLearned Grammar Rules with Probabilities:")
for (lhs, rhs), p in rule_probs.items():
    print(f"{lhs} -> {' '.join(rhs)} [{p:.3f}]")

# ---------------------------
# CONVERT TO nltk.Tree & DISPLAY
# ---------------------------
def tuple_to_nltk_tree(tree_tuple):
    if isinstance(tree_tuple, tuple):
        label = tree_tuple[0]
        children = [tuple_to_nltk_tree(child) for child in tree_tuple[1:]]
        return Tree(label, children)
    else:
        return tree_tuple

if best_tree:
    print("\nPCYK Parse Tree (tuple)- returns the most probable parse tree:")
    print(best_tree)
    nltk_tree = tuple_to_nltk_tree(best_tree)
    nltk_tree.pretty_print()
else:
    print("No parse tree found.Invalid input sentence")

"""# **Seq2Seq - use this**"""

import pandas as pd

df = pd.read_csv("/content/seq2seq.csv",encoding="latin-1")
pairs = list(zip(df["Source"], df["Target"]))

pairs = []
with open("translations.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))

# parts = line.strip().split(',')
# parts = line.strip().split('|')

import torch
import torch.nn as nn
import torch.optim as optim
import random

# Dummy dataset
pairs = [
    ("i am a student", "je suis un étudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she is happy", "elle est heureuse"),
    ("they are playing", "ils jouent"),
    ("you are smart", "tu es intelligent")
]

# Tokenize and build vocab
def tokenize(sentence):
    return sentence.lower().split()

def build_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for sent in sentences:
        for word in tokenize(sent):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

src_vocab = build_vocab([src for src, tgt in pairs])
tgt_vocab = build_vocab([tgt for src, tgt in pairs])
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

def encode(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[word] for word in tokenize(sentence)] + [vocab["<eos>"]]

data = [(encode(src, src_vocab), encode(tgt, tgt_vocab)) for src, tgt in pairs]

SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMBED_SIZE = 32
HIDDEN_SIZE = 64

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)    # IF U R USING RNN ---> self.rnn = nn.RNN(emb_dim, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

        '''# RNN version
        outputs, hidden = self.rnn(embedded)
        return hidden, None'''

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)    # IF U R USING RNN ----> self.rnn = nn.RNN(emb_dim, hid_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

        ''' # RNN version
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, None'''

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        tgt_len = tgt.shape[0]
        batch_size = 1
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(tgt_len, tgt_vocab_size)
        hidden, cell = self.encoder(src)

        input = tgt[0]  # <sos>
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

encoder = Encoder(SRC_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(TGT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
for epoch in range(100):
    total_loss = 0
    for src, tgt in data:
        src_tensor = torch.tensor(src).unsqueeze(1)  # (seq_len, 1)
        tgt_tensor = torch.tensor(tgt).unsqueeze(1)

        optimizer.zero_grad()
        output = model(src_tensor, tgt_tensor)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt_tensor = tgt_tensor[1:].view(-1)

        loss = criterion(output, tgt_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# --------------------------
# Inference
# --------------------------
def translate(model, sentence, max_len=10):
    model.eval()
    tokens = encode(sentence, src_vocab)
    src_tensor = torch.tensor(tokens).unsqueeze(1)

    hidden, cell = model.encoder(src_tensor)
    input = torch.tensor([tgt_vocab["<sos>"]])

    result = []
    for _ in range(max_len):
        output, hidden, cell = model.decoder(input, hidden, cell)
        top1 = output.argmax(1).item()
        if top1 == tgt_vocab["<eos>"]:
            break
        result.append(inv_tgt_vocab[top1])
        input = torch.tensor([top1])

    return ' '.join(result)


# Test translation
print("\nTranslation Examples:")
print("Input: 'i am happy'")
print("Output:", translate(model, "i am happy"))

print("Input: 'you are smart'")
print("Output:", translate(model, "you are smart"))

"""# **Transformers**

# Machine Translation - Deeksha
"""

!pip install datasets
!pip install sacremoses
!pip install evaluate

import pandas as pd
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Config
model_name = "Helsinki-NLP/opus-mt-en-fr"
batch_size = 4
epochs = 3

# Load dataset (CSV with 'src' and 'tgt' columns)
df = pd.read_csv("translation.txt",sep="\t")  # Your dataset here
print(df)
dataset = Dataset.from_pandas(df)

# Load model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocessing
def preprocess(example):
    model_inputs = tokenizer(example["source"], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-en-fr",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs',
    report_to="none"  # Disable WandB, Comet, etc.
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save model
model.save_pretrained("finetuned-marian-en-fr")
tokenizer.save_pretrained("finetuned-marian-en-fr")

from transformers import MarianMTModel, MarianTokenizer

def translate(text, model_dir="finetuned-marian-en-fr"):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**encoded)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example
print(translate("Hi, I am deeksha"))


import nltk
nltk.download('punkt')
import evaluate

# Load BLEU metric
bleu = evaluate.load("bleu")

# Example list of predictions and references
predictions = ["Je t'aime beaucoup."]
references = [["Je vous aime beaucoup."]]  # note: reference must be a list of lists

# Compute BLEU
results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score: {results['bleu']:.4f}")

"""# Machine Translation - Prethika"""

import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize, download
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt

download('punkt')

# Load dataset
df = pd.read_csv("seq2seq.csv", encoding='latin-1')

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df.rename(columns={"Source": "src", "Target": "tgt"}))

# Load model & tokenizer
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Tokenization
def preprocess_data(batch):
    source = tokenizer(batch["src"], max_length=128, padding="max_length", truncation=True)
    target = tokenizer(batch["tgt"], max_length=128, padding="max_length", truncation=True)
    source["labels"] = target["input_ids"]
    return source

dataset = dataset.map(preprocess_data, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training args
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()

# Translate example and evaluate
def translate(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens, max_length=100)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def evaluate_translation(predicted, reference):
    print("\n🔹 Predicted Translation:\n", predicted)
    print("🔸 Reference Translation:\n", reference)

    bleu = sentence_bleu([word_tokenize(reference)], word_tokenize(predicted))
    print(f"\n🟢 BLEU Score: {bleu:.4f}")

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=[predicted], references=[reference])

    print("\n🟢 ROUGE Scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    plt.bar(scores.keys(), scores.values(), color="skyblue")
    plt.title("ROUGE Scores - Machine Translation")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()

translated_text = translate("Hello, how are you?")
evaluate_translation(translated_text, "Hola, cómo estás?")

"""# Text Sumamrization - Deeksha

"""

import pandas as pd
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer

# Load a smaller dataset sample
data = pd.read_csv("summary.txt",sep = '\t')
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
    evaluation_strategy="no",  # skip eval
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

"""# Text Sumamrization - Prethika"""

import pandas as pd
import torch
import nltk
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import matplotlib.pyplot as plt
#pip install rouge_score
#pip install evaluate
nltk.download("punkt")

# Load CSV
df = pd.read_csv("dummy_summarization_data.csv")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Convert pandas to HF dataset
dataset = Dataset.from_pandas(df)

# Preprocessing function
def preprocess_function(examples):
    inputs = [f"summarize: {text}" for text in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize and split dataset
dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.train_test_split(test_size=0.3, seed=42)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()

# Evaluation
sample_input = df.iloc[0]["article"]
input_ids = tokenizer(f"summarize: {sample_input}", return_tensors="pt", max_length=512, truncation=True).input_ids
output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("🔹Generated Summary:\n", summary_text)
print("🔸Reference Summary:\n", df.iloc[0]["summary"])

# BLEU Score
bleu = sentence_bleu([nltk.word_tokenize(df.iloc[0]["summary"])], nltk.word_tokenize(summary_text))
print(f"\n🟢 BLEU Score: {bleu:.4f}")

# ROUGE Score
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=[summary_text], references=[df.iloc[0]["summary"]])

for k, v in results.items():
    print(f"{k}: {v:.4f}")

plt.bar(results.keys(), results.values(), color="skyblue")
plt.title("ROUGE Scores")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

"""# Sentiment Analysis"""

!pip install transformers datasets scikit-learn

import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Load and prepare dataset
df = pd.read_csv("/content/drive/MyDrive/SEMESTER 8/Natural Language Processing/Data/sentiment_data.csv")  # Replace with your file path
label_map = {"negative": 0, "positive": 1}
df["label"] = df["label"].map(label_map)

dataset = Dataset.from_pandas(df)

# Split into train and test
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocessing
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(preprocess, batched=True)

# Minimal training arguments
training_args = TrainingArguments(
    output_dir="./tmp",  # Required even if not saving
    per_device_train_batch_size=4,
    num_train_epochs=3,
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
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model (quick)
trainer.train()

# Example usage: Sentiment prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "positive" if predicted_class == 1 else "negative"

# Test the prediction
example = "I dont love this new phone!"
print(predict_sentiment(example))