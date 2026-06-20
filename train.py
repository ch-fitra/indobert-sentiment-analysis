from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

# === 1. Load dataset yang sudah disimpan di Fase 2 ===
dataset = load_from_disk("./smsa_dataset")

id2label = {0: "positive", 1: "neutral", 2: "negative"}
label2id = {"positive": 0, "neutral": 1, "negative": 2}

# === 2. Load tokenizer & model IndoBERT ===
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# === 3. Tokenisasi dataset ===
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True)

# === 4. Metrik evaluasi: accuracy + F1-macro (penting karena data imbalance) ===
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "f1_macro": f1_macro["f1"],
    }

# === 5. Training arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=True,                     # mixed precision, lebih cepat di RTX 4060
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

# === 6. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

# === 7. Mulai training ===
if __name__ == "__main__":
    print("Mulai training...")
    trainer.train()

    print("\nEvaluasi di test set...")
    test_results = trainer.evaluate(tokenized["test"])
    print(test_results)

    # Simpan model final
    model.save_pretrained("./model-sentiment-indo")
    tokenizer.save_pretrained("./model-sentiment-indo")
    print("\nModel disimpan di ./model-sentiment-indo")