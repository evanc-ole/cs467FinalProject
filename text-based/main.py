#!/usr/bin/env python3
# Requirements:
#   pip install torch torchvision torchaudio
#   pip install transformers datasets evaluate scikit-learn seqeval accelerate

import json
import pathlib
import re
import numpy as np
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# 1) Load data
RAW_PATH = pathlib.Path("backup.json")
data = json.loads(RAW_PATH.read_text())
records = [
    {"text": rec["text"], "target": rec["invoiceSnippet"]}
    for rec in data["__collections__"]["trainingData"].values()
]
print(f"Loaded {len(records)} records")

# 2) Tokenizer & labels
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
label_list = ["O", "B-INVOICE", "I-INVOICE"]
label2id   = {l:i for i,l in enumerate(label_list)}

# 3) Tokenize + label alignment
def tokenize_and_label(example):
    enc = tokenizer(
        example["text"],
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = [label2id["O"]] * len(enc["input_ids"])
    for m in re.finditer(re.escape(example["target"]), example["text"]):
        s, e = m.start(), m.end()
        for idx, (ts, te) in enumerate(enc["offset_mapping"]):
            if ts >= e or te <= s: 
                continue
            labels[idx] = label2id["B-INVOICE"] if ts == s else label2id["I-INVOICE"]
        break
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc

dataset = Dataset.from_list(records)
dataset = dataset.map(tokenize_and_label, batched=False)
print(dataset)

# 4) Metric
metric = evaluate.load("seqeval")
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_preds, true_labels = [], []
    for pred_row, lab_row in zip(preds, labels):
        pp, ll = [], []
        for p_id, l_id in zip(pred_row, lab_row):
            if l_id == -100: 
                continue
            pp.append(label_list[p_id]); ll.append(label_list[l_id])
        true_preds.append(pp); true_labels.append(ll)
    res = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": res["overall_precision"],
        "recall":    res["overall_recall"],
        "f1":        res["overall_f1"],
        "accuracy":  res["overall_accuracy"],
    }

# 5) 10-fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)
indices = np.arange(len(dataset))
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
    print(f"\n--- Fold {fold}/10 ---")
    train_ds = dataset.select(train_idx.tolist())
    val_ds   = dataset.select(val_idx.tolist())

    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_list),
        id2label={i:l for i,l in enumerate(label_list)},
        label2id=label2id
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir=f"./cv_fold_{fold}",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=50,
        save_strategy="no",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Fold {fold} →", metrics)
    fold_metrics.append(metrics)

# 6) Aggregate CV results
f1s = [m["eval_f1"] for m in fold_metrics]
print(f"\nAverage eval_f1 over 10 folds: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# 7) Train final model on all data and save
print("\nTraining final model on all data...")
final_model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label={i:l for i,l in enumerate(label_list)},
    label2id=label2id
)
final_args = TrainingArguments(
    output_dir="./final_model",
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=50,
    save_strategy="epoch",
    seed=42,
)
final_trainer = Trainer(
    model=final_model,
    args=final_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)
final_trainer.train()

# Save for later reuse
print("Saving final model to './final_model'")
final_trainer.save_model("./final_model")       # saves pytorch_model.bin + config.json
tokenizer.save_pretrained("./final_model")      # saves vocab & tokenizer config
print("Done.")
