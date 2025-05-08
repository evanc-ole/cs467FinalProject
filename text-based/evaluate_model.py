#!/usr/bin/env python3
"""
evaluate_model.py

Loads a fine-tuned DistilBERT token-classification model saved in ./final_model,
reads backup.json for records, predicts the 3-digit invoice code for each record,
and prints overall accuracy plus every incorrect example with full OCR text.
"""

import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

def main():
    # Paths
    model_dir = "./final_model"
    data_file = "backup.json"

    # Load model & tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load data
    with open(data_file, "r") as f:
        data = json.load(f)
    records = [
        {"text": rec["text"], "target": rec["invoiceSnippet"]}
        for rec in data["__collections__"]["trainingData"].values()
    ]

    correct = 0
    incorrect_examples = []

    for rec in records:
        text = rec["text"]
        target = rec["target"]

        # Tokenize and move tensors to device
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits[0]  # [seq_len, num_labels]
        preds = torch.argmax(logits, dim=-1).tolist()
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

        # Reconstruct predicted invoice string
        invoice_chars = []
        for token, pred in zip(tokens, preds):
            if pred in (1, 2):  # B-INVOICE or I-INVOICE
                invoice_chars.append(token.replace("##", ""))
        predicted = "".join(invoice_chars)

        # Compare with target
        if predicted == target:
            correct += 1
        else:
            incorrect_examples.append({
                "text": text,
                "target": target,
                "predicted": predicted
            })

    total = len(records)
    accuracy = correct / total if total > 0 else 0.0

    # Output results
    print(f"Total records:       {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy:            {accuracy:.4f}\n")

    if incorrect_examples:
        print("Incorrect examples:\n")
        for idx, ex in enumerate(incorrect_examples, 1):
            print(f"Example {idx}:")
            print(f"  Target:    {ex['target']}")
            print(f"  Predicted: {ex['predicted']}")
            print("  Full OCR text:")
            print(f"{ex['text']}\n{'-'*80}\n")

if __name__ == "__main__":
    main()

