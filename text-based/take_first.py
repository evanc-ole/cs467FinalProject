import json
import re

def predict_3_digits(text):
    """
    Tries three methods in sequence:
    1) First 3-digit number with (start-of-string OR whitespace) on the left 
       and (end-of-string OR whitespace) on the right.
    2) First 3-digit number with (start-of-string OR letter/whitespace) on the left 
       and (end-of-string OR letter/whitespace) on the right.
    3) Any 3-digit number anywhere.
    Returns the matched 3-digit string or None if no match is found.
    """
    # 1) 3-digit number with ^ or \s on the left and $ or \s on the right
    #    (no variable-width look-behind)
    match = re.search(r'(?:^|\s)(\d{3})(?=$|\s)', text)
    if match:
        return match.group(1)
    else: 
        return None
    
    # 2) 3-digit number with ^ or [A-Za-z\s] on the left
    #    and $ or [A-Za-z\s] on the right
    match = re.search(r'(?:^|[A-Za-z\s])(\d{3})(?=$|[A-Za-z\s])', text)
    if match:
        return match.group(1)
    
    # 3) Any 3-digit number
    match = re.search(r'(\d{3})', text)
    if match:
        return match.group(1)
    
    return None

def main():
    # Load the JSON data
    with open("backup.json", "r") as f:
        data = json.load(f)
    
    # Navigate to the training data. Your JSON structure may differ.
    training_data = data["__collections__"]["trainingData"]
    
    total = 0
    correct = 0
    incorrect_examples = []
    
    # Process each record
    for record_id, record_content in training_data.items():
        # 'text' is your full OCR text, 'invoiceSnippet' is the correct 3-digit label
        text = record_content["text"]
        invoice_snippet = record_content["invoiceSnippet"]
        
        # Generate a prediction using our baseline logic
        guess = predict_3_digits(text)
        
        # Track accuracy
        total += 1
        if guess == invoice_snippet:
            correct += 1
        else:
            incorrect_examples.append({
                "record_id": record_id,
                "text": text,
                "invoiceSnippet": invoice_snippet,
                "prediction": guess
            })
    
    # Calculate and print accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    
    # Print incorrect predictions
    if incorrect_examples:
        print("\nIncorrect Examples:")
        for ex in incorrect_examples:
            print(f"Record ID: {ex['record_id']}")
            print(f"  Text: {ex['text']}")
            print(f"  Correct: {ex['invoiceSnippet']}")
            print(f"  Predicted: {ex['prediction']}\n")

    print(f"Accuracy: {accuracy:.4f}")
if __name__ == "__main__":
    main()
