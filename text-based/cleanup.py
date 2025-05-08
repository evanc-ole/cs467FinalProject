import json
import sys
import os

def main():
    print("Starting cleanup process...")
    # Load the JSON data
    filename = "backup.json"
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        sys.exit(1)
        
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Navigate into the data structure to get trainingData
    training_data = data["__collections__"]["trainingData"]
    
    # We'll keep track of how many we remove
    removed_count = 0
    
    # Convert items to a list so we can modify training_data while iterating
    for record_id, record_content in list(training_data.items()):
        text = record_content.get("text", "")
        invoice_snippet = record_content.get("invoiceSnippet", "")
        
        # Check if the text contains the correct snippet
        if invoice_snippet not in text:
            del training_data[record_id]
            removed_count += 1
    
    # Overwrite the original file with the filtered data
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Done! Removed {removed_count} records where invoiceSnippet was not found in text.")

if __name__ == "__main__":
    main()
