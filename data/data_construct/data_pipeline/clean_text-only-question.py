import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('meta_file', type=str, help='Path to the data file')
parser.add_argument('clean_file', type=str, help='Path to the data file')
args = parser.parse_args()

# Load the data
with open(args.meta_file, 'r') as f:
    data = json.load(f)

# Initialize counts and correct predictions
type_counts = {}
type_correct = {}
concern_counts = {}
concern_correct = {}

# Filter the data to remove items with 4 or 5 correct answers in 'response'
filtered_data = []

for item in data:
    # Initialize correct_count to count correct responses
    correct_count = 0
    
    # Check each response and compare with the correct answer
    for resp in item['response']:
        # Ensure that 'resp' maps to a correct answer (A, B, C, D)
        try:
            pred_index = ord(resp.strip()[0]) - ord('A')  # Map 'A', 'B', 'C', 'D' to index
            pred = item['candidates'][pred_index]  # Get the candidate answer based on index
            if pred == item['correct_ans']:
                correct_count += 1
        except:
            continue  # Skip any invalid responses

    # If 4 or 5 answers are correct, skip this item
    if correct_count >= 1:
        continue
    
    # Add to the filtered list if not removed
    filtered_data.append(item)

# Output the filtered data if necessary
print("Filtered data length:", len(filtered_data))
input()
# Optionally, save the filtered data to a new file
with open(args.clean_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)
