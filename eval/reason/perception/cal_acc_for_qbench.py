import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('qbench_file', type=str, help='Path to the data file')
args = parser.parse_args()

# Load the data
with open(args.qbench_file, 'r') as f:
    data = json.load(f)

# Initialize counts and correct predictions
type_counts = {}
type_correct = {}
concern_counts = {}
concern_correct = {}

# Iterate over the data to populate counts and correct predictions
for item in data:
    t = item['type']
    c = item['concern']
    correct = item['correct_ans']
    # Ensure that 'pred_ans' is stripped of any extra characters like '.' and mapped correctly.
    try:
        pred_index = ord(item['pred_ans'].strip()[0]) - ord('A')
    except:
        pred_index = ord(item['response'].strip()[0]) - ord('A')
    try:
        pred = item['candidates'][pred_index]  # Map 'A', 'B', 'C', etc. to the correct answer
        print("Pred vs. Correct:", pred, correct)
    except:
        print(item)
        # print(pred_index, item['candidates'])
        # input()
    # Update type counts
    type_counts[t] = type_counts.get(t, 0) + 1
    type_correct[t] = type_correct.get(t, 0) + (1 if correct == pred else 0)

    # Update concern counts
    concern_counts[c] = concern_counts.get(c, 0) + 1
    concern_correct[c] = concern_correct.get(c, 0) + (1 if correct == pred else 0)

# Calculate accuracy for each type and concern
type_accuracy = {t: type_correct[t] / type_counts[t] for t in type_counts}
concern_accuracy = {c: concern_correct[c] / concern_counts[c] for c in concern_counts}

# Calculate overall accuracy
total_correct = sum(type_correct.values())
total_count = sum(type_counts.values())
overall_accuracy = total_correct / total_count if total_count > 0 else 0

# Print the results
print("Type Counts:", type_counts)
print("Concern Counts:", concern_counts)
print("Type Accuracy:", type_accuracy)
print("Concern Accuracy:", concern_accuracy)
print("Overall Accuracy:", overall_accuracy)