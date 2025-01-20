import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('qbench_file', type=str, help='Path to the data file')
args = parser.parse_args()

# Load the data
with open(args.qbench_file, 'r') as f:
    data = json.load(f)

correct_counts = 0
# Iterate over the data to populate counts and correct predictions
for item in data:
    correct = item['correct_ans']
    # Ensure that 'pred_ans' is stripped of any extra characters like '.' and mapped correctly.
    try:
        pred_index = ord(item['pred_ans'].strip()[0].upper()) - ord('A')
    except:
        pred_index = ord(item['response'].strip()[0].upper()) - ord('A')
    try:
        pred = item['candidates'][pred_index]  # Map 'A', 'B', 'C', etc. to the correct answer
        # print("Pred vs. Correct:", pred, correct)
    except:
        print(item)
        # print(pred_index, item['candidates'])
        # input()
    # Update type counts
    if correct == pred:
        correct_counts += 1


overall_accuracy = correct_counts / len(data) if len(data) > 0 else 0
print("Overall Accuracy:", overall_accuracy)