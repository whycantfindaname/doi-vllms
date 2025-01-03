import json
import argparse

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing the predictions')
args = parser.parse_args()

# Load the JSON data from the file
with open(args.json_file, 'r') as file:
    data_list = json.load(file)

# Clean the predicted answer by removing trailing punctuation and standardizing case
def clean_pred_ans(pred_ans):
    return pred_ans.strip().upper().split('.')[0]  # Remove trailing period and convert to uppercase

# Calculate the accuracy
def calculate_accuracy(data_list):
    correct_count = 0
    total_count = len(data_list)

    for data in data_list:
        pred_ans_clean = clean_pred_ans(data["pred_ans"])
        correct_ans_clean = data["correct_ans"].strip().upper().rstrip('.')
        
        if pred_ans_clean == correct_ans_clean:
            correct_count += 1
    print(f"Correct count: {correct_count}, Total count: {total_count}")
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

# Calculate and print the accuracy
accuracy = calculate_accuracy(data_list)
print(f"Accuracy: {accuracy * 100:.2f}%")
