import json
from collections import Counter

# Load data
file = 'data/meta_json/benchmark-v1/test/test_mcq_v1.json'

try:
    with open(file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Initialize lists for concerns and question types
concerns = []
question_types = []
concern_question_pairs = []

# Extract data from each item in the dataset
for item in data:
    mcq = item.get("mcq", [])
    concerns.extend([entry["concern"] for entry in mcq])
    question_types.extend([entry["question_type"] for entry in mcq])
    concern_question_pairs.extend([(entry["concern"], entry["question_type"]) for entry in mcq])

# Count occurrences
concern_counts = Counter(concerns)
question_type_counts = Counter(question_types)
concern_question_counts = Counter(concern_question_pairs)

# Print results
print("Concern counts:")
for concern, count in concern_counts.items():
    print(f"{concern}: {count}")

print("\nQuestion type counts:")
for question_type, count in question_type_counts.items():
    print(f"{question_type}: {count}")

print("\nConcern and question type pair counts:")
for pair, count in concern_question_counts.items():
    print(f"{pair}: {count}")
