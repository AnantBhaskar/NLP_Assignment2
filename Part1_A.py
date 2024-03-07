import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Loading the dataset
with open('NER/NER_TRAIN_JUDGEMENT.json', 'r') as f:
    data = json.load(f)

# Manually stratifying the data to ensure each class has at least two instances
class_instances = defaultdict(list)
for entry in data:
    annotations = entry['annotations']
    for annotation in annotations:
        for result in annotation['result']:  # Iterating over the list of annotations within 'result'
            label = result['value']['labels'][0]  # Accessing the 'labels' list and retrieve the first label
            class_instances[label].append(entry)

train_data = []
val_data = []

for instances in class_instances.values():
    instances_train, instances_val = train_test_split(instances, test_size=0.15, random_state=42)
    train_data.extend(instances_train)
    val_data.extend(instances_val)

def bio_chunking(text, annotations):
    tokens = text.split()
    labels = ['O'] * len(tokens)
    for annotation in annotations:
        for result in annotation['result']:
            start = result['value']['start']
            end = result['value']['end']
            label = result['value']['labels'][0]
            for i in range(len(tokens)):
                token_start = sum(len(tokens[j]) + 1 for j in range(i))
                token_end = token_start + len(tokens[i])
                if token_start >= start and token_end <= end:
                    if token_start == start:
                        labels[i] = 'B_' + label
                    else:
                        labels[i] = 'I_' + label
                elif token_start > end:
                    break
    return tokens, labels

# Processing and saving the train data
train_processed = {}
for entry in train_data:
    case_id = entry['id']
    text = entry['data']['text']
    annotations = entry['annotations']
    tokens, labels = bio_chunking(text, annotations)
    train_processed[case_id] = {'text': text, 'labels': labels}

with open('NER_train.json', 'w') as f:
    json.dump(train_processed, f, indent=4)

# Processing and saving the validation data
val_processed = {}
for entry in val_data:
    case_id = entry['id']
    text = entry['data']['text']
    annotations = entry['annotations']
    tokens, labels = bio_chunking(text, annotations)
    val_processed[case_id] = {'text': text, 'labels': labels}

with open('NER_val.json', 'w') as f:
    json.dump(val_processed, f, indent=4)
    
# Loading test data
with open('NER/NER_TEST_JUDGEMENT.json', 'r') as f:
    test_data = json.load(f)


# Processing and saving the test data
test_processed = {}
for entry in test_data:
    case_id = entry['id']
    text = entry['data']['text']
    annotations = entry['annotations']
    tokens, labels = bio_chunking(text, annotations)
    test_processed[case_id] = {'text': text, 'labels': labels}

with open('NER_test.json', 'w') as f:
    json.dump(test_processed, f, indent=4)
    
