import json

# Loading the dataset
with open('ATE/Laptop_Review_Train.json', 'r') as f:
    train_data = json.load(f)

with open('ATE/Laptop_Review_Val.json', 'r') as f:
    val_data = json.load(f)

with open('ATE/Laptop_Review_Test.json', 'r') as f:
    test_data = json.load(f)

# Performing BIO chunking for aspect term extraction
def bio_chunking(text, aspects):
    tokens = text.split()
    labels = ['O'] * len(tokens)
    for aspect in aspects:
        start = aspect['from']
        end = aspect['to']
        aspect_tokens = text[start:end].split()
        for idx, token in enumerate(tokens):
            if token in aspect_tokens:
                labels[idx] = 'B'
                for i in range(1, len(aspect_tokens)):
                    if idx + i < len(tokens):
                        labels[idx + i] = 'I'
    return tokens, labels

# Processing and saving the train data
train_processed = {}
for i, entry in enumerate(train_data, start=1):
    text = entry['raw_words']
    aspects = entry['aspects']
    tokens, labels = bio_chunking(text, aspects)
    train_processed[str(i)] = {'text': text, 'labels': labels}

with open('ATE_train.json', 'w') as f:
    json.dump(train_processed, f, indent=4)

# Processing and saving the validation data
val_processed = {}
for i, entry in enumerate(val_data, start=1):
    text = entry['raw_words']
    aspects = entry['aspects']
    tokens, labels = bio_chunking(text, aspects)
    val_processed[str(i)] = {'text': text, 'labels': labels}

with open('ATE_val.json', 'w') as f:
    json.dump(val_processed, f, indent=4)

# Processing and saving the test data
test_processed = {}
for i, entry in enumerate(test_data, start=1):
    text = entry['raw_words']
    aspects = entry['aspects']
    tokens, labels = bio_chunking(text, aspects)
    test_processed[str(i)] = {'text': text, 'labels': labels}

with open('ATE_test.json', 'w') as f:
    json.dump(test_processed, f, indent=4)
