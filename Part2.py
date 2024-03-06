import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# Load datasets
with open('NER_train.json', 'r') as f:
    ner_train_data = json.load(f)
with open('NER_val.json', 'r') as f:
    ner_val_data = json.load(f)
with open('NER_test.json', 'r') as f:
    ner_test_data = json.load(f)

with open('ATE_train.json', 'r') as f:
    ate_train_data = json.load(f)
with open('ATE_val.json', 'r') as f:
    ate_val_data = json.load(f)
with open('ATE_test.json', 'r') as f:
    ate_test_data = json.load(f)

# Function to tokenize and pad sequences
def tokenize_pad(data, max_length):
    texts = [data[key]['text'] for key in data]
    labels = [data[key]['labels'] for key in data]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    X_pad = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return X_pad, labels

# Function to get vocabulary size
def get_vocab_size(data):
    texts = [data[key]['text'] for key in data]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return len(tokenizer.word_index) + 1

# Define models
def create_rnn_model(embedding_dim, rnn_units, label_to_index):
    model = Sequential([
        Embedding(input_dim=embedding_dim, output_dim=rnn_units, input_length=max_length),
        SimpleRNN(rnn_units, return_sequences=True),
        Dense(50, activation='relu'),
        Dense(len(label_to_index), activation='softmax')  # Output layer changed to softmax
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(embedding_dim, lstm_units, label_to_index):
    model = Sequential([
        Embedding(input_dim=embedding_dim, output_dim=lstm_units, input_length=max_length),
        LSTM(lstm_units, return_sequences=True),
        Dense(50, activation='relu'),
        Dense(len(label_to_index), activation='softmax')  # Output layer changed to softmax
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(embedding_dim, gru_units, label_to_index):
    model = Sequential([
        Embedding(input_dim=embedding_dim, output_dim=gru_units, input_length=max_length),
        GRU(gru_units, return_sequences=True),
        Dense(50, activation='relu'),
        Dense(len(label_to_index), activation='softmax')  # Output layer changed to softmax
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Train models
batch_size = 32
epochs = 10
max_length = 100  # Adjust according to your data

# Tokenize and pad sequences for NER dataset
X_train_ner, y_train_ner = tokenize_pad(ner_train_data, max_length)
print(np.array(y_train_ner).shape)
X_val_ner, y_val_ner = tokenize_pad(ner_val_data, max_length)
X_test_ner, y_test_ner = tokenize_pad(ner_test_data, max_length)

# Convert labels to numerical indices using LabelEncoder for NER dataset
label_encoder_ner = LabelEncoder()
label_encoder_ner.fit([label for seq in y_train_ner + y_val_ner + y_test_ner for label in seq])
y_train_ner = np.array([label_encoder_ner.transform(seq) for seq in y_train_ner])
y_val_ner = np.array([label_encoder_ner.transform(seq) for seq in y_val_ner])
y_test_ner = np.array([label_encoder_ner.transform(seq) for seq in y_test_ner])

# Pad sequences of labels for NER dataset
max_label_length_ner = max(len(seq) for seq in y_train_ner + y_val_ner + y_test_ner)
y_train_ner = pad_sequences(y_train_ner, padding='post', maxlen=max_label_length_ner)
y_val_ner = pad_sequences(y_val_ner, padding='post', maxlen=max_label_length_ner)
y_test_ner = pad_sequences(y_test_ner, padding='post', maxlen=max_label_length_ner)

# Tokenize and pad sequences for ATE dataset
X_train_ate, y_train_ate = tokenize_pad(ate_train_data, max_length)
X_val_ate, y_val_ate = tokenize_pad(ate_val_data, max_length)
X_test_ate, y_test_ate = tokenize_pad(ate_test_data, max_length)


# Convert labels to numerical indices using LabelEncoder for ATE dataset
label_encoder_ate = LabelEncoder()
label_encoder_ate.fit([label for seq in y_train_ate + y_val_ate + y_test_ate for label in seq])
y_train_ate = np.array([label_encoder_ate.transform(seq) for seq in y_train_ate])
y_val_ate = np.array([label_encoder_ate.transform(seq) for seq in y_val_ate])
y_test_ate = np.array([label_encoder_ate.transform(seq) for seq in y_test_ate])

# Pad sequences of labels for ATE dataset
max_label_length_ate = max(len(seq) for seq in y_train_ate + y_val_ate + y_test_ate)
y_train_ate = pad_sequences(y_train_ate, padding='post', maxlen=max_label_length_ate)
y_val_ate = pad_sequences(y_val_ate, padding='post', maxlen=max_label_length_ate)
y_test_ate = pad_sequences(y_test_ate, padding='post', maxlen=max_label_length_ate)

# Define models
models = {'RNN': create_rnn_model, 'LSTM': create_lstm_model, 'GRU': create_gru_model}

# Training and plotting for each model setup for NER dataset
for model_type, create_model_func in models.items():
    model = create_model_func(get_vocab_size(ner_train_data), 100, label_to_index_ner)  # Assuming embedding dimension is 100
    history = model.fit(X_train_ner, np.array(y_train_ner), batch_size=batch_size, epochs=epochs, validation_data=(X_val_ner, np.array(y_val_ner)), verbose=0)
    
    # Plotting
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Model (NER) - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type} Model (NER) - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test_ner, np.array(y_test_ner), verbose=0)
    y_pred = model.predict_classes(X_test_ner)
    test_f1 = f1_score(np.array(y_test_ner).flatten(), y_pred.flatten(), average='macro')
    print(f'{model_type} Model (NER) - Test Accuracy: {test_acc}, Test Macro-F1: {test_f1}')

# Training and plotting for each model setup for ATE dataset
for model_type, create_model_func in models.items():
    model = create_model_func(get_vocab_size(ate_train_data), 100, label_to_index_ate)  # Assuming embedding dimension is 100
    history = model.fit(X_train_ate, np.array(y_train_ate), batch_size=batch_size, epochs=epochs, validation_data=(X_val_ate, np.array(y_val_ate)), verbose=0)
    
    # Plotting
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Model (ATE) - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type} Model (ATE) - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test_ate, np.array(y_test_ate), verbose=0)
    y_pred = model.predict_classes(X_test_ate)
    test_f1 = f1_score(np.array(y_test_ate).flatten(), y_pred.flatten(), average='macro')
    print(f'{model_type} Model (ATE) - Test Accuracy: {test_acc}, Test Macro-F1: {test_f1}')
