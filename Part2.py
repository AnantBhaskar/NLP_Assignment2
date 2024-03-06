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
    return X_pad, labels, tokenizer.word_index

# Define models
def create_rnn_model(vocab_size, max_length, label_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
        SimpleRNN(100, return_sequences=True),
        Dense(50, activation='relu'),
        Dense(label_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(vocab_size, max_length, label_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
        LSTM(100, return_sequences=True),
        Dense(50, activation='relu'),
        Dense(label_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(vocab_size, max_length, label_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
        GRU(100, return_sequences=True),
        Dense(50, activation='relu'),
        Dense(label_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_vocab_size(data):
    texts = [data[key]['text'] for key in data]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return len(tokenizer.word_index) + 1

# Train models
batch_size = 32
epochs = 10
max_length = 70  # Adjust according to your data

# Tokenize and pad sequences for NER dataset
X_train_ner, y_train_ner, label_to_index_ner = tokenize_pad(ner_train_data, max_length)
X_val_ner, y_val_ner, _ = tokenize_pad(ner_val_data, max_length)
X_test_ner, y_test_ner, _ = tokenize_pad(ner_test_data, max_length)

# Convert labels to numerical indices using LabelEncoder for NER dataset
label_encoder_ner = LabelEncoder()
label_encoder_ner.fit([label for seq in y_train_ner + y_val_ner + y_test_ner for label in seq])

# Calculate max_label_length_ner
max_label_length_ner = max(len(seq) for seq in y_train_ner + y_val_ner + y_test_ner)

# Pad sequences to the maximum length
y_train_ner = [pad_sequences([label_encoder_ner.transform(seq)], padding='post', maxlen=max_label_length_ner)[0] for seq in y_train_ner]
y_val_ner = [pad_sequences([label_encoder_ner.transform(seq)], padding='post', maxlen=max_label_length_ner)[0] for seq in y_val_ner]
y_test_ner = [pad_sequences([label_encoder_ner.transform(seq)], padding='post', maxlen=max_label_length_ner)[0] for seq in y_test_ner]

# Convert to NumPy array
y_train_ner = np.array(y_train_ner)
y_val_ner = np.array(y_val_ner)
y_test_ner = np.array(y_test_ner)

# Tokenize and pad sequences for ATE dataset
X_train_ate, y_train_ate, label_to_index_ate = tokenize_pad(ate_train_data, max_length)
X_val_ate, y_val_ate, _ = tokenize_pad(ate_val_data, max_length)
X_test_ate, y_test_ate, _ = tokenize_pad(ate_test_data, max_length)

# Convert labels to numerical indices using LabelEncoder for ATE dataset
label_encoder_ate = LabelEncoder()
label_encoder_ate.fit([label for seq in y_train_ate + y_val_ate + y_test_ate for label in seq])

# Calculate max_label_length_ate
max_label_length_ate = max(len(seq) for seq in y_train_ate + y_val_ate + y_test_ate)

# Pad sequences to the maximum length for ATE dataset
y_train_ate = [pad_sequences([label_encoder_ate.transform(seq)], padding='post', maxlen=max_label_length_ate)[0] for seq in y_train_ate]
y_val_ate = [pad_sequences([label_encoder_ate.transform(seq)], padding='post', maxlen=max_label_length_ate)[0] for seq in y_val_ate]
y_test_ate = [pad_sequences([label_encoder_ate.transform(seq)], padding='post', maxlen=max_label_length_ate)[0] for seq in y_test_ate]

# Convert to NumPy array for ATE dataset
y_train_ate = np.array(y_train_ate)
y_val_ate = np.array(y_val_ate)
y_test_ate = np.array(y_test_ate)

# Define models
models = {'RNN': create_rnn_model, 'LSTM': create_lstm_model, 'GRU': create_gru_model}

# Define a function to plot Loss and F1 scores
def plot_metrics(history, model_type):
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Model - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_score'], label='Training Macro-F1-score')
    plt.plot(history.history['val_f1_score'], label='Validation Macro-F1-score')
    plt.title(f'{model_type} Model - F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Training and plotting for each model setup for NER dataset
for model_type, create_model_func in models.items():
    model = create_model_func(get_vocab_size(ner_train_data), max_length, len(label_to_index_ner))
    history = model.fit(X_train_ner, y_train_ner, batch_size=batch_size, epochs=epochs, validation_data=(X_val_ner, y_val_ner), verbose=0)
    
    # Calculate F1 scores
    y_pred_train = model.predict(X_train_ner)
    y_pred_val = model.predict(X_val_ner)
    f1_train = f1_score(np.argmax(y_train_ner, axis=1), np.argmax(y_pred_train, axis=1), average='macro')
    f1_val = f1_score(np.argmax(y_val_ner, axis=1), np.argmax(y_pred_val, axis=1), average='macro')

    # Store F1 scores in history object
    history.history['f1_score'] = [f1_train] * epochs
    history.history['val_f1_score'] = [f1_val] * epochs
    
    # Plotting metrics
    plot_metrics(history, model_type)
