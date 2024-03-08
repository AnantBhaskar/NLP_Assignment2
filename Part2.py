
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
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

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
        Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
        SimpleRNN(50, return_sequences=True),
        Dense(label_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(vocab_size, max_length, label_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
        LSTM(50, return_sequences=True),
        Dense(label_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(vocab_size, max_length, label_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
        GRU(50, return_sequences=True),
        Dense(label_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define a function to create a generator for data loading
def data_generator(data, batch_size, max_length, label_encoder):
    texts = [data[key]['text'] for key in data]
    labels = [data[key]['labels'] for key in data]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    while True:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            sequences = tokenizer.texts_to_sequences(batch_texts)
            X_batch = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
            # Pad labels to match the length of the input sequences
            padded_labels = [pad_sequences([label_encoder.transform(seq)], padding='post', maxlen=max_length)[0] for seq in batch_labels]
            y_batch = np.array(padded_labels)
            yield X_batch, y_batch

def get_vocab_size(data):
    texts = [data[key]['text'] for key in data]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return len(tokenizer.word_index) + 1

# Train models
def train_model(model, train_generator, val_generator, epochs):
    history = model.fit(train_generator,
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(X_val) // batch_size,
                        callbacks=[F1ScoreCallback()])
    return history

# Callback to calculate F1 score
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(F1ScoreCallback, self).__init__()
        self.f1_scores = []
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(X_train), axis=-1)
        y_true = np.argmax(y_train, axis=-1)
        f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro')
        self.f1_scores.append(f1)

        y_pred_val = np.argmax(self.model.predict(X_val), axis=-1)
        y_true_val = np.argmax(y_val, axis=-1)
        val_f1 = f1_score(y_true_val.reshape(-1), y_pred_val.reshape(-1), average='macro')
        self.val_f1_scores.append(val_f1)

# Plot Loss and F1 score
def plot_metrics(history, task, model_type):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss {task} Dataset for {model_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.f1_scores, label='Training F1 Score')
    plt.plot(history.val_f1_scores, label='Validation F1 Score')
    plt.title(f'Model F1 Score {task} Dataset for {model_type}')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load datasets
ner_train_data = load_data('NER_train.json')
ner_val_data = load_data('NER_val.json')
ner_test_data = load_data('NER_test.json')

ate_train_data = load_data('ATE_train.json')
ate_val_data = load_data('ATE_val.json')
ate_test_data = load_data('ATE_test.json')

# Train models
batch_size = 32
epochs = 10
max_length = 70

# Tokenize and pad sequences for NER dataset
X_train, y_train, label_to_index = tokenize_pad(ner_train_data, max_length)
X_val, y_val, _ = tokenize_pad(ner_val_data, max_length)

# Convert labels to numerical indices using LabelEncoder for NER dataset
label_encoder = LabelEncoder()
labels = [label for seq in y_train + y_val for label in seq]
label_encoder.fit(labels)


# Train RNN model for NER dataset
model_rnn_ner = create_rnn_model(get_vocab_size(ner_train_data), max_length, len(label_to_index))
train_generator = data_generator(ner_train_data, batch_size, max_length, label_encoder)
val_generator = data_generator(ner_val_data, batch_size, max_length, label_encoder)
history_rnn_ner = train_model(model_rnn_ner, train_generator, val_generator, epochs)
plot_metrics(history_rnn_ner, 'NER', 'RNN')

# Train LSTM model for NER dataset
model_lstm_ner = create_lstm_model(get_vocab_size(ner_train_data), max_length, len(label_to_index))
train_generator = data_generator(ner_train_data, batch_size, max_length, label_encoder)
val_generator = data_generator(ner_val_data, batch_size, max_length, label_encoder)
history_lstm_ner = train_model(model_lstm_ner, train_generator, val_generator, epochs)
plot_metrics(history_lstm_ner, 'NER', 'LSTM')

# Train GRU model for NER dataset
model_gru_ner = create_gru_model(get_vocab_size(ner_train_data), max_length, len(label_to_index))
train_generator = data_generator(ner_train_data, batch_size, max_length, label_encoder)
val_generator = data_generator(ner_val_data, batch_size, max_length, label_encoder)
history_gru_ner = train_model(model_gru_ner, train_generator, val_generator, epochs)
plot_metrics(history_gru_ner, 'NER', 'GRU')

# Tokenize and pad sequences for ATE dataset
X_train, y_train, label_to_index = tokenize_pad(ate_train_data, max_length)
X_val, y_val, _ = tokenize_pad(ate_val_data, max_length)

# Convert labels to numerical indices using LabelEncoder for ATE dataset
label_encoder = LabelEncoder()
labels = [label for seq in y_train + y_val for label in seq]
label_encoder.fit(labels)


# Train RNN model for ATE dataset
model_rnn_ate = create_rnn_model(get_vocab_size(ate_train_data), max_length, len(label_to_index))
train_generator = data_generator(ate_train_data, batch_size, max_length, label_encoder)
val_generator = data_generator(ate_val_data, batch_size, max_length, label_encoder)
history_rnn_ate = train_model(model_rnn_ate, train_generator, val_generator, epochs)
plot_metrics(history_rnn_ate, 'ATE', 'RNN')

# Train LSTM model for ATE dataset
model_lstm_ate = create_lstm_model(get_vocab_size(ate_train_data), max_length, len(label_to_index))
train_generator = data_generator(ate_train_data, batch_size, max_length, label_encoder)
val_generator = data_generator(ate_val_data, batch_size, max_length, label_encoder)
history_lstm_ate = train_model(model_lstm_ate, train_generator, val_generator, epochs)
plot_metrics(history_lstm_ate, 'ATE', 'LSTM')

# Train GRU model for ATE dataset
model_gru_ate = create_gru_model(get_vocab_size(ate_train_data), max_length, len(label_to_index))
train_generator = data_generator(ate_train_data, batch_size, max_length, label_encoder)
val_generator = data_generator(ate_val_data, batch_size, max_length, label_encoder)
history_gru_ate = train_model(model_gru_ate, train_generator, val_generator, epochs)
plot_metrics(history_gru_ate, 'ATE', 'GRU')
