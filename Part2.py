import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, RobertaModel
import json
import matplotlib.pyplot as plt

# Load dataset function
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Load pre-trained RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Load NER datasets
ner_train_data = load_dataset('NER_train.json')
ner_val_data = load_dataset('NER_val.json')
ner_test_data = load_dataset('NER_test.json')

# Load ATE datasets
ate_train_data = load_dataset('ATE_train.json')
ate_val_data = load_dataset('ATE_val.json')
ate_test_data = load_dataset('ATE_test.json')

# Define RNN-based model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(roberta_model.embeddings.word_embeddings.weight)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

# Define LSTM-based model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(roberta_model.embeddings.word_embeddings.weight)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output

# Define GRU-based model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(roberta_model.embeddings.word_embeddings.weight)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.fc(output)
        return output

# Define hyperparameters and training parameters
input_size = roberta_model.config.hidden_size
hidden_size = 128
output_size = 2  # Assuming binary classification (B, I) for BIO tagging
learning_rate = 0.00005
max_training_steps = 40000
training_batch_size = 256
num_epochs = 10

# Define DataLoader for training data
def collate_fn(batch):
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    return torch.tensor(inputs), torch.tensor(labels)

# Define function to train and evaluate models
def train_and_evaluate_model(model, optimizer, train_loader, val_loader, criterion):
    train_loss = []
    val_loss = []
    train_f1 = []
    val_f1 = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            val_loss_value = 0.0
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                val_loss_value += loss.item()
                preds = torch.argmax(outputs, dim=-1)
                val_preds.extend(preds.tolist())
                val_targets.extend(labels.tolist())
            val_loss.append(val_loss_value / len(val_loader))
            val_f1.append(f1_score(val_targets, val_preds, average='macro'))

    return train_loss, val_loss, val_f1

# Train and evaluate models for NER
ner_train_loader = DataLoader(ner_train_data, batch_size=training_batch_size, shuffle=True, collate_fn=collate_fn)
ner_val_loader = DataLoader(ner_val_data, batch_size=training_batch_size, shuffle=False, collate_fn=collate_fn)

ner_model_rnn = RNNModel(input_size, hidden_size, output_size)
ner_model_lstm = LSTMModel(input_size, hidden_size, output_size)
ner_model_gru = GRUModel(input_size, hidden_size, output_size)

optimizer_ner_rnn = optim.Adam(ner_model_rnn.parameters(), lr=learning_rate)
optimizer_ner_lstm = optim.Adam(ner_model_lstm.parameters(), lr=learning_rate)
optimizer_ner_gru = optim.Adam(ner_model_gru.parameters(), lr=learning_rate)

ner_criterion = nn.CrossEntropyLoss()

ner_rnn_train_loss, ner_rnn_val_loss, ner_rnn_val_f1 = train_and_evaluate_model(ner_model_rnn, optimizer_ner_rnn, ner_train_loader, ner_val_loader, ner_criterion)
ner_lstm_train_loss, ner_lstm_val_loss, ner_lstm_val_f1 = train_and_evaluate_model(ner_model_lstm, optimizer_ner_lstm, ner_train_loader, ner_val_loader, ner_criterion)
ner_gru_train_loss, ner_gru_val_loss, ner_gru_val_f1 = train_and_evaluate_model(ner_model_gru, optimizer_ner_gru, ner_train_loader, ner_val_loader, ner_criterion)

# Plot NER results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(ner_rnn_train_loss, label='RNN Training Loss')
plt.plot(ner_rnn_val_loss, label='RNN Validation Loss')
plt.plot(ner_lstm_train_loss, label='LSTM Training Loss')
plt.plot(ner_lstm_val_loss, label='LSTM Validation Loss')
plt.plot(ner_gru_train_loss, label='GRU Training Loss')
plt.plot(ner_gru_val_loss, label='GRU Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('NER Model Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ner_rnn_val_f1, label='RNN Validation F1')
plt.plot(ner_lstm_val_f1, label='LSTM Validation F1')
plt.plot(ner_gru_val_f1, label='GRU Validation F1')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('NER Model Validation F1 Score')
plt.legend()

plt.show()

# Train and evaluate models for ATE
ate_train_loader = DataLoader(ate_train_data, batch_size=training_batch_size, shuffle=True, collate_fn=collate_fn)
ate_val_loader = DataLoader(ate_val_data, batch_size=training_batch_size, shuffle=False, collate_fn=collate_fn)

ate_model_rnn = RNNModel(input_size, hidden_size, output_size)
ate_model_lstm = LSTMModel(input_size, hidden_size, output_size)
ate_model_gru = GRUModel(input_size, hidden_size, output_size)

optimizer_ate_rnn = optim.Adam(ate_model_rnn.parameters(), lr=learning_rate)
optimizer_ate_lstm = optim.Adam(ate_model_lstm.parameters(), lr=learning_rate)
optimizer_ate_gru = optim.Adam(ate_model_gru.parameters(), lr=learning_rate)

ate_criterion = nn.CrossEntropyLoss()

ate_rnn_train_loss, ate_rnn_val_loss, ate_rnn_val_f1 = train_and_evaluate_model(ate_model_rnn, optimizer_ate_rnn, ate_train_loader, ate_val_loader, ate_criterion)
ate_lstm_train_loss, ate_lstm_val_loss, ate_lstm_val_f1 = train_and_evaluate_model(ate_model_lstm, optimizer_ate_lstm, ate_train_loader, ate_val_loader, ate_criterion)
ate_gru_train_loss, ate_gru_val_loss, ate_gru_val_f1 = train_and_evaluate_model(ate_model_gru, optimizer_ate_gru, ate_train_loader, ate_val_loader, ate_criterion)

# Plot ATE results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(ate_rnn_train_loss, label='RNN Training Loss')
plt.plot(ate_rnn_val_loss, label='RNN Validation Loss')
plt.plot(ate_lstm_train_loss, label='LSTM Training Loss')
plt.plot(ate_lstm_val_loss, label='LSTM Validation Loss')
plt.plot(ate_gru_train_loss, label='GRU Training Loss')
plt.plot(ate_gru_val_loss, label='GRU Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ATE Model Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ate_rnn_val_f1, label='RNN Validation F1')
plt.plot(ate_lstm_val_f1, label='LSTM Validation F1')
plt.plot(ate_gru_val_f1, label='GRU Validation F1')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('ATE Model Validation F1 Score')
plt.legend()

plt.show()
