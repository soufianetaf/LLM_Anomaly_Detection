# train_eval.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from lstm_model import LSTMClassifier
from word_embedding import prepare_data

# Définir un répertoire de base pour simplifier les chemins
base_dir = "/BGL&HDFS dataset and Methods of data processing/output"

# Paramètres avec les nouveaux chemins
train_file_hdfs = f"{base_dir}/HDFS_sessions_train.csv"
test_file_hdfs = f"{base_dir}/HDFS_sessions_test.csv"
train_file_bgl = f"{base_dir}/BGL_sequences_train.csv"
test_file_bgl = f"{base_dir}/BGL_sequences_test.csv"

# Préparation des données HDFS
train_loader_hdfs, test_loader_hdfs, vocab_size_hdfs = prepare_data(train_file_hdfs, test_file_hdfs)
# Préparation des données BGL
train_loader_bgl, test_loader_bgl, vocab_size_bgl = prepare_data(train_file_bgl, test_file_bgl)

# Utilisation du vocab_size plus grand des deux datasets
vocab_size = max(vocab_size_hdfs, vocab_size_bgl)

# Initialisation du modèle LSTM
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier(vocab_size).to(device)

# Fonction d'entraînement
def train_model(model, train_loader, num_epochs=5, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f" Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Fonction d'évaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    print(f"\n Résultats sur Test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Entraînement et évaluation sur HDFS
print("Entraînement sur HDFS...")
train_model(model, train_loader_hdfs)
evaluate_model(model, test_loader_hdfs)

# Entraînement et évaluation sur BGL
print("Entraînement sur BGL...")
train_model(model, train_loader_bgl)
evaluate_model(model, test_loader_bgl)
