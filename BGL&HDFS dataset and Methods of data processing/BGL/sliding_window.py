import pandas as pd
import os
from sklearn.model_selection import train_test_split

# -----------------------------
# PARAMÈTRES
# -----------------------------
structured_file = "/output/logs.log_structured_mapped.csv"
output_dir = "/output/"
seq_len =   50    # longueur de la fenêtre
stride = 50     # pas du sliding window

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Charger le fichier structuré
# -----------------------------
df = pd.read_csv(structured_file)
print(" Fichier structuré chargé :", structured_file)

# -----------------------------
# Fonction sliding window avec pas
# -----------------------------
def sliding_window_sequences(events, labels, seq_len, stride):
    sequences, seq_labels = [], []
    for i in range(0, len(events) - seq_len + 1, stride):
        seq = events[i : i + seq_len]
        label = 1 if max(labels[i : i + seq_len]) == 1 else 0
        sequences.append(" ".join(seq))
        seq_labels.append(label)
    return sequences, seq_labels

# -----------------------------
# Créer les séquences
# -----------------------------
events = df["EventId"].astype(str).tolist()
labels = df["Label"].tolist()

X, y = sliding_window_sequences(events, labels, seq_len=seq_len, stride=stride)
print(f" Nombre total de séquences générées : {len(X)}")

# -----------------------------
# Split train / test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42, stratify=y
)

# Sauvegarde
train_file = os.path.join(output_dir, "BGL_sequences_train.csv")
test_file = os.path.join(output_dir, "BGL_sequences_test.csv")

pd.DataFrame({"sequence": X_train, "label": y_train}).to_csv(train_file, index=False)
pd.DataFrame({"sequence": X_test, "label": y_test}).to_csv(test_file, index=False)

print(f" Séquences train sauvegardées : {train_file}")
print(f" Séquences test sauvegardées : {test_file}")
print(f"Train: {len(X_train)} séquences | Test: {len(X_test)} séquences")

# -----------------------------
# Comptage normal / anormal
# -----------------------------
train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

print("\n Répartition des séquences :")
print(f"Train -> Normales: {train_counts.get(0,0)} | Anormales: {train_counts.get(1,0)}")
print(f"Test  -> Normales: {test_counts.get(0,0)} | Anormales: {test_counts.get(1,0)}")
