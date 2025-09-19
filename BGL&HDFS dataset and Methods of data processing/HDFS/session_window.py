import pandas as pd
import os

# -----------------------------
structured_file = "/output/HDFS_logs/HDFS_combined_structured_mapped.csv"
output_dir = "/output/"
train_file = os.path.join(output_dir, "HDFS_sessions_train.csv")
test_file = os.path.join(output_dir, "HDFS_sessions_test.csv")
test_ratio = 0.7  # proportion pour le test

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Charger le fichier structuré
df = pd.read_csv(structured_file)
print(" Fichier structuré HDFS chargé :", structured_file)

# -----------------------------
# Création des fichiers train/test
import random
random.seed(42)

with open(train_file, "w") as f_train, open(test_file, "w") as f_test:
    f_train.write("sequence,label\n")
    f_test.write("sequence,label\n")

    for block_id, group in df.groupby("BlockId"):
        events = group["EventId"].astype(str).tolist()
        label = 1 if group["Label"].max() == 1 else 0
        sequence_str = " ".join(events)

        # Tirage aléatoire pour train/test
        if random.random() < test_ratio:
            f_test.write(f"{sequence_str},{label}\n")
        else:
            f_train.write(f"{sequence_str},{label}\n")

print(f" Fichiers sauvegardés : {train_file}, {test_file}")
