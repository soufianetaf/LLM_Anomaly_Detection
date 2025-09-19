import sys
import os
import pandas as pd
import re
import shutil
from glob import glob

# Ajouter le chemin du parser
sys.path.append("/content/logparser")
from logparser.Drain import LogParser

# -----------------------------
# PARAMÈTRES
# -----------------------------
input_dirs =   "/data/"  # logs HDFS originaux  # logs générés

output_dir = "/output/"
os.makedirs(output_dir, exist_ok=True)

log_file_pattern = "*.log"
log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
labels_filename = "anomaly_label.csv"

# Fichiers finaux
structured_file = os.path.join(output_dir, "HDFS_combined_structured.csv")
mapped_structured_file = structured_file.replace(".csv", "_mapped.csv")
templates_file = os.path.join(output_dir, "HDFS_combined_templates.csv")
mapped_templates_file = templates_file.replace(".csv", "_mapped.csv")

batch_size = 500_000  # nb de lignes par batch

# -----------------------------
# 1) Collecter tous les fichiers logs
# -----------------------------
log_files = []
for d in input_dirs:
    log_files.extend(glob(os.path.join(d, log_file_pattern)))
print(f" {len(log_files)} fichiers log détectés")

# -----------------------------
# 2) Initialiser Drain
# -----------------------------
parser = LogParser(
    log_format=log_format,
    indir=output_dir,  # Drain exige un dossier
    outdir=output_dir,
    depth=4,
    st=0.5,
    rex=[r"(blk_-?\d+)"],
    maxChild=100,
)

structured_files = []

# -----------------------------
# 3) Parsing en BATCH
# -----------------------------
batch_num = 0
for lf in log_files:
    print(f" Découpage et parsing du fichier : {lf}")
    with open(lf, "r") as f:
        batch = []
        for i, line in enumerate(f):
            batch.append(line)
            if (i + 1) % batch_size == 0:
                batch_num += 1
                batch_file = os.path.join(output_dir, f"batch_{batch_num}.log")
                with open(batch_file, "w") as fb:
                    fb.writelines(batch)
                parser.parse(batch_file)
                structured_files.append(batch_file + "_structured.csv")
                batch = []

        # Dernier batch
        if batch:
            batch_num += 1
            batch_file = os.path.join(output_dir, f"batch_{batch_num}.log")
            with open(batch_file, "w") as fb:
                fb.writelines(batch)
            parser.parse(batch_file)
            structured_files.append(batch_file + "_structured.csv")

print(f" Parsing terminé en {batch_num} batchs")

# -----------------------------
# 4) Fusion des structured
# -----------------------------
print(" Fusion des structured...")
first = True
with open(structured_file, "w") as fout:
    for file in structured_files:
        with open(file, "r") as fin:
            if first:
                fout.write(fin.read())  # header
                first = False
            else:
                next(fin)
                fout.write(fin.read())

df_parsed = pd.read_csv(structured_file)

# -----------------------------
# 5) Extraction BlockId
# -----------------------------
print(" Extraction BlockId...")
df_parsed["BlockId"] = df_parsed["Content"].apply(
    lambda x: re.search(r"(blk_-?\d+)", x).group(1) if re.search(r"(blk_-?\d+)", x) else None
)

# -----------------------------
# 6) Fusion labels
# -----------------------------
print(" Fusion anomaly_label.csv...")
labels_files = []
for d in input_dirs:
    csv_path = os.path.join(d, labels_filename)
    if os.path.exists(csv_path):
        labels_files.append(csv_path)

if labels_files:
    df_labels_list = [pd.read_csv(f) for f in labels_files]
    df_labels = pd.concat(df_labels_list, ignore_index=True)
    df_labels["Label"] = df_labels["Label"].map({"Normal": 0, "Anomaly": 1})
else:
    df_labels = pd.DataFrame(columns=["BlockId", "Label"])

df_parsed = df_parsed.merge(df_labels, on="BlockId", how="left")
df_parsed["Label"] = df_parsed["Label"].fillna(0).astype(int)

df_parsed.to_csv(structured_file, index=False)
print(" Labels ajoutés")
print(df_parsed["Label"].value_counts())

# -----------------------------
# 7) Fusion & mapping templates
# -----------------------------
print(" Fusion templates...")
df_templates_list = []
for file in structured_files:
    temp_file = file.replace("_structured.csv", "_templates.csv")
    if os.path.exists(temp_file):
        df_templates_list.append(pd.read_csv(temp_file))

if df_templates_list:
    df_temp = pd.concat(df_templates_list, ignore_index=True)
else:
    if os.path.exists(templates_file):
        df_temp = pd.read_csv(templates_file)
    else:
        df_temp = pd.DataFrame(columns=["EventId", "Content"])

# Remapping EventId
unique_event_ids = df_parsed["EventId"].unique()
event_mapping = {eid: f"E{i+1}" for i, eid in enumerate(unique_event_ids)}
df_parsed["EventId"] = df_parsed["EventId"].map(event_mapping)
df_temp["EventId"] = df_temp["EventId"].map(event_mapping)

df_parsed.to_csv(mapped_structured_file, index=False)
df_temp.to_csv(mapped_templates_file, index=False)

print(f" Résultats finaux : {mapped_structured_file}, {mapped_templates_file}")
