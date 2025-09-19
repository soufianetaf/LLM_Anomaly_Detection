import sys
import os
import pandas as pd

# -----------------------------
# Chemins et paramètres
# -----------------------------
input_dir = "/data/"
output_dir = "/output/"
log_files = ["BGL.log", "synthetic_logs.log"]
log_format = "<Date> <Time> <Node> <Level> <Content>"

os.makedirs(output_dir, exist_ok=True)

structured_file = os.path.join(output_dir, "logs.log_structured.csv")
templates_file  = os.path.join(output_dir, "logs.log_structured.csv")

mapped_structured_file = os.path.join(output_dir, "logs.log_structured_mapped.csv")
mapped_templates_file  = os.path.join(output_dir, "logs.log_templates_mapped.csv")

sys.path.append("/content/logparser")
from logparser.Drain import LogParser

# -----------------------------
# 1) Parsing avec Drain pour tous les fichiers
# -----------------------------
print(" Parsing des logs avec Drain...")
parser = LogParser(
    log_format=log_format,
    indir=input_dir,
    outdir=output_dir,
    depth=4,
    st=0.5,
    rex=[],
    maxChild=100,
)

for log_file in log_files:
    parser.parse(log_file)

# -----------------------------
# 2) Fusionner tous les fichiers structured générés
# -----------------------------
dfs = []
for log_file in log_files:
    structured_path = os.path.join(output_dir, log_file + "_structured.csv")
    if os.path.exists(structured_path):
        df = pd.read_csv(structured_path)
        dfs.append(df)
df_parsed = pd.concat(dfs, ignore_index=True)
df_parsed.to_csv(structured_file, index=False)
print(f" Parsing terminé et fusionné : {structured_file}")

# -----------------------------
# 3) Ajout des labels
# -----------------------------
print(" Ajout des labels...")
first_chars = []
for log_file in log_files:
    file_path = os.path.join(input_dir, log_file)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            first_chars.extend([line.strip().split()[0] for line in f])

df_parsed["Label"] = [0 if x == "-" else 1 for x in first_chars]
df_parsed.to_csv(structured_file, index=False)
print(f" Labels ajoutés et sauvegardés : {structured_file}")
print("Distribution des labels :")
print(df_parsed["Label"].value_counts())

# -----------------------------
# 4) Fusionner tous les fichiers templates générés
# -----------------------------
template_dfs = []
for log_file in log_files:
    template_path = os.path.join(output_dir, log_file + "_templates.csv")
    if os.path.exists(template_path):
        df_temp_log = pd.read_csv(template_path)
        template_dfs.append(df_temp_log)

if template_dfs:
    df_temp = pd.concat(template_dfs, ignore_index=True)
    df_temp.to_csv(templates_file, index=False)
    print(f" Templates fusionnés et sauvegardés : {templates_file}")
else:
    df_temp = pd.DataFrame()
    print(" Aucun fichier templates trouvé")

# -----------------------------
# 5) Mapping EventId -> E1, E2, ...
# -----------------------------
print(" Mapping des EventId...")
unique_event_ids = df_parsed["EventId"].unique()
event_mapping = {eid: f"E{i+1}" for i, eid in enumerate(unique_event_ids)}

df_parsed["EventId"] = df_parsed["EventId"].map(event_mapping)
if not df_temp.empty:
    df_temp["EventId"] = df_temp["EventId"].map(event_mapping)

# -----------------------------
# 6) Sauvegarde des fichiers mappés
# -----------------------------
df_parsed.to_csv(mapped_structured_file, index=False)
if not df_temp.empty:
    df_temp.to_csv(mapped_templates_file, index=False)

print(f" Mapping terminé et sauvegardé :")
print(f"Structured mapped file: {mapped_structured_file}")
print(f"Templates mapped file: {mapped_templates_file if not df_temp.empty else 'Aucun template'}")
