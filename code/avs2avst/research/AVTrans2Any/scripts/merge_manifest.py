import pandas as pd
from tqdm import tqdm
from examples.speech_to_text.data_utils import save_df_to_tsv
import csv
import os

MANIFEST_COLUMNS = ["id", "video", "audio", "n_frames", "tgt_text", "tgt_n_frames"]
audio = {}

manifest_files = ['data/En_Es/processed_mbart/train.tsv', 'data/En_Es/lrs3_mbart_processed/train.tsv']

write_file = 'data/En_Es/433h_es_covost/train.tsv'
manifest = {c: [] for c in MANIFEST_COLUMNS}

dataset = []
for manifest_file in manifest_files:
    with open(os.path.join(manifest_file), encoding='utf-8') as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        dataset += [dict(e) for e in reader]

assert len(dataset) > 0

for i, item in tqdm(enumerate(dataset)):
    manifest["id"].append(item['id'])
    if 'video' not in item:
        manifest["video"].append("None")
    else:
        manifest["video"].append(item['video'])
    manifest["audio"].append(item['audio'])
    manifest["n_frames"].append(item['n_frames'])
    manifest["tgt_text"].append(item['tgt_text'])
    manifest["tgt_n_frames"].append(item['tgt_n_frames'])

print(f"Writing manifest to {write_file}...")
save_df_to_tsv(pd.DataFrame.from_dict(manifest), write_file)

