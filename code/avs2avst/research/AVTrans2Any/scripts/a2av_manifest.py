import pandas as pd
from tqdm import tqdm
from examples.speech_to_text.data_utils import save_df_to_tsv
import csv
import os
import soundfile
# MANIFEST_COLUMNS = ["id", "video", "audio", "n_frames", "tgt_text", "tgt_n_frames"]
MANIFEST_COLUMNS = ["id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"]
audio = {}
for split in ['train', 'dev', 'test']:
    manifest_file = f'/apdcephfs/share_1316500/nlphuang/data/speechtranslation_new/AVTranslation/manifest/433h_es_filter/{split}.tsv'
    write_file = f'/apdcephfs/share_1316500/nlphuang/data/speechtranslation_new/AVTranslation/manifest/433h_es_filter_speech2speech/{split}.tsv'
    manifest = {c: [] for c in MANIFEST_COLUMNS}

    with open(os.path.join(manifest_file), encoding='utf-8') as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        dataset = [dict(e) for e in tqdm(reader)]
        assert len(dataset) > 0

    for i, item in tqdm(enumerate(dataset)):
        # manifest["id"].append(item['id'])
        # manifest["video"].append(item['src_audio'].replace('audio/', 'video/').replace('.wav', '.mp4'))
        # manifest["audio"].append(item['src_audio'])
        # manifest["n_frames"].append(item['src_n_frames'])
        # manifest["tgt_text"].append(item['tgt_audio'])
        # manifest["tgt_n_frames"].append(item['src_n_frames'])
        manifest["id"].append(item['id'])
        manifest["src_audio"].append(item['audio'])
        manifest["src_n_frames"].append(soundfile.info(item['audio']).frames)
        manifest["tgt_audio"].append(item['tgt_text'])
        manifest["tgt_n_frames"].append(item['tgt_n_frames'])

    print(f"Writing manifest to {write_file}...")
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), write_file)
