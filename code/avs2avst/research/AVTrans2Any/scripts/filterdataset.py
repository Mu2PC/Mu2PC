
import os
import sys
from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm
import logging
import glob
from scipy.io import wavfile
from multiprocessing import Pool
logger = logging.getLogger(f'main.{__name__}')

def read_wav(data):
    try:
        _, wav_data = wavfile.read(data['audio'])
        spec_length = len(wav_data)
    except:
        spec_length = 0

    return spec_length


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message):
    sys.stdout.write(f"\r{message}")


if __name__ == '__main__':

    write_file = "result.txt"
    error_file = "error.txt"
    manifest_files = glob.glob('/apdcephfs/share_1316500/nlphuang/data/speechtranslation_new/AVTranslation/manifest/433h_es/*.tsv')
    f_write = open(write_file, mode='w')
    f_error = open(error_file, mode='w')

    for manifest_file in manifest_files:
        with open(os.path.join(manifest_file)) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            s = [dict(e) for e in reader]
            assert len(s) > 0
            dataset = s

        n_data = len(dataset)
        skip = 0
        lengths = []

        pool = Pool(processes=os.cpu_count() - 3)
        for i, length in enumerate(pool.imap_unordered(read_wav, dataset), 1):
            if length == 0:
                f_error.write(f"{dataset[i]['audio']}\n")
                skip += 1
            else:
                lengths.append(length)

            bar = progbar(i, n_data)
            message = f'{bar} {i}/{n_data} '
            stream(message)

        lengths.sort(reverse=True)

        str = f"Dataset {Path(manifest_file).stem} total hours: {sum(lengths) / 16000 / 3600}, mean mel length: {np.mean(lengths)}, fail sample: {skip}"
        print(str)
        f_write.write(f"{str}\n")

