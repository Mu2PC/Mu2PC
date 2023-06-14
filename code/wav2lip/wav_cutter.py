import numpy as np
import os
import librosa
from matplotlib import pyplot as plt


def main():
    return 0


if __name__ == '__main__':
    src_lang = 'it'
    split = ['test', 'valid', 'train', 'extra']
    split_process = split[1]
    root_path = f'/mnt/disk3/lilinjun/data/mtedx/tts_wav/{src_lang}_en/{split_process}'
    manifest_path = f'/mnt/disk3/lilinjun/data/mtedx/manifest/{src_lang}2en/{src_lang}_{split_process}_align_vtt.txt'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        clip_list = [line for line in f.readlines()]
    wav_list = []
    for clip in clip_list:
        yt_id, clip_ts, file = clip.split('/')
        audio_path = os.path.join(root_path, yt_id, clip_ts + '.wav')
        if os.path.exists(audio_path):
            wav_list.append(audio_path)
            wav = librosa.load(audio_path, sr=22050)
            plt.plot(np.array(wav))
            plt.show()
        break
    # print(wav_list)

    main()
