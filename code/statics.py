import os
import librosa
from tqdm import tqdm
import cv2


def main():
    with open(filelist_path, 'r') as f:
        filelist = [fp.strip() for fp in f.readlines()]

    src_total = 0
    src_cnt = 0
    en_total = 0

    src_tokens= 0
    tgt_tokens = 0

    word_dict = {}
    en_dict = {}

    for file in tqdm(filelist):
        src_file = os.path.join(root, file, 'src.wav')
        tgt_file = os.path.join(root, file, 'en.wav')
        src_duration = librosa.get_duration(filename=src_file)
        en_duration = librosa.get_duration(filename=tgt_file)
        src_total += src_duration
        en_total += en_duration
        src_cnt += 1

        with open(os.path.join(root, file, 'src.txt'), 'r') as f:
            src_text = f.readlines()
            src_text = src_text[0].strip().split()
            src_tokens += len(src_text)

        for t in src_text:
            word_dict[t] = word_dict.get(t, 0) + 1

        with open(os.path.join(root, file, 'en.txt'), 'r') as f:
            en_text = f.readlines()
            en_text = en_text[0].strip().split()
            tgt_tokens += len(en_text)

        for t in en_text:
            en_dict[t] = en_dict.get(t, 0) + 1

    print(src_total / 60 / 60)
    print(en_total / 60 / 60)

    print(src_total / src_cnt)
    print(en_total / src_cnt)

    print(src_tokens, src_tokens/src_cnt)
    print(tgt_tokens, tgt_tokens/src_cnt)

    print(len(word_dict.keys()))
    print(len(en_dict.keys()))


def video_to_frames():
    v_path = f"/root/autodl-tmp/data/mlavt_tedx/es/main/5wSY_9EwKV0/005591_005698_00_en/src.avi"
    out_folder = f"/root/autodl-tmp/data/tmp_frames"
    os.makedirs(out_folder, exist_ok=True)

    # 打开视频
    video = cv2.VideoCapture(v_path)
    # 获取视频的帧速率
    fps = video.get(cv2.CAP_PROP_FPS)

    # 循环遍历视频的每一帧
    count = 0
    while video.isOpened():
        # 逐个读取视频的每一帧
        ret, frame = video.read()

        if ret:
            # 保存每一帧为一个图像文件
            cv2.imwrite(f'{out_folder}+/count.jpg', frame)
            count += 1
        else:
            break

    # 关闭视频
    video.release()


if __name__ == "__main__":
    # "/root/autodl-tmp/data/mlavt_tedx/es/main/DyDS4177Twc/013259_013298_00_en"
    root = "/root/autodl-tmp/data/mlavt_tedx/es"
    filelist_path = "/root/autodl-tmp/data/mlavt_tedx/es/file.list"

    video_to_frames()
    # main()