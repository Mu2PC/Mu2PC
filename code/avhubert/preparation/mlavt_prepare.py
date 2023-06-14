# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, glob, subprocess, shutil, math
from datetime import timedelta
import tempfile
from collections import OrderedDict
from pydub import AudioSegment
from tqdm import tqdm
import json

def read_csv(csv_file, delimit=','):
    lns = open(csv_file, 'r').readlines()
    keys = lns[0].strip().split(delimit)
    df = {key: [] for key in keys}
    for ln in lns[1:]:
        ln = ln.strip().split(delimit)
        for j, key in enumerate(keys):
            df[key].append(ln[j])
    return df

def make_short_manifest(pretrain_dir, output_fn):
    subdirs = os.listdir(pretrain_dir)
    min_interval = 0.4
    max_duration = 15
    df = {'fid': [], 'sent': [], 'start': [], 'end': []}
    for subdir in tqdm(subdirs):
        txt_fns = glob.glob(os.path.join(pretrain_dir, subdir+'/*txt'))
        for txt_fn in txt_fns:
            fid = os.path.relpath(txt_fn, pretrain_dir)[:-4]
            lns = open(txt_fn).readlines()
            raw_text = lns[0].strip().split(':')[-1].strip()
            conf = lns[1].strip().split(':')[-1].strip()
            word_intervals = []
            for i_line, ln in enumerate(lns):
                if ln[:4] == 'WORD':
                    start_index = i_line
                    break
            for ln in lns[start_index+1:]:
                word, start, end, score = ln.strip().split()
                word_intervals.append([word, float(start), float(end)])
            if word_intervals[-1][-1] < max_duration:
                df['fid'].append(fid)
                df['sent'].append(raw_text)
                df['start'].append(0)
                df['end'].append(-1)
                continue
            sents, cur_sent = [], []
            for i_word, (word, start, end) in enumerate(word_intervals):
                if i_word == 0:
                    cur_sent.append([word, start, end])
                else:
                    assert start >= cur_sent[-1][-1], f"{fid} , {word}, start-{start}, prev-{cur_sent[-1][-1]}"
                    if start - cur_sent[-1][-1] > min_interval:
                        sents.append(cur_sent)
                        cur_sent = [[word, start, end]]
                    else:
                        cur_sent.append([word, start, end])
            if len(cur_sent) > 0:
                sents.append(cur_sent)
            for i_sent, sent in enumerate(sents):
                df['fid'].append(fid+'_'+str(i_sent))
                sent_words = ' '.join([x[0] for x in sent])
                if i_sent == 0:
                    sent_start = 0
                else:
                    sent_start = (sent[0][1] + sents[i_sent-1][-1][2])/2
                if i_sent == len(sents)-1:
                    sent_end = -1
                else:
                    sent_end = (sent[-1][2] + sents[i_sent+1][0][1])/2
                df['sent'].append(sent_words)
                df['start'].append(sent_start)
                df['end'].append(sent_end)
    durations = [y-x for x, y in zip(df['start'], df['end'])]
    num_long = len(list(filter(lambda x: x > 15, durations)))
    print(f"Percentage of >15 second: {100*num_long/len(durations)}%")
    num_long = len(list(filter(lambda x: x > 20, durations)))
    print(f"Percentage of >20 second: {100*num_long/len(durations)}%")
    with open(output_fn, 'w') as fo:
        fo.write('id,text,start,end\n')
        for i in range(len(df['fid'])):
            fo.write(','.join([df['fid'][i], df['sent'][i], '%.3f' % (df['start'][i]), '%.3f' % (df['end'][i])])+'\n')
    return

def trim_video_frame(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    decimal, fps = 9, 25
    for fid, start, end in zip(df['id'], df['start'], df['end']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    i_raw = -1
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total videos in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        raw_path = os.path.join(raw_dir, raw_fid+'.mp4')
        tmp_dir = tempfile.mkdtemp()
        cmd = ffmpeg + " -i " + raw_path + " " + tmp_dir + '/%0' + str(decimal) + 'd.png -loglevel quiet'
        subprocess.call(cmd, shell=True)
        num_frames = len(glob.glob(tmp_dir+'/*png'))
        for fid, start_sec, end_sec in fid_info:
            sub_dir = os.path.join(tmp_dir, fid)
            os.makedirs(sub_dir, exist_ok=True)
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            start_frame_id, end_frame_id = int(start_sec*fps), min(int(end_sec*fps), num_frames)
            imnames = [tmp_dir+'/'+str(x+1).zfill(decimal)+'.png' for x in range(start_frame_id, end_frame_id)]
            for ix, imname in enumerate(imnames):
                shutil.copyfile(imname, sub_dir+'/'+str(ix).zfill(decimal)+'.png')
            output_path = os.path.join(output_dir, fid+'.mp4')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [ffmpeg, "-i", sub_dir+'/%0'+str(decimal)+'d.png', "-y", "-crf", "20", output_path, "-loglevel", "quiet"]

            pipe = subprocess.call(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT) # subprocess.PIPE
        shutil.rmtree(tmp_dir)
    return

def trim_audio(csv_fn, raw_dir, output_dir, ffmpeg, rank, nshard):
    df = read_csv(csv_fn)
    raw2fid = OrderedDict()
    for fid, start, end in zip(df['id'], df['start'], df['end']):
        if '_' in fid:
            raw_fid = '_'.join(fid.split('_')[:-1])
        else:
            raw_fid = fid
        if raw_fid in raw2fid:
            raw2fid[raw_fid].append([fid, start, end])
        else:
            raw2fid[raw_fid] = [[fid, start, end]]
    i_raw = -1
    num_per_shard = math.ceil(len(raw2fid.keys())/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fid_info_shard = list(raw2fid.items())[start_id: end_id]
    print(f"Total audios in current shard: {len(fid_info_shard)}/{len(raw2fid.keys())}")
    for raw_fid, fid_info in tqdm(fid_info_shard):
        i_raw += 1
        tmp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(tmp_dir, 'tmp.wav')
        cmd = ffmpeg + " -i " + os.path.join(raw_dir, raw_fid+'.mp4') + " -f wav -vn -y " + wav_path + ' -loglevel quiet'
        subprocess.call(cmd, shell=True)
        raw_audio = AudioSegment.from_wav(wav_path)
        for fid, start_sec, end_sec in fid_info:
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec == -1:
                end_sec = 24*3600
            t1, t2 = int(start_sec*1000), int(end_sec*1000)
            new_audio = raw_audio[t1: t2]
            output_path = os.path.join(output_dir, fid+'.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_audio.export(output_path, format="wav")
        shutil.rmtree(tmp_dir)
    return

def trim_pretrain(root_dir, ffmpeg, rank=0, nshard=1, step=1):
    pretrain_dir = os.path.join(root_dir, 'pretrain')
    print(f"Trim original videos in pretrain")
    csv_fn = os.path.join(root_dir, 'short-pretrain.csv')
    if step == 1:
        print(f"Step 1. Make csv file {csv_fn}")
        make_short_manifest(pretrain_dir, csv_fn)
    else:
        print(f"Step 2. Trim video and audio")
        output_video_dir, output_audio_dir = os.path.join(root_dir, 'short-pretrain'), os.path.join(root_dir, 'audio/short-pretrain/')
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(output_audio_dir, exist_ok=True)
        trim_video_frame(csv_fn, pretrain_dir, output_video_dir, ffmpeg, rank, nshard)
        trim_audio(csv_fn, pretrain_dir, output_audio_dir, ffmpeg, rank, nshard)
    return

def prep_wav(lrs3_root, ffmpeg, rank, nshard):
    output_dir = f"{lrs3_root}/audio/"
    video_fns = glob.glob(lrs3_root + '/trainval/*/*mp4') + glob.glob(lrs3_root + '/test/*/*mp4')
    video_fns = sorted(video_fns)
    num_per_shard = math.ceil(len(video_fns)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    video_fns = video_fns[start_id: end_id]
    print(f"{len(video_fns)} videos")
    # subdirs = os.listdir(input_dir)
    for video_fn in tqdm(video_fns):
        base_name = '/'.join(video_fn.split('/')[-3:])
        audio_fn = os.path.join(output_dir, base_name.replace('mp4', 'wav'))
        os.makedirs(os.path.dirname(audio_fn), exist_ok=True)
        cmd = ffmpeg + " -i " + video_fn + " -f wav -vn -y " + audio_fn + ' -loglevel quiet'
        subprocess.call(cmd, shell=True)
    return

def get_file_label(lrs3_root):
    video_ids_total, labels_total = [], []
    for split in ['trainval', 'test']:
        subdirs = os.listdir(os.path.join(lrs3_root, split))
        for subdir in tqdm(subdirs):
            video_fns = glob.glob(os.path.join(lrs3_root, split, subdir, '*mp4'))
            video_ids = ['/'.join(x.split('/')[-3:])[:-4] for x in video_fns]
            for video_id in video_ids:
                txt_fn = os.path.join(lrs3_root, video_id+'.txt')
                label = open(txt_fn).readlines()[0].split(':')[1].strip()
                labels_total.append(label)
                video_ids_total.append(video_id)
    pretrain_csv = os.path.join(lrs3_root, 'short-pretrain.csv')
    df = read_csv(pretrain_csv)
    for video_id, label in zip(df['id'], df['text']):
        video_ids_total.append(os.path.join('short-pretrain', video_id))
        labels_total.append(label)
    video_id_fn, label_fn = os.path.join(lrs3_root, 'file.list'), os.path.join(lrs3_root, 'label.list')
    print(video_id_fn, label_fn)
    with open(video_id_fn, 'w') as fo:
        fo.write('\n'.join(video_ids_total)+'\n')
    with open(label_fn, 'w') as fo:
        fo.write('\n'.join(labels_total)+'\n')
    return

def get_file_label_mlavt(lrs3_root):
    # with open(f"{lrs3_root}/file.list", 'r', encoding="utf-8") as f:
    #     # /root/autodl-tmp/data/mlavt_tedx/fr/main/1aSFNRzgTaY/000600_000700_00_0_none
    #     # main/-JQrfEXTUnA/002700_002800_00_0_none
    #     id_list = ['/'.join(i.strip().rsplit('/', 3)[-3:]) for i in f.readlines()]
    #
    # with open(f"{lrs3_root}/file.list", 'w', encoding="utf-8") as f:
    #     for ids in id_list:
    #         f.writelines(ids + '\n')
    #
    # print(aaa)

    with open(f"{lrs3_root}/file.list", 'r', encoding="utf-8") as f:
        # main/eyRbdaAz1W0/004300_004400_00_0/00
        id_list = [i.strip() for i in f.readlines()]

    # src_asr_root = "/root/autodl-tmp/data/mtedx/asr/es"
    src_asr_root = lrs3_root
    label_list = []
    # /root/autodl-tmp/data/mtedx/asr/es/eyRbdaAz1W0/004300_004400_00.txt
    for yid in tqdm(id_list):
        # asr_file = yid.split('/', 1)[-1].rsplit('/', 1)[0].rsplit('_', 1)[0] + '.txt'
        asr_file = f"{prepare_task}.txt"
        asr_path = os.path.join(src_asr_root, yid, asr_file)
        if not os.path.exists(asr_path):
            print(asr_path)
        with open(asr_path, 'r') as f:
            src_label = f.readlines()
            assert len(src_label) == 1
            src_label = src_label[0].strip()
            label_list.append(src_label)

    with open(f"{lrs3_root}/label_{prepare_task}.list", 'w') as f:
        f.write('\n'.join(label_list)+'\n')

def get_chars_json(lrs3_root):
    src_asr_root = lrs3_root
    with open(f"{lrs3_root}/file.list", 'r', encoding="utf-8") as f:
        id_list = [i.strip() for i in f.readlines()]
    special_chars = dict()
    for yid in tqdm(id_list):
        asr_file = f"{prepare_task}.txt"
        asr_path = os.path.join(src_asr_root, yid, asr_file)
        with open(asr_path, 'r') as f:
            src_label = f.readlines()
            assert len(src_label) == 1
            src_label = src_label[0].strip()
            src_label = src_label.upper()
            for c in src_label:
                special_chars[c] = special_chars.get(c, 0) + 1
    special_chars_list = sorted(special_chars.items(), key=lambda x: x[1], reverse=True)
    special_chars_sorted = {}
    for rank, (d_key, d_val) in enumerate(special_chars_list, 1):
        special_chars_sorted[d_key] = d_val
    with open(f"{args.lrs3}/{prepare_task}_char.json", 'w') as fs:
        json.dump(special_chars_sorted, fs)


def filter_label(lrs3_root):
    with open(f"{args.lrs3}/src_char.json", 'r') as f:
        silent_chars_dict = json.load(f)
    silent_chars_check = list(silent_chars_dict.keys())

    with open(f"{args.lrs3}/src_char_audible.json", 'r') as f:
        audible_chars_dict = json.load(f)
    audible_chars_check = list(audible_chars_dict.keys())
    print(audible_chars_check)

    with open(f"{lrs3_root}/file.list", 'r') as f:
        # main/eyRbdaAz1W0/004300_004400_00_0/00
        id_list = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/label.list", 'r') as f:
        label_list = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/nframes.video", 'r') as f:
        nframes_video = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/nframes.audio", 'r') as f:
        nframes_audio = [i.strip() for i in f.readlines()]

    assert len(id_list) == len(label_list)
    assert len(id_list) == len(nframes_video)
    assert len(id_list) == len(nframes_audio)

    '''
    首先是删掉发声的特殊符号的数据
    % 替换为 por ciento
    ° 替换为 grados
    + 替换为 más
    然后过滤掉不发声的特殊符号
    '''

    f_filelist = open(f"{lrs3_root}/new_file.list", 'w')
    f_labellist = open(f"{lrs3_root}/new_label_{prepare_task}.list", 'w')
    f_nvideo = open(f"{lrs3_root}/new_nframes.video", 'w')
    f_naudio = open(f"{lrs3_root}/new_nframes.audio", 'w')

    cnt = 0
    remained_list = []
    for fid, label, nvideo, naudio in zip(id_list, label_list, nframes_video, nframes_audio):
        is_remain = True
        data_dir = os.path.join(lrs3_root, fid)
        if not len(os.listdir(data_dir)) == 6:
            continue
        label = label.upper()
        for c in label:
            if c in audible_chars_check:
                cnt += 1
                is_remain = False
                print(cnt, label)
                break
        if is_remain:
            remained_list.append([fid, label, nvideo, naudio])
            print(fid, file=f_filelist)
            print(label, file=f_labellist)
            print(nvideo, file=f_nvideo)
            print(naudio, file=f_naudio)

    f_filelist.close()
    f_labellist.close()
    f_nvideo.close()
    f_naudio.close()

def filter_label_en(lrs3_root):
    with open(f"{args.lrs3}/en_char.json", 'r') as f:
        silent_chars_dict = json.load(f)
    silent_chars_check = list(silent_chars_dict.keys())

    with open(f"{args.lrs3}/en_char_audible.json", 'r') as f:
        audible_chars_dict = json.load(f)
    audible_chars_check = list(audible_chars_dict.keys())
    print(audible_chars_check)

    with open(f"{lrs3_root}/file.list", 'r') as f:
        id_list = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/label.list", 'r') as f:
        src_label_list = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/label_en.list", 'r') as f:
        label_list = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/nframes.video", 'r') as f:
        nframes_video = [i.strip() for i in f.readlines()]

    with open(f"{lrs3_root}/nframes.audio", 'r') as f:
        nframes_audio = [i.strip() for i in f.readlines()]

    assert len(id_list) == len(label_list)
    assert len(id_list) == len(src_label_list)
    assert len(id_list) == len(nframes_video)
    assert len(id_list) == len(nframes_audio)

    '''
    首先是删掉发声的特殊符号的数据
    % 替换为 por ciento
    ° 替换为 grados
    + 替换为 más
    然后过滤掉不发声的特殊符号
    '''

    f_filelist = open(f"{lrs3_root}/new_file.list", 'w')
    f_src_labellist = open(f"{lrs3_root}/new_label_src.list", 'w')
    f_labellist = open(f"{lrs3_root}/new_label_en.list", 'w')
    f_nvideo = open(f"{lrs3_root}/new_nframes.video", 'w')
    f_naudio = open(f"{lrs3_root}/new_nframes.audio", 'w')

    cnt = 0
    remained_list = []
    for fid, src_label, label, nvideo, naudio in zip(id_list, src_label_list, label_list, nframes_video, nframes_audio):
        is_remain = True
        data_dir = os.path.join(lrs3_root, fid)
        # if not len(os.listdir(data_dir)) == 6:
        #     continue
        label = label.upper()
        for c in label:
            if c in audible_chars_check:
                cnt += 1
                is_remain = False
                print(cnt, label)
                break
        if is_remain:
            remained_list.append([fid, label, nvideo, naudio])
            print(fid, file=f_filelist)
            print(src_label, file=f_src_labellist)
            print(label, file=f_labellist)
            print(nvideo, file=f_nvideo)
            print(naudio, file=f_naudio)

    f_filelist.close()
    f_src_labellist.close()
    f_labellist.close()
    f_nvideo.close()
    f_naudio.close()

def filter_char(lrs3_root):
    with open(f"{args.lrs3}/src_char.json", 'r') as f:
        silent_chars_dict = json.load(f)
    silent_chars_check = list(silent_chars_dict.keys())
    print((silent_chars_check))

    with open(f"{args.lrs3}/src_char_remain.json", 'r') as f:
        remain_chars_dict = json.load(f)
    remain_chars_check = list(remain_chars_dict.keys())
    print(remain_chars_check)

    with open(f"{lrs3_root}/new_label.list", 'r') as f:
        label_list = [i.strip() for i in f.readlines()]

    remain_label_list = []
    for label in tqdm(label_list):
        remain_label = ''
        for c in label:
            if c == "%":
                c = " por ciento".upper()
                remain_label += c
                continue
            elif c == "°":
                c = " grados".upper()
                remain_label += c
                continue
            elif c == "+":
                c = "más".upper()
                remain_label += c
                continue
            if c in remain_chars_check:
                remain_label += c
        remain_label_list.append(remain_label)

    with open(f"{lrs3_root}/new_new_label.list", 'w') as f:
        for item in remain_label_list:
            f.writelines(item + '\n')

def filter_char_en(lrs3_root):
    with open(f"{args.lrs3}/en_char.json", 'r') as f:
        silent_chars_dict = json.load(f)
    silent_chars_check = list(silent_chars_dict.keys())
    print((silent_chars_check))

    with open(f"{args.lrs3}/en_char_remain.json", 'r') as f:
        remain_chars_dict = json.load(f)
    remain_chars_check = list(remain_chars_dict.keys())
    print(remain_chars_check)

    with open(f"{lrs3_root}/new_label_en.list", 'r') as f:
        label_list = [i.strip() for i in f.readlines()]

    remain_label_list = []
    for label in tqdm(label_list):
        remain_label = ''
        for c in label:
            if c == "%":
                c = " percent".upper()
                remain_label += c
                continue
            elif c == "°":
                c = " degrees".upper()
                remain_label += c
                continue
            elif c == "+":
                c = "plus".upper()
                remain_label += c
                continue
            if c in remain_chars_check:
                remain_label += c
        remain_label_list.append(remain_label)

    with open(f"{lrs3_root}/new_new_label_en.list", 'w') as f:
        for item in remain_label_list:
            f.writelines(item + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, default="/root/autodl-tmp/data/mlavt_tedx/es", help='lrs3 root dir')
    parser.add_argument('--ffmpeg', type=str, default="", help='path to ffmpeg')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    parser.add_argument('--step', type=int, default=1,
                        help='Steps (1: split labels, 2: trim video/audio, '
                             '3: prep audio for trainval/test, 4: get labels and file list)')
    args = parser.parse_args()

    src_lang = 'es'
    prepare_task = 'en'

    # prepare_task = 'src'
    # get_file_label_mlavt(args.lrs3)
    # get_chars_json(args.lrs3)
    # prepare_task = 'en'
    # get_file_label_mlavt(args.lrs3)
    # get_chars_json(args.lrs3)

    # filter_label(args.lrs3)
    # filter_char(args.lrs3)

    # filter_label_en(args.lrs3)
    filter_char_en(args.lrs3)

    # if args.step <= 2:
    #     trim_pretrain(args.lrs3, args.ffmpeg, args.rank, args.nshard, step=args.step)
    # elif args.step == 3:
    #     print(f"Extracting audio for trainval/test")
    #     prep_wav(args.lrs3, args.ffmpeg, args.rank, args.nshard)
    # elif args.step == 4:
    #     get_file_label(args.lrs3)
