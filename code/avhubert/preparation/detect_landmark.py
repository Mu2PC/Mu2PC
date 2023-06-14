# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,os,pickle,math
import cv2,dlib,time
import numpy as np
from tqdm import tqdm

def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames

def detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, flist_fn, rank, nshard):

    def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)
    input_dir = root_dir #
    output_dir = landmark_dir #
    fids = [ln.strip() for ln in open(flist_fn).readlines()]
    num_per_shard = math.ceil(len(fids)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fids = fids[start_id: end_id]
    print(f"{len(fids)} files")
    for fid in tqdm(fids):
        print(f"rank: {args.rank}")
        output_fn = os.path.join(output_dir, fid, f'00.pkl')
        video_path = os.path.join(input_dir, fid, f'{args.task}.avi')
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str,
                        default="/root/autodl-tmp/data/mlavt_tedx/es", help='root dir')
    parser.add_argument('--landmark', type=str,
                        default="/root/autodl-tmp/data/mlavt_tedx/es/landmark", help='landmark dir')
    parser.add_argument('--manifest', type=str,
                        default="/root/autodl-tmp/data/mlavt_tedx/es/file.list", help='a list of filenames')
    parser.add_argument('--cnn_detector', type=str,
                        default="/root/autodl-tmp/model/av_hubert/avhubert/checkpoints/mmod_human_face_detector.dat",
                        help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', type=str,
                        default="/root/autodl-tmp/model/av_hubert/avhubert/checkpoints/shape_predictor_68_face_landmarks.dat",
                        help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--task', type=str,
                        default='src', help='src or tgt')
    parser.add_argument('--rank', type=int,
                        default=0, help='rank id')
    parser.add_argument('--nshard', type=int,
                        default=1, help='number of shards')
    parser.add_argument('--ffmpeg', type=str,
                        default="/usr/bin/ffmpeg", help='ffmpeg path')
    args = parser.parse_args()

    # /root/autodl-tmp/data/mlavt_tedx/es/main/-JQrfEXTUnA/002700_002800_00_0_none
    # /root/autodl-tmp/data/mlavt_tedx/es/landmark

    # fids = [ln.strip() for ln in open(args.manifest).readlines()]
    # # with open(args.manifest, 'w') as f:
    # #     for fid in fids:
    # #         f.writelines(fid + '\n')
    #
    # for fid in tqdm(fids):
    #     mv_from_path = os.path.join(args.root, fid, 'src.pkl')
    #     mv_to_dir = os.path.join(args.landmark, fid, '00.pkl')
    #     os.makedirs(os.path.dirname(mv_to_dir), exist_ok=True)
    #     os.system(f"mv {mv_from_path} {mv_to_dir}")
    #
    # print(aaa)

    args.landmark = "/root/autodl-tmp/data/mlavt_tedx/es/landmark_en"
    args.task = 'en'

    print("rank:{}, nshard:{}".format(args.rank, args.nshard))

    import skvideo
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    print(skvideo.getFFmpegPath())
    import skvideo.io
    detect_face_landmarks(args.face_predictor, args.cnn_detector, args.root, args.landmark, args.manifest, args.rank, args.nshard)
