import os
import random
import numpy as np


if __name__ == '__main__':
    manifest = "/root/autodl-tmp/data/mlavt_tedx/es/file.list"
    fids = [ln.strip() for ln in open(manifest).readlines()]
    random.shuffle(fids)

    valid = fids[:1024]
    test = fids[1024:1024+1536]
    train = fids[1024+1536:]

    with open("/root/autodl-tmp/data/mlavt_tedx/es/valid.txt", 'w') as f:
        for fid in valid:
            f.writelines(fid + '\n')
    with open("/root/autodl-tmp/data/mlavt_tedx/es/test.txt", 'w') as f:
        for fid in test:
            f.writelines(fid + '\n')
    with open("/root/autodl-tmp/data/mlavt_tedx/es/train.txt", 'w') as f:
        for fid in train:
            f.writelines(fid + '\n')
