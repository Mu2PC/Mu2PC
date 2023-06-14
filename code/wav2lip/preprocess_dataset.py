import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root",
					default="/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/data/lrs2/main",
					help="Root folder of the LRS2 dataset",
					required=False)
parser.add_argument("--preprocessed_root",
					default="/root/autodl-tmp/data/mlavt_tedx_preprocessed",
					help="Root folder of the preprocessed dataset",
					required=False)
parser.add_argument('--thread_num', '-t', type=int, default=1, required=False)
parser.add_argument('--rank', '-r', type=int, default=0, required=False)
parser.add_argument('--cuda_device', '-d', type=int, default=0, required=False)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
print(args.cuda_device)

# args.data_root = "/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/data/lrs2_es/main"
# args.preprocessed_root = "/mnt/disk3/lilinjun/data/lrs2_es_preprocessed"

# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
# 									device='cuda:{}'.format(id)) for id in range(args.ngpu)]
fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
									device='cuda:0')]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):
	vfile = os.path.join(vfile, "src.avi")
	vidname, clipname = vfile.split('/')[-3:-1]
	fulldir = path.join(args.preprocessed_root, src_lang, vidname, clipname)

	wavpath = path.join(fulldir, 'audio.wav')
	if not os.path.exists(wavpath):
		video_stream = cv2.VideoCapture(vfile)

		frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			frames.append(frame)

		os.makedirs(fulldir, exist_ok=True)

		batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

		i = -1
		for fb in batches:
			preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

			for j, f in enumerate(preds):
				i += 1
				if f is None:
					continue

				x1, y1, x2, y2 = f
				cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])

def process_audio_file(vfile, args):
	vfile = os.path.join(vfile, "src.wav")
	# '/root/autodl-tmp/data/mlavt_tedx/es/main/-JQrfEXTUnA/002700_002800_00_0_none/src.wav'
	vidname, clipname = vfile.split('/')[-3:-1]

	fulldir = path.join(args.preprocessed_root, src_lang, vidname, clipname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')
	if not os.path.exists(wavpath):
		command = template.format(vfile, wavpath)
		subprocess.call(command, shell=True)

	
def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile, args, gpu_id)
		process_audio_file(vfile, args)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	filelist_path = f"/root/autodl-tmp/data/mlavt_tedx/{src_lang}/file.list"
	with open(filelist_path, 'r') as f:
		filelist = [line.strip() for line in f.readlines()]

	print('Started processing for {} with {} GPUs'.format(filelist_path, args.ngpu))

	# filelist = glob(path.join(args.data_root, '*/*/*.avi'))

	# jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	# p = ThreadPoolExecutor(args.ngpu)
	jobs = []
	for i, vfile in enumerate(filelist):
		if i % args.thread_num != args.rank: continue
		jobs.append((vfile, args, 0))
	# mp_handler(jobs[0])

	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	# print('Dumping audios...')
	#
	# for i, vfile in enumerate(tqdm(filelist)):
	# 	if i % args.thread_num != args.rank: continue
	# 	try:
	# 		process_audio_file(vfile, args)
	# 	except KeyboardInterrupt:
	# 		exit(0)
	# 	except:
	# 		traceback.print_exc()
	# 		continue

if __name__ == '__main__':
	src_lang = "es"
	main(args)