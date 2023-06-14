from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
					help='Name of saved checkpoint to load weights from', required=False,
					default='/root/autodl-tmp/model/wav2lip/checkpoints/wav2lip_gan.pth')

parser.add_argument('--face', type=str,
					help='Filepath of video/image that contains faces to use', required=False)
parser.add_argument('--audio', type=str,
					help='Filepath of video/audio file to use as raw audio source', required=False)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--thread_num', '-t', type=int, default=1, required=False)

parser.add_argument('--rank', '-r', type=int, default=0, required=False)

parser.add_argument('--cuda_device', '-d', type=int, default=0, required=False)

args = parser.parse_args()
args.img_size = 96
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
args.checkpoint_path = '/root/autodl-tmp/model/wav2lip/checkpoints/wav2lip_gan.pth'

# if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
# 	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size

	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			# print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for i, (rect, image) in enumerate(zip(predictions, images)):
		# cv2.imwrite(f'temp/{i}.jpg', image)
		if rect is None:
			# cv2.imwrite(f'temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results


def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		# print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
	#
	# xi = len(mels) / len(frames)
	# frames = frames[:10]
	for i, m in enumerate(mels):
		# 慢放补
		# idx = i // 3

		# # 倒放
		du_id = i // len(frames)
		idx = i%len(frames) if not du_id%2 else len(frames)-i%len(frames)-1
		# print(idx)
		# # 从头补
		# idx = 0 if args.static else i%len(frames)
		# # 只用最后一帧补
		# idx = i
		# if i >= len(frames): idx = len(frames) - 1
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint


def load_model(path):
	model = Wav2Lip()
	# print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()


def wav2lip_one_video(pair):
	face_in, audio_in = pair
	outfile_path = face_in.rsplit('/', 1)[0] + '/en.avi'
	# outfile_path = os.path.join(out_root, outfile_name)
	if os.path.exists(outfile_path):
		return
	outfile_folder = outfile_path.rsplit('/', 1)[0]
	os.makedirs(outfile_folder, exist_ok=True)

	if not os.path.isfile(face_in):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif face_in.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(face_in)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(face_in)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		# print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	# print("Number of frames available for inference: " + str(len(full_frames)))

	if not audio_in.endswith('.wav'):
		# print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_in, f'temp/temp_{args.rank}.wav')

		subprocess.call(command, shell=True)
		audio_in = f'temp/temp_{args.rank}.wav'

	wav = audio.load_wav(audio_in, 16000)
	mel = audio.melspectrogram(wav)
	# print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80. / fps
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
		i += 1

	# print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	# for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
	# 																total=int(
	# 																	np.ceil(float(len(mel_chunks)) / batch_size)))):
	for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
		if i == 0:
			model = load_model(args.checkpoint_path)
			# print("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(f'temp/result_{args.rank}.avi',
								  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
			# out = cv2.VideoWriter('/mnt/disk3/lilinjun/model/wav2lip/results/tmp_01.avi',
			# 					  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	# args.outfile 这个地方改成根据id生成，默认值'results/result_voice.mp4'
	# outfile = os.path.join('results', face_in.split('/')[-1].split('.')[0] + '.avi')
	# outfile = os.path.join('results', face_in.split('/')[-1].split('.')[0] + '_' + audio_in.split('/')[-1].split('.')[0] + '_.avi')
	# outfile = 'temp/fxxk_00_1.avi'
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel quiet'.format(audio_in,
																				  f'temp/result_{args.rank}.avi',
																				  outfile_path)
	# command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_in, f'temp/result_{split_process}_{rank}.avi', outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')


def generate_manifest():
	manifest = open(manifest_path, 'w', encoding='utf-8')
	manifest_list = []
	for yt_id in os.listdir(video_root):
		for utt_timestamp in os.listdir(os.path.join(video_root, yt_id)):
			face_path = os.path.join(video_root, yt_id, utt_timestamp, "00.avi")
			wav_file = "_".join([utt_timestamp.rsplit('_', 2)[0], utt_timestamp.rsplit('_', 1)[-1]]) + ".wav"
			wav_path = os.path.join(tts_root, yt_id, wav_file)
			if os.path.exists(face_path) and os.path.exists(wav_path):
				manifest_list.append([face_path, wav_path])
				manifest.writelines(f"{yt_id}/{utt_timestamp}\n")
	manifest.close()


def get_completed_wav2lip_num():
	num = 0
	for yt_id in os.listdir(out_root):
		num += len(os.listdir(os.path.join(out_root, yt_id)))
	print('out', num)
	return num


def main():
	# /root/autodl-tmp/data/mtedx/video_clip/es/-fx5k8gAPU8/002000_002100_00/00.avi/wav
	# with open(manifest_path, 'r', encoding='utf-8') as f:
	# 	wav2lip_id = [line.strip() for line in f.readlines()]
	## /root/autodl-tmp/data/mlavt_tedx/es/main/zJ8-ZoXvsu8/001853_001912_00_en
	# wav2lip_id = glob(os.path.join(video_root, '*/*_en/src.avi'))
	# with open('/root/autodl-tmp/data/mlavt_tedx/es/manifest_en_w2l.txt', 'w') as f:
	# 	for item in wav2lip_id:
	# 		f.writelines(item+'\n')
	with open('/root/autodl-tmp/data/mlavt_tedx/es/manifest_en_w2l.txt', 'r') as f:
		wav2lip_id = [i.strip() for i in f.readlines()]
	wav2lip_list = []
	for clip_id in wav2lip_id:
		face_path = clip_id
		wav_path = clip_id.rsplit('/', 1)[0] + '/en.wav'
		wav2lip_list.append([face_path, wav_path])
	# for yt_id in os.listdir(video_root):
	# 	for utt_timestamp in os.listdir(os.path.join(video_root, yt_id)):
	# 		face_path = os.path.join(video_root, yt_id, utt_timestamp, "00.avi")
	# 		wav_file = "_".join([utt_timestamp.rsplit('_', 2)[0], utt_timestamp.rsplit('_', 1)[-1]]) + ".wav"
	# 		wav_path = os.path.join(tts_root, yt_id, wav_file)
	# 		if os.path.exists(face_path) and os.path.exists(wav_path):
	# 			wav2lip_list.append([face_path, wav_path])

	for idx, pair in enumerate(tqdm(wav2lip_list)):
		# ii += 1
		# if ii == 5: break
		if idx % args.thread_num != args.rank: continue
		try:
			# print(pair)
			wav2lip_one_video(pair)
			print("{} wav2lip is completed.".format(idx))
		# except (ValueError, RuntimeError) as err:
		except Exception as err:
			# print(pair)
			print(pair, err, file=error_file)
			error_file.flush()
		# break


if __name__ == '__main__':
	# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
	# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
	src_lang = 'es'
	# /root/autodl-tmp/data/mtedx/target_video/es
	# /root/autodl-tmp/data/mlavt_tedx/es/main/zJ8-ZoXvsu8/001853_001912_00_en
	video_root = f"/root/autodl-tmp/data/mlavt_tedx/es/main"
	tts_root = f"/root/autodl-tmp/data/mlavt_tedx/es/main"
	out_root = f"/root/autodl-tmp/data/mlavt_tedx/es/main"
	# os.makedirs(out_root, exist_ok=True)

	# manifest_path = f"/root/autodl-tmp/data/mtedx/target_video/manifest_{src_lang}.txt"

	error_file = open(f'/root/autodl-tmp/data/mlavt_tedx/es/es_re_errors.txt', 'w')

	# rank = 0
	# pair = ['/mnt/disk3/lilinjun/data/mtedx/tmp/es/train/align_vtt/pycrop/GRUeop52l7k/005551_005584/00.avi',
	# 		'/mnt/disk3/lilinjun/data/mtedx/tts/es_en/train/_hNw_XkjODQ/007545_007637.wav']
	# wav2lip_one_video(pair)

	# generate_manifest()
	# get_completed_wav2lip_num()
	# print(aaa)

	main()
	error_file.close()
