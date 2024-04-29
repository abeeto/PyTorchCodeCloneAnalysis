# !usr/bin/python
# coding=utf-8

# 提取特征

import os
import json
import librosa
import sklearn
from glob import glob
import matplotlib.mlab as mlab
from operator import itemgetter
import numpy as np
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure, binary_erosion
from pandas.io.json import json_normalize

IDX_FREQ_I = 0
IDX_TIME_J = 1

DEFAULT_FS = 44100

DEFAULT_WINDOW_SIZE = 4096

DEFAULT_OVERLAP_RATIO = 0.5

DEFAULT_FAN_VALUE = 15

DEFAULT_AMP_MIN = 10

PEAK_NEIGHBORHOOD_SIZE = 20

MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

PEAK_SORT = True

FINGERPRINT_REDUCTION = 20


NT_REDUCTION = 20

SAMPLE_SECOND = 30


def fingerprint(channel_samples, Fs=DEFAULT_FS,
				wsize=DEFAULT_WINDOW_SIZE,
				wratio=DEFAULT_OVERLAP_RATIO,
				fan_value=DEFAULT_FAN_VALUE,
				amp_min=DEFAULT_AMP_MIN):
	"""
	FFT the channel, log transform output, find local maxima, then return
	locally sensitive hashes.
	"""
	# FFT the signal and extract frequency components
	arr2D = mlab.specgram(
		channel_samples,
		NFFT=wsize,
		Fs=Fs,
		window=mlab.window_hanning,
		noverlap=int(wsize * wratio))[0]
	# print(arr2D[0].shape)

	# apply log transform since specgram() returns linear array
	arr2D = 10 * np.log10(arr2D)
	arr2D[arr2D == -np.inf] = 0  # replace infs with zeros

	# find local maxima
	local_maxima = get_2D_peaks(arr2D, amp_min=amp_min)

	# return hashes
	return generate_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D, amp_min=DEFAULT_AMP_MIN):
	# http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
	struct = generate_binary_structure(2, 1)
	neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

	# find local maxima using our fliter shape
	local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
	background = (arr2D == 0)
	eroded_background = binary_erosion(background, structure=neighborhood,
									   border_value=1)

	# Boolean mask of arr2D with True at peaks
	detected_peaks = local_max ^ eroded_background

	# extract peaks
	amps = arr2D[detected_peaks]
	j, i = np.where(detected_peaks)

	# filter peaks
	amps = amps.flatten()
	peaks = zip(i, j, amps)
	peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

	# get indices for frequency and time
	frequency_idx = [x[1] for x in peaks_filtered]
	time_idx = [x[0] for x in peaks_filtered]
	return [i for i in zip(frequency_idx, time_idx)]

def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
	"""
	Hash list structure:
	   sha1_hash[0:20] time_offset
	[(e05b341a9b77a51fd26, 32), ... ]
	"""
	if PEAK_SORT:
		peaks.sort(key=itemgetter(1))

	result = []
	for i in range(len(peaks)):
		for j in range(1, fan_value):
			if (i + j) < len(peaks):
				
				freq1 = peaks[i][IDX_FREQ_I]
				freq2 = peaks[i + j][IDX_FREQ_I]
				t1 = peaks[i][IDX_TIME_J]
				t2 = peaks[i + j][IDX_TIME_J]
				t_delta = t2 - t1

				if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
					# hstr = "%s|%s|%s"%(freq1, freq2, t_delta)
					result.append([freq1, freq2, t_delta])
					# hstr = hstr.encode('utf8')
					# print(hstr)
					# h = hashlib.sha1(hstr)
					# yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)
	return np.array(result)

def readFileWav(filename):
	fs, audiofile = wavfile.read(filename)
	audiofile = audiofile.T
	# channels = audiofile.shape[0]
	start = int((audiofile.shape[1] - (SAMPLE_SECOND * fs)) / 2)
	data = audiofile[:,start:(start + (SAMPLE_SECOND * fs))]
	return data, fs

def readFile(filename):
	from pydub import AudioSegment
	audiofile = AudioSegment.from_file(filename)
	start = int((audiofile.duration_seconds - SAMPLE_SECOND) / 2)
	audiofile = audiofile[start * 1000:(start + SAMPLE_SECOND) * 1000]
	data = np.frombuffer(audiofile._data, np.int16)
	channels = []
	for chn in range(audiofile.channels):
		channels.append(data[chn::audiofile.channels])
	channels = np.array(channels)
	return channels, audiofile.frame_rate


def getOneMp3(filename):
	channels, fs = readFile(filename)
	channel_samples = channels[0]
	# wsize = DEFAULT_WINDOW_SIZE
	# wratio = DEFAULT_OVERLAP_RATIO
	# arr2D = mlab.specgram(
	# 	channel_samples,
	# 	NFFT=wsize,
	# 	Fs=fs,
	# 	window=mlab.window_hanning,
	# 	noverlap=int(wsize * wratio))[0]
	# return arr2D
	return fingerprint(channel_samples, Fs = fs)
	# print(fs)
	# for channel in channels:
		# hashes = fingerprint(channel, Fs=fs)
		# print(len(set(hashes)))

def getSampleDataWave(filename):
	# print(idx, filename)
	data = getOneMp3(filename)
	data = data.T
	ret = {}
	for k, v in enumerate(data):
		ret['mean_%s'%k] = float(np.mean(v))
		ret['std_%s'%k] = float(np.std(v))
		ret['max_%s'%k] = float(np.max(v))
		ret['min_%s'%k] = float(np.min(v))
	return ret

def getSampleData(filename, savename, retStr = False):
	# from pydub import AudioSegment
	# audiofile = AudioSegment.from_file(filename)
	# start = int((audiofile.duration_seconds - SAMPLE_SECOND) / 2)

	ret = {'mid': savename}
	# y, sr = librosa.load(filename, mono=True, offset=start, duration=SAMPLE_SECOND)
	y, sr = librosa.load(filename, mono=True, duration=SAMPLE_SECOND)
	# y, sr = librosa.load(filename, mono=True, sr = 16000)
	onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=512, aggregate=np.median)
	tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
	spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	zcr = librosa.feature.zero_crossing_rate(y)
	zc = librosa.zero_crossings(y, pad = False)
	mfccs = librosa.feature.mfcc(y=y, sr=sr)
	# mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
	ret['chroma_stft'] = float(np.mean(chroma_stft))
	ret['spec_cent'] = float(np.mean(spec_cent))
	ret['spec_cent_pos'] = getCenter(spec_cent)
	ret['spec_bw'] = float(np.mean(spec_bw))
	ret['rolloff'] = float(np.mean(rolloff))
	ret['zcr'] = float(np.mean(zcr))
	ret['zc'] = int(sum(zc))
	ret['bpm'] = tempo
	# print(mfccs.var(axis=1))
	# ret['mfcc_mean'] = float(mfccs.mean(axis=1))
	# ret['mfcc_var'] = float(mfccs.var(axis=1))
	for k, v in enumerate(mfccs):
		ret['mfcc_%s'%k] = float(np.mean(v))
	ret.update(getSampleDataWave(filename))
	if retStr:
		return json.dumps(ret)
	return ret

def getCenter(spec_cent):
	spec_cent = spec_cent.T
	half = sum(spec_cent) / 2
	for i in range(spec_cent.shape[0]):
		num = sum(spec_cent[0:i])
		if num >= half:
			return float(i) / spec_cent.shape[0]

def audioToWav(filename):
	from pydub import AudioSegment
	audiofile = AudioSegment.from_file(filename)
	audiofile.export('temp.wav', format = 'wav')


if __name__ == '__main__':
	ret = getSampleData('/mnt/d/__odyssey__/TorchEncoder/music/59910.mp3', 'test', True)
	print(ret)
	ret = getSampleData('/mnt/d/__odyssey__/TorchEncoder/music/64093.mp3', 'test', True)
	print(ret)
	# fileList = glob('music/*.mp3')
	# fileList.sort()
	# result = []
	# peakFiles = set()
	# if os.path.isfile('peak_data.json'):
	# 	f = open('peak_data.json', 'r')
	# 	for i in f:
	# 		ret = json.loads(i)
	# 		result.append(ret)
	# 		peakFiles.add(ret['filename'])
	# 	f.close()
	# 	f = open('peak_data.json', 'a+')
	# else:
	# 	f = open('peak_data.json', 'w')
	# for idx, filename in enumerate(fileList):
	# 	if os.path.basename(filename) in peakFiles:
	# 		continue
	# 	print(idx, filename)
	# 	audioToWav(filename)
	# 	ret = getSampleData('temp.wav', os.path.basename(filename))
	# 	f.write(json.dumps(ret) + '\r\n')
	# 	f.flush()
	# 	result.append(ret)
	# f.close()
	# if os.path.isfile('temp.wav'):
	# 	os.remove('temp.wav')
	# header = list(ret.keys())
	# print(header)
	# data = json_normalize(result)
	# data.to_csv('data_peak.csv', columns = header)