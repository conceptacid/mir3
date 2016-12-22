################################################################################################################
# prepares spectrograms of a mini-experiment
################################################################################################################

import os, sys, csv, datetime
import numpy as np
import scipy.io.wavfile as wave
from spectrograms import *

EXPERIMENT_DIR = "/media/ubuntu/DATA/MIR/MiniExperiment5k"
CHUNK_DIR = "/media/ubuntu/DATA/MIR/Chopped"
expected_sample_rate = 22050

if not os.path.exists(EXPERIMENT_DIR):
	raise "Experiment dir not found"

def read_all(experiment_dir, data_dir, dataset_name):
	data = []
	with open(os.path.join(experiment_dir, dataset_name+'.csv'), 'rt') as csvfile:	
		csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in csv_reader:
			instrument, chunk_relative_path = row

			filename = os.path.basename(chunk_relative_path)
			filename = os.path.splitext(filename)[0]

			item = np.load(os.path.join(data_dir, filename+'.npy'))
			#print("data shape", item.shape)
			data.append(item)
	return np.stack(data)


def preprocess_dataset(experiment_dir, medley_db_chunk_dir, dataset_name, num_filt=128, generate_mean=True):
	dataset_path = os.path.join(experiment_dir, dataset_name)
	if not os.path.exists(dataset_path):
		os.mkdir(dataset_path)

	ffc_dir = os.path.join(dataset_path, 'FCC')
	if not os.path.exists(ffc_dir):
		os.mkdir(ffc_dir)

	mffc_dir = os.path.join(dataset_path, 'MFCC')
	if not os.path.exists(mffc_dir):
		os.mkdir(mffc_dir)	

	#img_fcc_dir = os.path.join(dataset_path, 'IMG_FCC')
	#if not os.path.exists(img_fcc_dir):
	#	os.mkdir(img_fcc_dir)		

	#img_mfcc_dir = os.path.join(dataset_path, 'IMG_MFCC')
	#if not os.path.exists(img_mfcc_dir):
	#	os.mkdir(img_mfcc_dir)	

	
	processed = 0
	print("Step 1/3 creating specrograms....")
	with open(os.path.join(experiment_dir, dataset_name+'.csv'), 'rt') as csvfile:	
		csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in csv_reader:
			instrument, chunk_relative_path = row

			filename = os.path.basename(chunk_relative_path)
			filename = os.path.splitext(filename)[0]

			chunk_path = os.path.join(medley_db_chunk_dir, chunk_relative_path)

			if not os.path.exists(chunk_path):
				print("Warning: chunk not found", chunk_path)

			sample_rate, signal = wave.read(chunk_path)
			assert sample_rate, expected_sample_rate

			frames = frame_signal(sample_rate, signal)
			fft = calculate_fft(sample_rate, frames, num_filt)
			mfft = calculate_mfcc(fft)

			np.save(os.path.join(ffc_dir, filename), fft.T)
			np.save(os.path.join(mffc_dir, filename), mfft.T)
			
			processed += 1
			print("Processed ", processed, " chunks", end='\r')

	print("\n")
	
	if generate_mean:
		print("Step 2/3 calculation FFT mean, stdev....")
		data = read_all(experiment_dir, ffc_dir, dataset_name)
		print("fft data shape", data.shape)
		fft_mean = np.mean(data, axis = 0)
		fft_stddev = np.std(data, axis = 0)
		np.save(os.path.join(dataset_path, "fft_mean"), fft_mean)
		np.save(os.path.join(dataset_path, "fft_std"), fft_stddev)
		print("FFT shape=", fft_mean.shape)

		print("Step 3/3 calculation mel-FFT mean, stdev....")
		data = read_all(experiment_dir, mffc_dir, dataset_name)
		print("mel-fft data shape", data.shape)
		mfft_mean = np.mean(data, axis = 0)
		mfft_stddev = np.std(data, axis = 0)
		np.save(os.path.join(dataset_path, "mfft_mean"), mfft_mean)	
		np.save(os.path.join(dataset_path, "mfft_std"), mfft_stddev)
		print("mel-FFT shape=", mfft_mean.shape)

			#chunk_path = os.path.join(medley_db_chunk_dir, chunk_relative_path)

			#if not os.path.exists(chunk_path):
			#	print("Warning: chunk not found", chunk_path)

			#sample_rate, signal = wave.read(chunk_path)

t0 = datetime.datetime.now()

preprocess_dataset(EXPERIMENT_DIR, CHUNK_DIR, 'train')
preprocess_dataset(EXPERIMENT_DIR, CHUNK_DIR, 'test', generate_mean = False)

t1 = datetime.datetime.now()
print("elapsed ", (t1-t0))		