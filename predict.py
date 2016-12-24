import os, sys, math
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wave

from config import CONFIG
from subprocess import call
from random import shuffle
from model import build_model
from data_classes import *
from spectrograms import *

n_classes = vector_by_class.shape[0]

chunk_duration             = CONFIG['chunk_duration']
supported_sampling_rate    = CONFIG['supported_sampling_rate']
silence_level_threshold_db = CONFIG['silence_level_threshold_db']
out_track_temp_path        = CONFIG['playout_track_temp_path']
model_name                 = CONFIG['predict_on_model']
directory_of_mean_and_dev  = CONFIG['directory_of_mean_and_dev']
init_load_model_from       = os.path.join(CONFIG['model_dir'], model_name)

if __name__ == "__main__":

	wavefile = sys.argv[1]
	print("sys.argv", sys.argv)
	print("loading wave ", wavefile)
	if not os.path.exists(wavefile):
		raise "input wave file not found"

	mean = None
	stddev = None
	normalize_data = False

	mean_path = os.path.join(directory_of_mean_and_dev, 'mfft_mean.npy')
	std_path = os.path.join(directory_of_mean_and_dev, 'mfft_std.npy')
	if os.path.exists(mean_path) and os.path.exists(std_path):
		mean = np.load(mean_path)
		stddev = np.load(std_path)
		normalize_data = True
		print("Found mean, std files. Data shape=", mean.shape)
	else:
		raise "Could not read man/stddev files"

	# read the mean, stddev and determine the data shape
	data_shape = mean.shape
	n_bands, n_samples = data_shape

	print("mean_shape = ", data_shape)

	sampling_rate, data = wave.read(wavefile)

	if sampling_rate != supported_sampling_rate:
		if os.path.exists(out_track_temp_path):
			os.remove(out_track_temp_path)
		call(["sox", wavefile, "-r", str(supported_sampling_rate), out_track_temp_path])
		sampling_rate, data = wave.read(out_track_temp_path)
		if sampling_rate != supported_sampling_rate:
			raise "Bad sampling rate: only "+str(supported_sampling_rate)+" is supported!"

	chunk_duration_samples = sampling_rate * chunk_duration    

	# normalize all
	data = normalize_wave(data)

	max_chunks = data.shape[0] // (sampling_rate*chunk_duration)

	g = tf.Graph()
	with g.as_default():
		with tf.Session(graph=g) as sess:
			X, _, keep_prob, Y_pred, _, _, _, _, _ = build_model(n_bands, n_samples, n_classes, 0.0, model_name=model_name)

			init = tf.global_variables_initializer()
			sess.run(init)

			saver = tf.train.Saver()
			if len(init_load_model_from) > 0:
				print("LOADING MODEL FROM", init_load_model_from)
				saver.restore(sess, init_load_model_from)
			else:
				print("Can't read the model")
				raise "Can't read the model"
				

			for chunk_index in range(max_chunks-1):

				#print("sampling_rate=", sampling_rate, "chunk_duration_samples", chunk_duration_samples)

				start = chunk_index * chunk_duration_samples / sampling_rate

				start = str(int(start // 60)) + ":" + str(int(start % 60)).ljust(10)

				# get a little chunk of samples
				chunk = data[ chunk_index * chunk_duration_samples : (chunk_index+1) * chunk_duration_samples]

				mx = max(chunk.min(), chunk.max(), key=abs)
				mx = abs(mx)
				mx = 20*math.log10(mx)

				# filter out silence
				if mx < silence_level_threshold_db:
					#print("offset", start, "silence detected", mx, "dB")
					print("offset", start, "-")
				else:
					frames = frame_signal(sampling_rate, chunk)
					fft = calculate_fft(sampling_rate, frames)
					mfcc = calculate_mfcc(fft)
					#print("shapes ", fft.shape, mfcc.shape)
					#mfcc = spectrogram(sampling_rate, chunk)
					mfcc = (mfcc.T - mean)/stddev;
					batch_xs = np.stack([mfcc])
					batch_xs = np.expand_dims(batch_xs, axis=3)

					predicted_one_hot = sess.run([Y_pred], feed_dict={X: batch_xs, keep_prob: 1.0})[0]
					result_dev = np.std(predicted_one_hot)
					if result_dev < 0.001:
						print("offset", start, "I don't know what is that!")							
					else:
						class_index = np.argmax(predicted_one_hot)
						confidence = predicted_one_hot[0, class_index]*100
						print("offset", start, " ", instrument_groups[class_index].rjust(15), " ", confidence, "%", "dev", result_dev)	
