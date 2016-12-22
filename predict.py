import os, sys, csv, ast
import tensorflow as tf
import numpy
import scipy
#import pylab
import scipy.io.wavfile as wave

from spectrograms import *
from data_classes import *


song_name = "ClaraBerryAndWooldog_Boys"
song_suffix = "04_03"

chunk_duration_sec = 1.0
max_chunks = 0

n_bands   = 127
n_samples = 98 # 198
n_classes = 12
learning_rate = 0.000001 #0.0000001
max_epochs = 500
max_steps = 0 #10000
batch_size = 200

model_name = 'not_used'
model_path = './model/model-100'

if __name__ == "__main__":	

	#with open(os.path.join(spectrogram_path, 'onehot.csv'), 'rt') as csvfile:
	#	reader = csv.reader(csvfile, delimiter=';', quotechar='|')
	#	for row in reader:
	#		name, one_hot = row
	#		one_hot = ast.literal_eval(one_hot)
	#		class_one_hot_by_name[name] = np.array(one_hot)


	classes = ['bells, chimes', 'banjo, mandolin', 'drums or perc', 'bass', 'electric guitar', 'guitar', 'brass', 'keyboards', 'vibraphone', 'vocals or speech', 'strings', 'woodwind']


	with tf.Graph().as_default():
			g = tf.Graph()
			with g.as_default():

				print("---------------------------------------------------------------------------------------------------------------------------------")
				print("Initializing Model...")
				print("---------------------------------------------------------------------------------------------------------------------------------")
				X, Y, keep_prob, Y_pred, cost, optimizer, summary = build_model(n_bands, n_samples, n_classes, learning_rate, model_name=model_name)
			
				sess = tf.Session(graph=g)
				saver = tf.train.Saver()
				saver.restore(sess, model_path)
				print("Model ", model_path, "restored")
				#init = tf.global_variables_initializer()
				#sess.run(init)

				print("---------------------------------------------------------------------------------------------------------------------------------")
				print("Reading Wave...")
				print("---------------------------------------------------------------------------------------------------------------------------------")
				wav_path = os.path.join(ROOT_PATH, song_name, song_name + "_RAW", song_name+"_RAW_" + song_suffix + ".wav")
				sample_rate, wave_signal = wave.read(wav_path)





				total_duration = wave_signal.shape[0]
				chunk_duration = int(chunk_duration_sec * sample_rate)
				mx = max(wave_signal.min(), wave_signal.max(), key=abs)


				# normalize each chunk (!?????)
				wave_signal = wave_signal/(mx + 1e-08)



				print("Opened ",wav_path)
				print("Sample rate", sample_rate)
				print("Duration ",  total_duration // sample_rate, " sec")
				print("Max ", mx)
				print("        ")

				chunk_index = 0
				offset = 0
				while offset <= total_duration:
					
					signal = signal = wave_signal[ offset : (offset + chunk_duration)]
					mx = max(signal.min(), signal.max(), key=abs)
					
					


					

					frames = frame_signal(sample_rate, signal)
					filter_banks = calculate_fft(sample_rate, frames)
					mfcc = calculate_mfcc(filter_banks)

					# normalize
					filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
					mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

					data = numpy.stack([mfcc.T])
					data = numpy.expand_dims(data, axis=3)

					if data.shape != (1,n_bands, n_samples, 1):
						break;

					#print("shape=", data.shape)
					
					prediction = sess.run([Y_pred], feed_dict={X: data, keep_prob: 1.0})[0][0]
					index = numpy.argmax(prediction)
					#print("Prediction=", prediction, classes[index], "probability", prediction[index])
					

					print("Prediction=", classes[index], "probability", prediction[index], "Chunk ", chunk_index, offset / sample_rate, "Max ", mx)



					if max_chunks and chunk_index >= max_chunks:
						break
					chunk_index = chunk_index + 1
					offset += chunk_duration





	#signal = signal/(mx + 1e-08)

	

			
		


