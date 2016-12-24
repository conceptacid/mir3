import numpy as np
import csv
import ast
import os, sys
from random import shuffle
from data_classes import *

class Medley:
	def __init__(self, spectrogram_path, limit=0):
		
		self.directory_prefix = 'MFCC' #'FCC'                # todo: why FCC? fix that!
		self.chunks = {}
		self.spectrogram_path = spectrogram_path

		for i, v in class_one_hot_by_name.items():
			print(v, i)

		
		self.chunks['train'] = []
		self.chunks['test'] = []

		self.build_set('train')
		self.build_set('test')


		print()

		print("DEBUG                  ",len(self.chunks['train']))
		#print(self.chunks['train'][0][0].shape)
		#raise "STOP"

		self.batch_index = {}
		self.batch_index['train'] = 0
		self.batch_index['test'] = 0

		self.chunks['special'] = [0]
		self.chunks['special'][0] = self.chunks['test'][0]
		print("*****Special item is ", self.chunks['special'][0])

		self.normalize_data = False

		mean_path = os.path.join(self.spectrogram_path, 'train/mfft_mean.npy')                        # add m
		std_path = os.path.join(self.spectrogram_path, 'train/mfft_std.npy')
		if os.path.exists(mean_path) and os.path.exists(std_path):
			self.mean = np.load(mean_path)
			self.stddev = np.load(std_path)
			self.normalize_data = True
			print("Found mean, std files. Data shape=", self.mean.shape)


	def get_input_data_shape(self):
		if not self.normalize_data:
			raise "Data shape cannot be determined, calculate mean, std to continue...."
		return self.mean.shape	

	def build_set(self, set_name):
		self.chunks[set_name] = []
		with open(os.path.join(self.spectrogram_path, set_name+'.csv'), 'rt') as csvfile:
			reader = csv.reader(csvfile, delimiter=';', quotechar='|')
			for row in reader:
				instrument, wave_relative_path = row
				instrument_one_hot = class_one_hot_by_name[instrument]
				# get wave's filename without ext, save it as npy
				name = os.path.basename(wave_relative_path)
				name = os.path.splitext(name)[0]
				data_file_path = os.path.join( set_name, self.directory_prefix, name + '.npy')
				self.chunks[set_name].append([data_file_path, instrument_one_hot])

	def read_data(self, path):
		path = os.path.join(self.spectrogram_path, path)
		#print("READ from", path)
		data = np.load(path)
		if self.normalize_data:
			data = (data-self.mean)/self.stddev
		return data

	def new_epoch(self, train_index = 0, test_index = 0):
		self.batch_index['train'] = train_index
		self.batch_index['test'] = test_index
		shuffle(self.chunks['train'])
		shuffle(self.chunks['test'])

		#np.random.shuffle(self.chunks['train'])
		#np.random.shuffle(self.chunks['test'])

	def get_next_batch_names(self, batch_size, batch_type='train'):
		total_size = len(self.chunks[batch_type])

		has_more_data = self.batch_index[batch_type] != total_size
		if not has_more_data:
			self.batch_index[batch_type] = 0
		
		num_chunks_left = total_size - self.batch_index[batch_type]
		size = min(num_chunks_left, batch_size)
		start = self.batch_index[batch_type]
		end = start  + size
		batch_filenames = self.chunks[batch_type][ start : end ]
		self.batch_index[batch_type] += size
		has_more_data = self.batch_index[batch_type] != total_size
		return batch_filenames, has_more_data

	def get_next_batch(self, batch_size, batch_type='train'):	
		batch_filenames, has_more_data = self.get_next_batch_names(batch_size, batch_type)
		batch = np.stack( [ [self.read_data(entry[0]), entry[1]] for entry in batch_filenames ] )
		#print(batch)
		xs = np.stack(batch[:,0])
		ys = np.stack(batch[:,1])
		return xs, ys, has_more_data

	# returns one predefined item
	def get_special_batch(self):
		batch_filenames = self.chunks['special'][ 0 : 1 ]

		batch = np.stack( [ [self.read_data(entry[0]), entry[1]] for entry in batch_filenames ] )
		#print(batch)
		xs = np.stack(batch[:,0])
		ys = np.stack(batch[:,1])

		return xs,ys, True