################################################################################################################
# mini experiment prepares a train/test datasets with smaller amount of data (subset of the whole MedleyDB)
################################################################################################################

import os, csv
import sys
import math
import numpy as np
from data_classes import *

MEDLEY_DATA_DIR = "/media/ubuntu/AE08E30B08E2D17F/MedleyDB/Audio"
CHUNK_DIR = "/media/ubuntu/DATA/MIR/Chopped"
OUTPUT_DIR = "/media/ubuntu/DATA/MIR/MiniExperiment40k"


################################################################################################################

def collect_chunks(medley_audio_path, medley_chopped_path):
	chunks_per_song = {}
	chunks_per_instrument = {}
	for song in os.listdir(medley_chopped_path):
	    raw = song + "_RAW"
	    if not os.path.exists(os.path.join(medley_chopped_path, song, raw)):
	        continue
	    collect_chunks_per_instrument(medley_audio_path, medley_chopped_path, song, chunks_per_instrument)
	    chunks_per_song[song] = 0
	    for track in os.listdir(os.path.join(medley_chopped_path, song, raw)):
	        for chunk in os.listdir(os.path.join(medley_chopped_path, song, raw, track)):
	            chunks_per_song[song] += 1
	return chunks_per_song, chunks_per_instrument           

################################################################################################################

def prepare_set(output_dir, set_name, num_chunks, chunks_per_instrument):
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	csv_file = open(os.path.join(output_dir, set_name + '.csv'), 'wt')
	csv_writer = csv.writer(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL) 

	done = False
	while not done:
		consumed = 0
		for instrument, chunks in chunks_per_instrument.items():
			if num_chunks == 0:
				done = True
				break
				
			if len(chunks) == 0: continue
			chunk = chunks.pop(0)
			csv_writer.writerow([instrument, chunk])	
			consumed += 1
			num_chunks -= 1
		if not consumed: break # exhausted


################################################################################################################	

chunks_per_song, chunks_per_instrument = collect_chunks(MEDLEY_DATA_DIR, CHUNK_DIR)

#num_chunks_per_instrument = {}
#for k, v in chunks_per_instrument.items():
#    num_chunks_per_instrument[k] = len(v)
    
prepare_set(OUTPUT_DIR, 'train', 40000, chunks_per_instrument)  
prepare_set(OUTPUT_DIR, 'test', 10000, chunks_per_instrument)  