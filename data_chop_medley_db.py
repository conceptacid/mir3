######################################################################################################################
# normalize, resample and chop all medley db files while retaining the directory structure
# elapsed time ca 24m.
######################################################################################################################

import os, sys, math, csv, datetime
import numpy as np
import scipy.io.wavfile as wave
from subprocess import call
from spectrograms import normalize_wave

sampling_rate = 22050                                                     # TODO: config
silence_level_threshold_db = -50                                          # TODO: config
chunk_duration = 3                                                        # TODO: config
chunk_duration_samples = sampling_rate * chunk_duration    


INPUT_DIR = '/media/ubuntu/AE08E30B08E2D17F/MedleyDB/Audio'                # TODO: config
OUTPUT_DIR = '/media/ubuntu/DATA/MIR/Chopped'                              # TODO: config

t0 = datetime.datetime.now()

if not os.path.exists(OUTPUT_DIR):
	raise "Output directory doesn't exists!"

chunks_csv = open(os.path.join(OUTPUT_DIR, 'chunks.csv'), 'wt')
chunks_csv_writer = csv.writer(chunks_csv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL) 

song_names = os.listdir(INPUT_DIR)
for song_name in song_names:
	raw_name = song_name + '_RAW'
	raw_path = os.path.join(INPUT_DIR, song_name, raw_name)
	
	if not os.path.exists(raw_path):
		continue
	
	track_names = os.listdir(raw_path)

	out_song_dir = os.path.join(OUTPUT_DIR, song_name)
	if not os.path.exists(out_song_dir):
		print("make dir ", out_song_dir)
		os.mkdir(out_song_dir)

	out_raw_path = os.path.join(out_song_dir, song_name + '_RAW')
	if not os.path.exists(out_raw_path):
		print("make dir ", out_raw_path)
		os.mkdir(out_raw_path)

	for track_name in track_names:
		if track_name.startswith('.'):
			continue

		print("Processing file...", track_name)

		track_path = os.path.join(raw_path, track_name)
		
		if not os.path.exists(track_path):
			continue

		out_track_temp_path = os.path.join(out_raw_path, track_name)
		track_name_without_ext = os.path.splitext(track_name)[0]
		
		if not os.path.exists(out_track_temp_path):
			call(["sox", track_path, "-r", str(sampling_rate), out_track_temp_path])

		sr, wave_data = wave.read(out_track_temp_path)
		if sr != sampling_rate:
			raise "Unexpected sampling rate: "+out_track_temp_path

		wave_data = normalize_wave(wave_data)
		wave_duration = wave_data.shape[0] / sampling_rate
		num_chunks = int(wave_duration) // chunk_duration
	
		for chunk_index in range(num_chunks):
			chunk_relative_name = track_name_without_ext + "_chunk_" + str(chunk_index) + ".wav"
			
			out_track_path = os.path.join(out_raw_path, track_name_without_ext)
			if not os.path.exists(out_track_path):
				os.mkdir(out_track_path)

			out_chunk_path = os.path.join(out_track_path, chunk_relative_name)
			chunk = wave_data[ chunk_index * chunk_duration_samples : (chunk_index+1) * chunk_duration_samples]
			
			mx = max(chunk.min(), chunk.max(), key=abs)
			mx = abs(mx)
			#print("max", 20*math.log10(mx), mx)
			
			# filter out silence
			mx = 20*math.log10(mx)
		
			# filter out silence
			if mx > silence_level_threshold_db and (not os.path.exists(out_chunk_path)):
				print("    writing chunk", out_chunk_path)
				chunks_csv_writer.writerow([song_name, track_name, chunk_relative_name])
				wave.write(out_chunk_path, sampling_rate, chunk)

		os.remove(out_track_temp_path)


t1 = datetime.datetime.now()
print("elapsed ", (t1-t0))		



