##################################################################################################
# defines groups of instruments used for classification
# defines one-hot vectors for the classes
##################################################################################################

import numpy as np
import os, sys, yaml

# instruments by groups
instrument_by_group = {'woodwind': ['flute', 'clarinet', 'bassoon'], 
											'vibraphone': ['glockenspiel', 'vibraphone'],
											'bells, chimes': ['cowbell', 'chimes'],
											'banjo, mandolin': ['banjo', 'mandolin'],
											'strings': ['viola', 'violin', 'violin section', 'cello', 'string section'],
											'bass': ['double bass', 'electric bass'],
											'electric guitar': ['distorted electric guitar'],
											'guitar': ['lap steel guitar', 'acoustic guitar', 'clean electric guitar'],
											'vocals or speech': ['male singer', 'male speaker', 'vocalists', 'female singer', 'male rapper' ],
											'keyboards' : ['melodica', 'piano','tack piano','synthesizer'],
											'drums or perc' : ['drum set', 'kick drum', 'toms', 'cymbal', 'snare drum','drum machine',
																							 'shaker', 'high hat','cabasa', 'guiro', 'claps', 'bass drum',
																							 'auxiliary percussion', 'timpani', 'tabla', 'tambourine'],
											 'brass': ['trombone', 'horn section', 'trumpet', 'brass section', 'tenor saxophone', 'french horn', 
																 'euphonium']}
group_by_instrument = {}

#groups by instruments
for g, instuments in instrument_by_group.items():
		for i in instuments:
				group_by_instrument[i] = g


# just the names of the classes
# to get the group name by predictions use instrument_by_group[ int(np.argmax(predictions, axis=1)) ]
instrument_groups = ['woodwind','vibraphone','bells, chimes', 'banjo, mandolin', 'strings', 'bass', 'electric guitar','guitar', 'vocals or speech', 'keyboards','drums or perc','brass']

# one-hot vectors for each instrument group
vector_by_class = np.eye(len(instrument_groups))

class_one_hot_by_name = {}

for group_index, group in enumerate(instrument_groups):
	class_one_hot_by_name[group] = vector_by_class[group_index]
	print("group", group , "->",vector_by_class[group_index])

def collect_chunks_per_instrument(medley_audio_path, medley_chopped_path, name, chunks_by_instrument):
		dir = os.path.join(medley_audio_path, name)
		yaml_file = name + '_METADATA.yaml'
		path = os.path.join(dir, yaml_file)

		stream = open(path, "r")
		doc = yaml.load(stream)
		raw_dir =  name + '_RAW'

		for k,s in doc['stems'].items():
				for k, r in s['raw'].items():
						instrument = group_by_instrument.get(  r['instrument']  )
						if instrument == None:
								continue

						wave_filename = r['filename']
						if instrument not in chunks_by_instrument:
								chunks_by_instrument[instrument] = []

						wave_name = os.path.splitext(wave_filename)[0]
						track_dir = os.path.join(medley_chopped_path, name, raw_dir, wave_name)
						for chunk in os.listdir(track_dir):
							chunks_by_instrument[instrument].append(os.path.join(name, raw_dir, wave_name, chunk))

